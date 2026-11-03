"""
preprocess.py
=============
Shared preprocessing pipeline for CEDAR signature images.

This module provides ONE canonical preprocessing function that is used both
during training (inside the PyTorch Dataset) and during inference (in infer.py
and the Streamlit GUI).  Using the same function in both places guarantees that
the model always sees inputs in exactly the same format it was trained on.

Steps applied to every image (train AND inference):
  1. Accept a file path OR a PIL Image.
  2. Convert to grayscale  (handles colour scans and phone photos).
  3. Resize to IMG_HEIGHT x IMG_WIDTH  (default 155 x 220, CEDAR paper size).
  4. Convert pixel values to float32 in [0, 1].
  5. Invert intensities  (CEDAR originals have dark strokes on white background;
     after inversion, strokes are bright on a dark background, which tends to
     work better with CNNs).
  6. Return a PyTorch tensor of shape (1, H, W)  (single channel, no batch dim).

Additional augmentation  (TRAINING ONLY — pass augment=True):
  - Random rotation ±5°
  - Random translate ±5 % of image size
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import random
import math

# ---------------------------------------------------------------------------
# Configurable image dimensions  (change here and everywhere updates at once)
# ---------------------------------------------------------------------------
IMG_HEIGHT = 155   # rows
IMG_WIDTH  = 220   # cols

# Augmentation parameters (training only)
AUG_ROTATION_DEG  = 5      # ± degrees
AUG_TRANSLATE_PCT = 0.05   # fraction of image size in each axis

# In-memory cache for resized grayscale images from disk
_IMAGE_CACHE: dict[tuple[Path, int, int], Image.Image] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image(
    source: Union[str, Path, Image.Image],
    augment: bool = False,
    img_height: int = IMG_HEIGHT,
    img_width:  int = IMG_WIDTH,
) -> torch.Tensor:
    """
    Preprocess a single signature image into a model-ready tensor.

    Parameters
    ----------
    source : str | Path | PIL.Image.Image
        File path (PNG/JPG/JPEG accepted) or an already-opened PIL image.
    augment : bool
        If True, apply random rotation and translation (use during training).
    img_height : int
        Target image height in pixels.
    img_width : int
        Target image width in pixels.

    Returns
    -------
    torch.Tensor
        Shape (1, img_height, img_width), dtype float32, values in [0, 1],
        with foreground strokes bright and background dark.
    """

    # ------------------------------------------------------------------
    # Step 1-3 – Load, convert to grayscale, and resize (cached if path)
    # ------------------------------------------------------------------
    if isinstance(source, (str, Path)):
        path_key = (Path(source).resolve(), img_height, img_width)
        if path_key in _IMAGE_CACHE:
            img = _IMAGE_CACHE[path_key]
        else:
            img = Image.open(source).convert("L")
            img = img.resize((img_width, img_height), Image.BILINEAR)
            _IMAGE_CACHE[path_key] = img
    elif isinstance(source, Image.Image):
        img = source.convert("L")
        img = img.resize((img_width, img_height), Image.BILINEAR)
    else:
        raise TypeError(
            f"source must be a file path or PIL Image, got {type(source)}"
        )

    # ------------------------------------------------------------------
    # Step 4 – Optional augmentation (training only)
    # ------------------------------------------------------------------
    if augment:
        # Random rotation ± AUG_ROTATION_DEG degrees
        angle = random.uniform(-AUG_ROTATION_DEG, AUG_ROTATION_DEG)
        img = TF.rotate(img, angle, fill=255)   # fill with white (paper colour)

        # Random translation ± AUG_TRANSLATE_PCT of image dimensions
        max_dx = int(img_width  * AUG_TRANSLATE_PCT)
        max_dy = int(img_height * AUG_TRANSLATE_PCT)
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        img = TF.affine(img, angle=0, translate=(dx, dy), scale=1.0, shear=0, fill=255)

    # ------------------------------------------------------------------
    # Step 5 – Convert to float32 tensor  →  values in [0, 1]
    # ------------------------------------------------------------------
    # TF.to_tensor handles PIL → (C, H, W) float32 in [0, 1] automatically
    tensor = TF.to_tensor(img)   # shape: (1, H, W)

    # ------------------------------------------------------------------
    # Step 6 – Invert so strokes are bright (foreground ≈ 1, bg ≈ 0)
    # ------------------------------------------------------------------
    tensor = 1.0 - tensor

    return tensor   # shape (1, img_height, img_width)

