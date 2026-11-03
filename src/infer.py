"""
infer.py
========
Inference module for the Siamese Signature Verification Network.

Public API
----------
  from src.infer import verify

  result = verify("path/to/reference.png", "path/to/query.png")
  # result = {
  #     "distance":   0.312,
  #     "verdict":    "Genuine",
  #     "confidence": 87.4,   # percentage
  # }

Design notes
------------
- The model and threshold are loaded ONCE (on first call) and cached in module-
  level globals.  Subsequent calls skip the loading step entirely, which is
  important for the Streamlit app where verify() is called many times.

- `verify()` accepts both file paths and already-opened PIL images, so the
  Streamlit app can pass Streamlit's uploaded-file objects after converting
  them with PIL.Image.open().

Confidence mapping
------------------
We want:
  - distance << threshold  →  high confidence GENUINE
  - distance ≈  threshold  →  ~50 % confidence  (uncertain)
  - distance >> threshold  →  high confidence FORGED

A sigmoid centred at the threshold does exactly this:

  raw_score     = (threshold - distance) / scale   (positive when genuine)
  sigmoid_score = 1 / (1 + exp(-raw_score))         (in [0, 1])

When distance < threshold: raw_score > 0 → sigmoid > 0.5 → "Genuine"
When distance > threshold: raw_score < 0 → sigmoid < 0.5 → "Forged"

confidence % = sigmoid_score × 100  for genuine
             = (1 - sigmoid_score) × 100  for forged

`scale` controls how quickly confidence grows away from the threshold.
A good default is threshold / 4 (roughly one quarter of the threshold range).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image

# ── Project imports ─────────────────────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model      import build_siamese_network
from src.preprocess import preprocess_image

# ---------------------------------------------------------------------------
# Module-level cache (populated on first call to verify())
# ---------------------------------------------------------------------------
_model:     torch.nn.Module | None = None
_threshold: float | None           = None
_device:    torch.device | None    = None

MODEL_PATH     = ROOT / "models" / "best_model.pth"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

# Confidence scale factor  (see module docstring)
CONFIDENCE_SCALE = 0.15   # tunable: smaller → steeper sigmoid curve


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_resources() -> None:
    """Load model and threshold into module-level cache (runs once)."""
    global _model, _threshold, _device

    if _model is not None:
        return   # already loaded

    # ---- Device ------------------------------------------------------------
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model -------------------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Please run  python -m src.train  to train the model first."
        )

    ckpt   = torch.load(MODEL_PATH, map_location=_device)
    model  = build_siamese_network(backbone="lenet5")
    model.load_state_dict(ckpt["model_state"])
    model.to(_device)
    model.eval()
    _model = model

    # ---- Threshold ---------------------------------------------------------
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}\n"
            "Please run  python -m src.evaluate  to generate the threshold."
        )

    with open(THRESHOLD_PATH) as fh:
        data = json.load(fh)
    _threshold = float(data["threshold"])

    print(
        f"[Infer] Model loaded from {MODEL_PATH.name}  "
        f"| threshold = {_threshold:.4f}"
    )


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_model_ready() -> bool:
    """Return True if both the model checkpoint and threshold file exist."""
    return MODEL_PATH.exists() and THRESHOLD_PATH.exists()


def verify(
    img1: Union[str, Path, Image.Image],
    img2: Union[str, Path, Image.Image],
    threshold_override: float | None = None,
) -> dict:
    """
    Compare two signature images and return a verification result.

    Parameters
    ----------
    img1 : str | Path | PIL.Image.Image
        Reference signature (known genuine).
    img2 : str | Path | PIL.Image.Image
        Signature to verify.
    threshold_override : float | None
        If provided, use this distance threshold instead of the saved EER
        threshold.  Useful for the GUI's threshold slider.

    Returns
    -------
    dict with keys:
        "distance"   : float  — Euclidean distance between the embeddings
        "verdict"    : str    — "Genuine" or "Forged"
        "confidence" : float  — Confidence percentage (0–100)
    """
    # ---- Load resources on first call --------------------------------------
    _load_resources()

    thr = threshold_override if threshold_override is not None else _threshold

    # ---- Preprocess images -------------------------------------------------
    # augment=False for inference — we want deterministic results
    t1 = preprocess_image(img1, augment=False).unsqueeze(0).to(_device)  # (1,1,H,W)
    t2 = preprocess_image(img2, augment=False).unsqueeze(0).to(_device)  # (1,1,H,W)

    # ---- Forward pass ------------------------------------------------------
    with torch.no_grad():
        emb1, emb2 = _model(t1, t2)

    distance = F.pairwise_distance(emb1, emb2, p=2).item()

    # ---- Decision rule -----------------------------------------------------
    is_genuine = distance <= thr

    # Confidence via sigmoid centred at threshold
    # Scale proportional to threshold so near-zero distance yields ~95%+ genuine confidence
    scale        = max(thr / 3.0, 0.005)
    raw_score    = (thr - distance) / scale             # + when genuine, - when forged
    sigmoid_val  = _sigmoid(raw_score)

    if is_genuine:
        confidence = sigmoid_val * 100          # high sigmoid → high genuine confidence
        verdict    = "Genuine"
    else:
        confidence = (1.0 - sigmoid_val) * 100  # low sigmoid → high forged confidence
        verdict    = "Forged"

    return {
        "distance":   round(distance,   4),
        "verdict":    verdict,
        "confidence": round(confidence, 2),
    }

