"""
model.py
========
Siamese Neural Network with a swappable CNN backbone.

Architecture overview
---------------------
                 ┌─────────────────┐
  image_1 ──────►│                 ├──► embedding_1 ──┐
                 │  Shared branch  │                   ├──► (emb1, emb2)
  image_2 ──────►│  (LeNet-5 CNN)  ├──► embedding_2 ──┘
                 └─────────────────┘

The SAME branch (with shared weights) processes both images.
The loss function (contrastive loss in loss.py) works on the Euclidean
distance between embedding_1 and embedding_2.

Backbone: LeNet-5 adapted for 1-channel input
----------------------------------------------
  Conv(1→6,  5×5) → ReLU → MaxPool(2×2)
  Conv(6→16, 5×5) → ReLU → MaxPool(2×2)
  Flatten
  Linear(flat_size → 256) → ReLU
  Linear(256 → 128)   ← 128-dimensional embedding

The flatten size is computed DYNAMICALLY (dummy forward pass) so that
changing IMG_HEIGHT / IMG_WIDTH in preprocess.py automatically works here.

Swapping backbones
------------------
Call `build_siamese_network(backbone="resnet18")` to swap to a pretrained
ResNet-18 branch (the final FC is replaced to output 128-d embeddings).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

from src.preprocess import IMG_HEIGHT, IMG_WIDTH


# ---------------------------------------------------------------------------
# LeNet-5 backbone (1-channel input → 128-d embedding)
# ---------------------------------------------------------------------------

class LeNet5Branch(nn.Module):
    """
    A LeNet-5–style CNN that maps a (1, H, W) signature image to a
    128-dimensional embedding vector.

    The flatten size after the convolutional layers is computed automatically
    so the class works for any input resolution without manual calculation.
    """

    def __init__(self, img_height: int = IMG_HEIGHT, img_width: int = IMG_WIDTH) -> None:
        super().__init__()

        # ---- Convolutional feature extractor --------------------------------
        self.conv_block = nn.Sequential(
            # Block 1: 1 → 6 feature maps, 5×5 kernel, then pool
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 6 → 16 feature maps, 5×5 kernel, then pool
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ---- Dynamically compute the flat size after conv_block --------------
        # Feed a dummy zero tensor through conv_block to see the output shape.
        with torch.no_grad():
            dummy    = torch.zeros(1, 1, img_height, img_width)
            conv_out = self.conv_block(dummy)            # (1, 16, h', w')
            flat_size = conv_out.numel()                 # = 16 * h' * w'

        # ---- Fully-connected classifier → embedding -------------------------
        self.fc_block = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),                         # 128-d embedding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 1, H, W)

        Returns
        -------
        torch.Tensor
            Shape (batch, 128)  — L2-normalised embedding.
        """
        features  = self.conv_block(x)                   # (B, 16, h', w')
        flat      = features.view(features.size(0), -1)  # (B, flat_size)
        embedding = self.fc_block(flat)                  # (B, 128)

        # L2-normalise so distances live on a unit hypersphere;
        # makes the contrastive margin more interpretable.
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


# ---------------------------------------------------------------------------
# ResNet-18 backbone (optional swap-in)
# ---------------------------------------------------------------------------

class ResNet18Branch(nn.Module):
    """
    Pretrained ResNet-18 with the final FC replaced to output a 128-d embedding.
    Input is expected to be 1-channel; the first conv is patched accordingly.
    """

    def __init__(self) -> None:
        super().__init__()

        base = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)

        # Patch the input conv to accept 1-channel greyscale instead of 3-channel
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace the classification head with an embedding head
        in_features   = base.fc.in_features       # 512 for ResNet-18
        base.fc       = nn.Linear(in_features, 128)

        self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(x)
        return nn.functional.normalize(embedding, p=2, dim=1)


# ---------------------------------------------------------------------------
# Siamese Network wrapper
# ---------------------------------------------------------------------------

class SiameseNetwork(nn.Module):
    """
    Wraps a single branch (shared weights) into a Siamese Network.

    Both images are passed through THE SAME branch object, so the weights
    are automatically shared (PyTorch shares parameters when you call
    the same module twice).

    Parameters
    ----------
    branch : nn.Module
        The backbone module (LeNet5Branch or ResNet18Branch).
    """

    def __init__(self, branch: nn.Module) -> None:
        super().__init__()
        self.branch = branch

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        img1, img2 : torch.Tensor
            Each of shape (batch, 1, H, W)

        Returns
        -------
        emb1, emb2 : torch.Tensor
            Each of shape (batch, 128)
        """
        emb1 = self.branch(img1)
        emb2 = self.branch(img2)
        return emb1, emb2


# ---------------------------------------------------------------------------
# Factory function — swap backbones here
# ---------------------------------------------------------------------------

def build_siamese_network(
    backbone: str = "lenet5",
    img_height: int = IMG_HEIGHT,
    img_width:  int = IMG_WIDTH,
) -> SiameseNetwork:
    """
    Build and return a SiameseNetwork with the specified backbone.

    Parameters
    ----------
    backbone : str
        "lenet5"  → LeNet-5 branch  (default, trained from scratch).
        "resnet18" → ResNet-18 branch (pretrained ImageNet weights).
    img_height, img_width : int
        Used only for LeNet-5 to compute the flatten size.

    Returns
    -------
    SiameseNetwork
    """
    if backbone == "lenet5":
        branch = LeNet5Branch(img_height=img_height, img_width=img_width)
    elif backbone == "resnet18":
        branch = ResNet18Branch()
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose 'lenet5' or 'resnet18'.")

    return SiameseNetwork(branch)

