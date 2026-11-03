"""
loss.py
=======
Contrastive Loss for Siamese Networks.

Formula (Hadsell, Chopra, LeCun 2006)
--------------------------------------
  L = y       * 0.5 * d²
    + (1 - y) * 0.5 * max(0, margin - d)²

where
  y = label (1 = genuine / same, 0 = forged / different)
  d = Euclidean distance between the two embeddings

Intuition
---------
- When y = 1  (genuine pair):   loss = 0.5 * d²
    → the network is penalised if the embeddings are far apart.
    → it is pushed to bring genuine pairs CLOSE together.

- When y = 0  (forged pair):    loss = 0.5 * max(0, margin - d)²
    → there is NO penalty once the distance already exceeds `margin`.
    → below the margin the network is pushed to push forged pairs APART.

After training the decision rule is simply:
  d ≤ threshold  →  Genuine
  d >  threshold →  Forged

The threshold is tuned on the validation set (EER point) in evaluate.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (mean over the batch).

    Parameters
    ----------
    margin : float
        Minimum distance expected between embeddings of genuine/forged pairs
        (forged pairs beyond this distance contribute zero loss).
        Typical values: 1.0 – 2.0 when embeddings are L2-normalised.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1:  torch.Tensor,
        emb2:  torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Parameters
        ----------
        emb1, emb2 : torch.Tensor
            Shape (batch, embedding_dim).  Must be L2-normalised embeddings
            from the Siamese network branches.
        label : torch.Tensor
            Shape (batch,), float32.
            1  =  genuine / same-writer pair
            0  =  forged  / different-class pair

        Returns
        -------
        torch.Tensor
            Scalar loss value (mean over the batch).
        """
        # Euclidean distance between embedding pairs — shape (batch,)
        distance = F.pairwise_distance(emb1, emb2, p=2, keepdim=False)

        # Contrastive loss components
        # Genuine pairs (label = 1): penalise large distances
        loss_genuine = label * 0.5 * distance.pow(2)

        # Forged pairs (label = 0): penalise distances smaller than margin
        loss_forged  = (1.0 - label) * 0.5 * F.relu(self.margin - distance).pow(2)

        loss = loss_genuine + loss_forged
        return loss.mean()

