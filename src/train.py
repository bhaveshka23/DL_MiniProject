"""
train.py
========
Training script for the Siamese Signature Verification Network.

How to run
----------
  python -m src.train          (from the signature-verification/ folder)
  -- or --
  cd signature-verification && python -m src.train

All hyperparameters live in the CONFIG section below.  Change them there and
they propagate to every part of the training pipeline automatically.

What this script does
---------------------
1. Loads the CEDAR dataset and builds train / val / test DataLoaders.
2. Builds a SiameseNetwork (LeNet-5 by default).
3. Trains for CONFIG["epochs"] epochs:
   - Forward pass through both branches.
   - Compute contrastive loss.
   - Backprop + Adam step.
   - Log train & val loss per epoch.
   - Save the best checkpoint (lowest val loss) to models/best_model.pth.
4. Plots and saves the loss curves to models/loss_curve.png.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")        # non-interactive backend (safe on headless servers)
import matplotlib.pyplot as plt

# ── Make sure src/ is importable regardless of where we run from ────────────
ROOT = Path(__file__).resolve().parent.parent   # …/signature-verification/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import get_dataloaders
from src.model   import build_siamese_network
from src.loss    import ContrastiveLoss

# ===========================================================================
# CONFIG  ←  all hyperparameters live here
# ===========================================================================
CONFIG = {
    # --- Paths ---------------------------------------------------------------
    "data_root":      ROOT / "data",
    "model_dir":      ROOT / "models",

    # --- Dataset -------------------------------------------------------------
    "train_writers":  35,
    "val_writers":    10,
    "test_writers":   10,
    "seed":           42,

    # --- Model ---------------------------------------------------------------
    "backbone":       "lenet5",   # "lenet5" | "resnet18"

    # --- Loss ----------------------------------------------------------------
    "margin":         1.0,        # contrastive loss margin

    # --- Optimiser -----------------------------------------------------------
    "lr":             1e-3,
    "weight_decay":   1e-4,

    # --- Training loop -------------------------------------------------------
    "epochs":         20,
    "batch_size":     32,
    "num_workers":    0,          # set > 0 if you have multiple CPU cores
}
# ===========================================================================


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
) -> float:
    """
    Run one epoch (train or validation).

    If optimizer is None the function runs in evaluation mode (no backprop).
    Returns the mean loss over all batches.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for img1, img2, labels in loader:
            img1   = img1.to(device)
            img2   = img2.to(device)
            labels = labels.to(device)

            emb1, emb2 = model(img1, img2)
            loss       = criterion(emb1, emb2, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def plot_loss_curves(
    train_losses: list[float],
    val_losses:   list[float],
    save_path:    Path,
) -> None:
    """Save a clean loss-curve plot."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss",      color="#2196F3", linewidth=2)
    ax.plot(epochs, val_losses,   label="Validation loss", color="#FF5722", linewidth=2,
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Train] Loss curve saved to {save_path}")


def train() -> None:
    """Main training function."""
    set_seed(CONFIG["seed"])

    # ---- Device ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ---- Output directory --------------------------------------------------
    model_dir = Path(CONFIG["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- DataLoaders -------------------------------------------------------
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        data_root    = CONFIG["data_root"],
        batch_size   = CONFIG["batch_size"],
        num_workers  = CONFIG["num_workers"],
        train_writers= CONFIG["train_writers"],
        val_writers  = CONFIG["val_writers"],
        test_writers = CONFIG["test_writers"],
        seed         = CONFIG["seed"],
    )

    # Save test writer IDs so evaluate.py can reload them without re-splitting
    test_ids_path = model_dir / "test_writer_ids.json"
    with open(test_ids_path, "w") as fh:
        json.dump(test_ids, fh)
    print(f"[Train] Test writer IDs saved to {test_ids_path}")

    # ---- Model, loss, optimiser --------------------------------------------
    model     = build_siamese_network(backbone=CONFIG["backbone"]).to(device)
    criterion = ContrastiveLoss(margin=CONFIG["margin"])
    optimizer = optim.Adam(
        model.parameters(),
        lr           = CONFIG["lr"],
        weight_decay = CONFIG["weight_decay"],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable parameters: {total_params:,}")

    # ---- Training loop -----------------------------------------------------
    best_val_loss  = float("inf")
    train_losses   = []
    val_losses     = []
    best_ckpt_path = model_dir / "best_model.pth"

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = run_epoch(model, val_loader,   criterion, None,      device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch:3d}/{CONFIG['epochs']}]  "
            f"Train Loss: {train_loss:.4f}  |  "
            f"Val Loss:   {val_loss:.4f}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                    "config":      {k: str(v) for k, v in CONFIG.items()},
                },
                best_ckpt_path,
            )
            print(f"           [OK] Best model saved (val_loss={val_loss:.4f})")

    print(f"\n[Train] Training complete.  Best val loss: {best_val_loss:.4f}")

    # ---- Plot loss curves --------------------------------------------------
    plot_loss_curves(train_losses, val_losses, model_dir / "loss_curve.png")

    print("\n[Train] Next step: run  python -m src.evaluate  to evaluate on test set.")


if __name__ == "__main__":
    train()

