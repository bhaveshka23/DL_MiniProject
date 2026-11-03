"""
evaluate.py
===========
Evaluation script for the trained Siamese Signature Verification Network.

What this script does
---------------------
1. Reloads the best checkpoint from models/best_model.pth.
2. Re-builds the test DataLoader using the same writer split saved during
   training (models/test_writer_ids.json).
3. Computes pair distances on the TEST set (writers never seen in training).
4. Finds the optimal decision threshold on the VALIDATION set (the EER point),
   saves it to models/threshold.json.
5. Reports on the test set:
   - Accuracy, FAR (False Accept Rate), FRR (False Reject Rate), EER
   - ROC curve + AUC  → saved to models/roc_curve.png
   - Confusion matrix → saved to models/confusion_matrix.png
   - Distance distribution plot (genuine vs forged) → models/distance_dist.png
   - A grid of example pairs with predicted and true labels → models/example_pairs.png

Vocabulary
----------
  FAR  (False Accept Rate)   = forged pairs accepted as genuine / total forged
  FRR  (False Reject Rate)   = genuine pairs rejected as forged / total genuine
  EER                        = point where FAR ≈ FRR  (lower is better)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import _parse_cedar_files, _build_pairs, SignaturePairDataset
from src.model   import build_siamese_network
from src.preprocess import IMG_HEIGHT, IMG_WIDTH

from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helper: load model from checkpoint
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt   = torch.load(ckpt_path, map_location=device)
    model  = build_siamese_network(backbone="lenet5")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"[Evaluate] Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f})")
    return model


# ---------------------------------------------------------------------------
# Helper: compute distances for an entire DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_distances(
    model:  torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    distances : np.ndarray  shape (N,)
    labels    : np.ndarray  shape (N,)  — 1=genuine, 0=forged
    """
    all_dist  = []
    all_labels = []

    for img1, img2, label in loader:
        img1, img2 = img1.to(device), img2.to(device)
        emb1, emb2 = model(img1, img2)
        dist = F.pairwise_distance(emb1, emb2, p=2).cpu().numpy()
        all_dist.append(dist)
        all_labels.append(label.numpy())

    return np.concatenate(all_dist), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Helper: find EER threshold from distances + labels
# ---------------------------------------------------------------------------

def find_eer_threshold(
    distances: np.ndarray,
    labels:    np.ndarray,
) -> tuple[float, float]:
    """
    Sweep distance thresholds and find the EER point.

    Convention:
      - distances > threshold  →  forged  (label 0)
      - distances ≤ threshold  →  genuine (label 1)

    Returns
    -------
    threshold : float  — distance at EER
    eer       : float  — EER value (0–1)
    """
    thresholds = np.linspace(distances.min(), distances.max(), 500)
    best_diff  = float("inf")
    best_thr   = float(thresholds[0])
    best_eer   = 0.5

    genuine_mask = (labels == 1)
    forged_mask  = (labels == 0)
    n_genuine    = max(int(genuine_mask.sum()), 1)
    n_forged     = max(int(forged_mask.sum()), 1)

    for thr in thresholds:
        pred_genuine = (distances <= thr)
        frr = (pred_genuine[genuine_mask] == 0).sum() / n_genuine  # missed genuine
        far = (pred_genuine[forged_mask]  == 1).sum() / n_forged   # accepted forged

        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_thr  = float(thr)
            best_eer  = float((far + frr) / 2.0)

    return best_thr, best_eer


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_distance_distribution(
    distances: np.ndarray,
    labels:    np.ndarray,
    threshold: float,
    save_path: Path,
) -> None:
    """Histogram of pair distances, split by genuine / forged."""
    fig, ax = plt.subplots(figsize=(9, 5))

    genuine_dist = distances[labels == 1]
    forged_dist  = distances[labels == 0]

    bins = np.linspace(distances.min(), distances.max(), 50)
    ax.hist(genuine_dist, bins=bins, alpha=0.6, label="Genuine pairs", color="#4CAF50")
    ax.hist(forged_dist,  bins=bins, alpha=0.6, label="Forged pairs",  color="#F44336")
    ax.axvline(threshold, color="navy", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Count")
    ax.set_title("Pair Distance Distribution (Test Set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc(
    distances: np.ndarray,
    labels:    np.ndarray,
    save_path: Path,
) -> float:
    """
    Plot the ROC curve and return AUC.

    sklearn's roc_curve expects a 'score' that is high for the positive class.
    Our positive class (genuine) has LOW distance, so we negate distances.
    """
    neg_dist = -distances   # higher = more likely genuine
    fpr, tpr, _ = roc_curve(labels, neg_dist)
    auc         = roc_auc_score(labels, neg_dist)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return float(auc)


def plot_confusion_matrix(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    save_path: Path,
) -> None:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Forged (0)", "Genuine (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_example_pairs(
    distances: np.ndarray,
    labels:    np.ndarray,
    preds:     np.ndarray,
    test_pairs,                  # list of (path1, path2, label)
    save_path: Path,
    n_show:    int = 8,
) -> None:
    """
    Show a grid of example pairs with their true label, predicted label
    and distance.
    """
    import random as rnd
    from PIL import Image as PILImage

    indices = list(range(len(test_pairs)))
    rnd.shuffle(indices)
    indices = indices[:n_show]

    fig, axes = plt.subplots(n_show, 2, figsize=(6, n_show * 2))
    for row, idx in enumerate(indices):
        path1, path2, true_label = test_pairs[idx]
        pred_label  = preds[idx]
        dist        = distances[idx]

        img1 = PILImage.open(path1).convert("L")
        img2 = PILImage.open(path2).convert("L")

        true_str = "Genuine" if true_label == 1 else "Forged"
        pred_str = "Genuine" if pred_label == 1 else "Forged"
        color    = "green" if true_label == pred_label else "red"

        axes[row, 0].imshow(img1, cmap="gray")
        axes[row, 0].axis("off")
        axes[row, 0].set_title("Reference", fontsize=8)

        axes[row, 1].imshow(img2, cmap="gray")
        axes[row, 1].axis("off")
        axes[row, 1].set_title(
            f"True:{true_str} | Pred:{pred_str} | d={dist:.3f}",
            fontsize=7,
            color=color,
        )

    plt.suptitle("Example Pairs (green = correct, red = wrong)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate() -> None:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = ROOT / "models"
    data_root = ROOT / "data"

    # ---- Load model --------------------------------------------------------
    ckpt_path = model_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Please run  python -m src.train  first."
        )
    model = load_model(ckpt_path, device)

    # ---- Reload exact train/val/test split ---------------------------------
    ids_path = model_dir / "test_writer_ids.json"
    if not ids_path.exists():
        raise FileNotFoundError(
            f"test_writer_ids.json not found at {ids_path}. "
            "Please re-run  python -m src.train."
        )
    with open(ids_path) as fh:
        test_ids = json.load(fh)

    # We also need val writer IDs to pick the EER threshold.
    # Re-build the same split from the training seed.
    import random, json as _json
    from src.train import CONFIG

    genuine_files, forged_files = _parse_cedar_files(data_root)
    all_writers = sorted(set(genuine_files.keys()) & set(forged_files.keys()))

    rng = random.Random(CONFIG["seed"])
    writers_shuffled = all_writers.copy()
    rng.shuffle(writers_shuffled)

    train_ids  = writers_shuffled[:CONFIG["train_writers"]]
    val_ids    = writers_shuffled[CONFIG["train_writers"]: CONFIG["train_writers"] + CONFIG["val_writers"]]

    # ---- Build pairs -------------------------------------------------------
    val_pairs  = _build_pairs(val_ids,  genuine_files, forged_files, seed=CONFIG["seed"] + 1)
    test_pairs = _build_pairs(test_ids, genuine_files, forged_files, seed=CONFIG["seed"] + 2)

    val_ds  = SignaturePairDataset(val_pairs,  augment=False)
    test_ds = SignaturePairDataset(test_pairs, augment=False)

    val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ---- Compute distances (val → threshold selection) ---------------------
    print("[Evaluate] Computing validation distances ...")
    val_distances, val_labels = compute_distances(model, val_loader, device)

    threshold, eer_val = find_eer_threshold(val_distances, val_labels)
    print(f"[Evaluate] EER threshold (val) = {threshold:.4f}  |  EER = {eer_val:.4f}")

    # Save threshold
    thr_path = model_dir / "threshold.json"
    with open(thr_path, "w") as fh:
        json.dump({"threshold": threshold, "eer_val": eer_val}, fh, indent=2)
    print(f"[Evaluate] Threshold saved to {thr_path}")

    # ---- Compute distances (test set) -------------------------------------
    print("[Evaluate] Computing test distances ...")
    test_distances, test_labels = compute_distances(model, test_loader, device)

    # Classify using saved threshold (dist ≤ thr → genuine)
    test_preds = (test_distances <= threshold).astype(int)

    # ---- Metrics -----------------------------------------------------------
    acc = accuracy_score(test_labels, test_preds)

    genuine_mask = test_labels == 1
    forged_mask  = test_labels == 0

    frr = (test_preds[genuine_mask] == 0).sum() / max(genuine_mask.sum(), 1)
    far = (test_preds[forged_mask]  == 1).sum() / max(forged_mask.sum(),  1)
    eer_test = (far + frr) / 2

    auc = plot_roc(test_distances, test_labels, model_dir / "roc_curve.png")

    print("\n" + "=" * 50)
    print("  TEST SET RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {acc  * 100:.2f} %")
    print(f"  FAR       : {far  * 100:.2f} %")
    print(f"  FRR       : {frr  * 100:.2f} %")
    print(f"  EER (test): {eer_test * 100:.2f} %")
    print(f"  AUC       : {auc:.4f}")
    print("=" * 50 + "\n")

    # ---- Plots -------------------------------------------------------------
    plot_distance_distribution(
        test_distances, test_labels, threshold,
        model_dir / "distance_dist.png"
    )
    print("[Evaluate] Distance distribution saved.")

    plot_confusion_matrix(test_labels, test_preds, model_dir / "confusion_matrix.png")
    print("[Evaluate] Confusion matrix saved.")

    plot_example_pairs(
        test_distances, test_labels, test_preds,
        test_pairs,
        model_dir / "example_pairs.png",
        n_show=8,
    )
    print("[Evaluate] Example pairs grid saved.")

    print(f"\n[Evaluate] All plots saved in {model_dir}/")
    print("[Evaluate] Done.  You can now run:  streamlit run app.py")


if __name__ == "__main__":
    evaluate()

