"""
dataset.py
==========
CEDAR Signature Dataset — pair generation, splitting by writer, DataLoaders.

CEDAR layout on disk
--------------------
  data/
    full_org/   – genuine signatures
                  original_<writer_id>_<sample_id>.png
                  e.g.  original_1_1.png  …  original_55_24.png
    full_forg/  – forged signatures
                  forgeries_<writer_id>_<sample_id>.png
                  e.g.  forgeries_1_1.png  …  forgeries_55_24.png

Pair strategy
-------------
- POSITIVE pair (label = 1): two different genuine signatures of the same writer.
- NEGATIVE pair (label = 0): one genuine + one forged signature of the same writer.
- Pairs are balanced 50 / 50.
- Writers are split into train / val / test at the writer level so no test
  writer's signature is ever seen during training.

Default split: 35 train, 10 validation, 10 test writers (out of 55).
"""

from __future__ import annotations

import os
import re
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocess import preprocess_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_cedar_files(data_root: str | Path) -> Tuple[Dict, Dict]:
    """
    Walk full_org/ and full_forg/ and return two dicts:
        genuine_files[writer_id] = [Path, Path, …]   (24 paths)
        forged_files[writer_id]  = [Path, Path, …]   (24 paths)

    writer_id is an integer (1-55).
    """
    data_root   = Path(data_root)
    # Support both data/full_org and data/signatures/full_org
    if not (data_root / "full_org").exists() and (data_root / "signatures" / "full_org").exists():
        data_root = data_root / "signatures"

    org_dir     = data_root / "full_org"
    forg_dir    = data_root / "full_forg"

    if not org_dir.exists() or not forg_dir.exists():
        raise FileNotFoundError(
            f"CEDAR dataset folders not found at '{data_root}'.\n"
            f"Expected '{org_dir}' and '{forg_dir}'.\n"
            "Please extract the CEDAR dataset so that full_org/ and full_forg/ are inside data/."
        )

    # Regex patterns to extract writer and sample IDs
    genuine_pat = re.compile(r"original_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
    forged_pat  = re.compile(r"forgeries_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)

    genuine_files: Dict[int, List[Path]] = defaultdict(list)
    forged_files:  Dict[int, List[Path]] = defaultdict(list)

    # Parse genuine signatures
    for f in sorted(org_dir.iterdir()):
        m = genuine_pat.match(f.name)
        if m:
            writer_id = int(m.group(1))
            genuine_files[writer_id].append(f)

    # Parse forged signatures
    for f in sorted(forg_dir.iterdir()):
        m = forged_pat.match(f.name)
        if m:
            writer_id = int(m.group(1))
            forged_files[writer_id].append(f)

    return dict(genuine_files), dict(forged_files)


def _build_pairs(
    writer_ids: List[int],
    genuine_files: Dict[int, List[Path]],
    forged_files:  Dict[int, List[Path]],
    seed: int = 42,
) -> List[Tuple[Path, Path, int]]:
    """
    Build a balanced list of (img1_path, img2_path, label) pairs for the
    given writer_ids.

    Returns
    -------
    List of tuples (path1, path2, label)  where label ∈ {0, 1}.
    """
    rng = random.Random(seed)
    pairs: List[Tuple[Path, Path, int]] = []

    for wid in writer_ids:
        gfiles = genuine_files.get(wid, [])
        ffiles  = forged_files.get(wid, [])

        if len(gfiles) < 2:
            continue   # need at least 2 genuine to make a positive pair

        # ---- Positive pairs (two genuine images, same writer) -------------
        # Generate all C(n, 2) genuine combinations, then cap to avoid
        # a huge imbalance with negative pairs.
        genuine_combos = [
            (gfiles[i], gfiles[j])
            for i in range(len(gfiles))
            for j in range(i + 1, len(gfiles))
        ]
        rng.shuffle(genuine_combos)

        # Number of negative pairs = min(len(gfiles), len(ffiles)) combos
        n_neg = min(len(gfiles), len(ffiles))
        # Cap positive pairs to the same count for balance
        pos_pairs = genuine_combos[:n_neg]

        for g1, g2 in pos_pairs:
            pairs.append((g1, g2, 1))   # label 1 = genuine / same

        # ---- Negative pairs (genuine + forged, same writer) ---------------
        # Pair each genuine with one forged (shuffle both lists first)
        g_shuffle = gfiles.copy(); rng.shuffle(g_shuffle)
        f_shuffle  = ffiles.copy();  rng.shuffle(f_shuffle)

        for g, f in zip(g_shuffle[:n_neg], f_shuffle[:n_neg]):
            pairs.append((g, f, 0))    # label 0 = forged / different

    rng.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SignaturePairDataset(Dataset):
    """
    PyTorch Dataset that returns (img1_tensor, img2_tensor, label) triples.

    Parameters
    ----------
    pairs : list of (Path, Path, int)
        Pre-built pairs from _build_pairs().
    augment : bool
        Pass True for the training set to enable data augmentation.
    """

    def __init__(
        self,
        pairs: List[Tuple[Path, Path, int]],
        augment: bool = False,
    ) -> None:
        self.pairs   = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path1, path2, label = self.pairs[idx]

        img1 = preprocess_image(path1, augment=self.augment)
        img2 = preprocess_image(path2, augment=self.augment)

        # Return label as a float tensor so contrastive loss works without
        # an extra cast step.
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img1, img2, label_tensor


# ---------------------------------------------------------------------------
# Public factory — build train / val / test DataLoaders in one call
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root:    str | Path,
    batch_size:   int  = 32,
    num_workers:  int  = 0,
    train_writers: int = 35,
    val_writers:   int = 10,
    test_writers:  int = 10,
    seed:          int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[int]]:
    """
    Parse the CEDAR dataset and return (train_loader, val_loader, test_loader,
    test_writer_ids).

    The writer split is performed HERE and is deterministic (controlled by
    `seed`) so that the test writers are never exposed during training.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    test_writer_ids : list[int]
        Writer IDs in the test split (used by evaluate.py for visualisation).
    """
    genuine_files, forged_files = _parse_cedar_files(data_root)

    # Use only writers that appear in BOTH genuine and forged directories
    all_writers = sorted(set(genuine_files.keys()) & set(forged_files.keys()))

    if len(all_writers) < train_writers + val_writers + test_writers:
        raise ValueError(
            f"Only {len(all_writers)} writers found; requested "
            f"{train_writers + val_writers + test_writers}. "
            "Check your data directory."
        )

    rng = random.Random(seed)
    writers_shuffled = all_writers.copy()
    rng.shuffle(writers_shuffled)

    train_ids = writers_shuffled[:train_writers]
    val_ids   = writers_shuffled[train_writers : train_writers + val_writers]
    test_ids  = writers_shuffled[train_writers + val_writers : train_writers + val_writers + test_writers]

    print(f"[Dataset] Writers - train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}")

    # Build pair lists (deterministic via seed)
    train_pairs = _build_pairs(train_ids, genuine_files, forged_files, seed=seed)
    val_pairs   = _build_pairs(val_ids,   genuine_files, forged_files, seed=seed + 1)
    test_pairs  = _build_pairs(test_ids,  genuine_files, forged_files, seed=seed + 2)

    print(
        f"[Dataset] Pairs - train: {len(train_pairs)} | "
        f"val: {len(val_pairs)} | test: {len(test_pairs)}"
    )

    train_ds = SignaturePairDataset(train_pairs, augment=True)
    val_ds   = SignaturePairDataset(val_pairs,   augment=False)
    test_ds  = SignaturePairDataset(test_pairs,  augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, test_ids

