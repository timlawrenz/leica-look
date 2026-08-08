#!/usr/bin/env python3
"""
Create and manage the held-out Leica split for Phase 2a gate hardening.

Fix #2: the gate currently compares to the Leica centroid computed from ALL
270+ Leica images — the TRAINING SET itself. A LoRA that memorizes the images
scores well without learning a real 'look.'

This module:
  1. Creates a deterministic train/holdout split (seed=42)
  2. Writes held_out_split.json with image IDs
  3. Provides load_split() for training/evaluation scripts

Reserves ~35 images (~13%) for held-out. Training uses ~230 images.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Tuple, Set, Optional

import numpy as np

PROJECT_ROOT = Path("/home/tim/source/activity/leica-look")
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "lora-transfer"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "verified.csv"
SPLIT_PATH = EXPERIMENT_DIR / "held_out_split.json"


def create_split(
    csv_path: Path = REGISTRY_PATH,
    output_path: Path = SPLIT_PATH,
    n_held_out: int = 35,
    seed: int = 42,
) -> dict:
    """
    Create a deterministic train/holdout split from verified Leica images.

    Args:
        csv_path: Path to verified.csv
        output_path: Where to write the split JSON
        n_held_out: Number of Leica images to reserve (default 35)
        seed: Random seed for reproducibility

    Returns:
        dict with 'train_ids', 'holdout_ids', metadata
    """
    # Load all Leica image IDs
    leica_ids = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["class"] == "positive":
                fid = row["flickr_id"]
                file_path = row.get("file_path", "")
                if file_path and Path(file_path).exists():
                    leica_ids.append(fid)

    n_total = len(leica_ids)
    if n_total < n_held_out + 10:
        raise ValueError(
            f"Not enough Leica images: {n_total} total, need at least {n_held_out + 10}"
        )

    # Deterministic shuffle
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    held_out_indices = set(indices[:n_held_out].tolist())
    train_indices = set(indices[n_held_out:].tolist())

    train_ids = [leica_ids[i] for i in sorted(train_indices)]
    holdout_ids = [leica_ids[i] for i in sorted(held_out_indices)]

    split = {
        "created": "2026-08-08",
        "n_total": n_total,
        "n_train": len(train_ids),
        "n_holdout": len(holdout_ids),
        "seed": seed,
        "description": (
            f"Deterministic train/holdout split (seed={seed}). "
            f"Training uses {len(train_ids)} images; held-out evaluation uses {len(holdout_ids)}. "
            f"Gate must confirm shift toward held-out centroid (generalization, not memorization)."
        ),
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
    }

    with open(output_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Split created: {len(train_ids)} train + {len(holdout_ids)} holdout")
    print(f"Written to: {output_path}")
    return split


def load_split(split_path: Path = SPLIT_PATH) -> dict:
    """Load the train/holdout split."""
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Run create_split() first."
        )
    with open(split_path) as f:
        return json.load(f)


def get_train_ids(split: Optional[dict] = None) -> Set[str]:
    """Get the set of training image flickr_ids."""
    if split is None:
        split = load_split()
    return set(split["train_ids"])


def get_holdout_ids(split: Optional[dict] = None) -> Set[str]:
    """Get the set of held-out image flickr_ids."""
    if split is None:
        split = load_split()
    return set(split["holdout_ids"])


if __name__ == "__main__":
    # Create the split
    if SPLIT_PATH.exists():
        print(f"Split already exists: {SPLIT_PATH}")
        existing = load_split()
        print(f"  {existing['n_train']} train + {existing['n_holdout']} holdout")
        print("  Delete the file to regenerate, or load it directly.")
    else:
        create_split()
