#!/usr/bin/env python3
"""Verify all embedding outputs."""
import numpy as np
from pathlib import Path

emb_dir = Path("/mnt/nas-ai-models/training-data/leica-look/embeddings")
models = sorted([d.name for d in emb_dir.iterdir() if d.is_dir()])

print(f"{'Model':<20} {'File':<15} {'Shape':<15} {'NaN':<8} {'Inf':<8}")
print("-" * 70)

for model in models:
    for f in sorted((emb_dir / model).glob("*.npy")):
        d = np.load(f)
        n_nan = int(np.isnan(d).sum())
        n_inf = int(np.isinf(d).sum())
        print(f"{model:<20} {f.name:<15} {str(d.shape):<15} {n_nan:<8} {n_inf:<8}")

labels = np.load(emb_dir / "labels.npy")
n_lines = len((emb_dir / "image_ids.txt").read_text().splitlines())
print(f"\nLabels: {labels.shape}, pos={int((labels==1).sum())}, neg={int((labels==0).sum())}")
print(f"Image IDs: {n_lines} lines")
print(f"Total embedding files: {sum(1 for m in models for _ in (emb_dir/m).glob('*.npy'))}")
