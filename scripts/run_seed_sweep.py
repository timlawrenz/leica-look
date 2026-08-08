#!/usr/bin/env python3
"""
Seed sweep — Issue #10
======================
Re-run top-10 LR configurations with 5 non-42 seeds to measure AUC variance.
Output: experiments/discriminator-seed-sweep/results.csv
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_logistic_probes import (
    load_metadata,
    stratify_groups,
    run_split,
    EMBEDDINGS_DIR,
)

# ───────────────────────────────────────────────────────
# Top-10 LR configs from Phase 1 (by AUC, unique model/pooling/n)
# ───────────────────────────────────────────────────────
TOP_10_CONFIGS = [
    # (model, pooling, n_per_class, C, original_auc)
    ("clip-vitl14",    "multicrop",  50,  10.0, 0.9867),
    ("clip-vitl14",    "cls",        50,  10.0, 0.9778),
    ("clip-vitl14",    "cls_patch",  250, 10.0, 0.9721),
    ("clip-vitl14",    "multicrop",  250, 0.1,  0.9669),
    ("clip-vitl14",    "cls",        250, 0.1,  0.9651),
    ("clip-vitl14",    "cls_patch",  50,  1.0,  0.9644),
    ("clip-vitl14",    "patch_mean", 250, 1.0,  0.9545),
    ("clip-vitl14",    "patch_gem",  250, 1.0,  0.9429),
    ("siglip-so400m",  "cls_patch",  50,  1.0,  0.9378),
    ("clip-vitl14",    "patch_max",  250, 10.0, 0.9292),
]

SEEDS = [7, 17, 23, 99, 1234]

OUTPUT_DIR = REPO_ROOT / "experiments/discriminator-seed-sweep"
OUTPUT_CSV = OUTPUT_DIR / "results.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata once
    flickr_ids, scene_types, bodies, labels = load_metadata()
    groups = stratify_groups(scene_types, bodies)
    print(f"Loaded {len(labels)} images ({labels.sum()} pos, {(labels==0).sum()} neg)")

    fieldnames = [
        "model", "pooling", "dataset_size", "C", "seed",
        "auc", "test_pos", "test_neg", "embedding_dim",
        "original_auc", "runtime_sec",
    ]

    # Track existing results for resumability
    existing = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                existing.add((
                    row["model"], row["pooling"], row["dataset_size"],
                    row["C"], row["seed"],
                ))
        print(f"Found {len(existing)} existing results — will skip")

    total = len(TOP_10_CONFIGS) * len(SEEDS)
    completed = len(existing)

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()

        for model_name, pool_name, n_per_class, C_target, original_auc in TOP_10_CONFIGS:
            emb_path = EMBEDDINGS_DIR / model_name / f"{pool_name}.npy"
            if not emb_path.exists():
                print(f"  ⚠️  Missing embeddings: {emb_path}")
                continue

            t0 = time.time()
            X = np.load(emb_path).astype(np.float32)
            embedding_dim = X.shape[1]

            print(f"\n{model_name}/{pool_name} n={n_per_class} C={C_target} "
                  f"(original AUC={original_auc:.4f}, dim={embedding_dim})")

            for seed in SEEDS:
                config_key = (
                    model_name, pool_name, str(n_per_class),
                    str(C_target), str(seed),
                )
                if config_key in existing:
                    print(f"  seed={seed:>4}  SKIP (already done)")
                    continue

                split_info = run_split(X, labels, groups, n_per_class, seed)
                test_pos = split_info["test_pos"]
                test_neg = split_info["test_neg"]

                # Extract the AUC for our specific C value
                auc_val = split_info["results"].get(C_target, "ERROR")

                row = {
                    "model": model_name,
                    "pooling": pool_name,
                    "dataset_size": str(n_per_class),
                    "C": str(C_target),
                    "seed": str(seed),
                    "auc": auc_val if isinstance(auc_val, float) else "",
                    "test_pos": test_pos,
                    "test_neg": test_neg,
                    "embedding_dim": embedding_dim,
                    "original_auc": original_auc,
                    "runtime_sec": round(time.time() - t0, 2),
                }
                writer.writerow(row)
                f.flush()
                completed += 1
                print(f"  seed={seed:>4}  AUC={auc_val!s:>10}  "
                      f"test=({test_pos}p+{test_neg}n)")

    print(f"\n{'='*60}")
    print(f"Done. {completed}/{total} seed-sweep evaluations complete.")
    print(f"Results: {OUTPUT_CSV}")

    # ── Summary statistics ──
    if OUTPUT_CSV.exists():
        print(f"\n{'='*60}")
        print("SUMMARY: Mean ± Std AUC across seeds")
        print(f"{'='*60}")
        from collections import defaultdict
        seed_data = defaultdict(list)

        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                if row["auc"] and row["auc"] != "ERROR":
                    key = (row["model"], row["pooling"], row["dataset_size"], row["C"])
                    seed_data[key].append(float(row["auc"]))

        flagged = []
        for (model, pool, n, C), aucs in sorted(seed_data.items()):
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            original = TOP_10_CONFIGS_LOOKUP.get((model, pool, int(n), float(C)), None)
            orig_str = f" orig={original:.4f}" if original else ""
            drop = original - mean_auc if original else 0

            flag = ""
            if drop > 0.05:
                flag = " ⚠️ DROP >0.05 (lucky split!)"
                flagged.append((model, pool, n, C, drop))

            print(f"  {model:20s} {pool:15s} n={n:>3s} C={C:>5s}  "
                  f"mean={mean_auc:.4f} ± {std_auc:.4f}  "
                  f"range=[{min(aucs):.4f}, {max(aucs):.4f}]{orig_str}{flag}")

        if flagged:
            print(f"\n⚠️  FLAGGED ({len(flagged)} configs with AUC drop > 0.05 across seeds):")
            for model, pool, n, C, drop in flagged:
                key = (model, pool, int(n), float(C))
                orig_val = TOP_10_CONFIGS_LOOKUP.get(key)
                print(f"  {model}/{pool} n={n} C={C}: drop={drop:.4f} "
                      f"(orig={orig_val:.4f})" if orig_val else
                      f"  {model}/{pool} n={n} C={C}: drop={drop:.4f}")


# Lookup for original AUC values
TOP_10_CONFIGS_LOOKUP = {
    (m, p, int(n), float(C)): orig
    for m, p, n, C, orig in TOP_10_CONFIGS
}

if __name__ == "__main__":
    main()
