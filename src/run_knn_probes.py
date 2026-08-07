#!/usr/bin/env python3
"""
k-NN Probes — Issue #7
=======================
Run k-NN classifiers (non-parametric baseline) across all 7 models × 6 pooling × 3 dataset sizes.
Grid: 7 × 6 × 3 × 3 k values = 378 evaluations.

k values: {1, 5, 11}
Uses cosine distance (better for normalized/angular embeddings).

Output: experiments/discriminator-knn/results.csv
"""

import csv
import os
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFIED_CSV = REPO_ROOT / "data/registry/verified.csv"
EMBEDDINGS_DIR = Path("/mnt/nas-ai-models/training-data/leica-look/embeddings")
OUTPUT_DIR = REPO_ROOT / "experiments/discriminator-knn"
OUTPUT_CSV = OUTPUT_DIR / "results.csv"

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
MODELS = [
    "dinov2-small",
    "dinov2-base",
    "dinov2-large",
    "dinov2-giant",
    "dinov3-vitl16",
    "siglip-so400m",
    "clip-vitl14",
]

POOLING = [
    "cls",
    "patch_mean",
    "patch_max",
    "patch_gem",
    "cls_patch",
    "multicrop",
]

DATASET_SIZES = [50, 100, 250]
K_VALUES = [1, 5, 11]
RANDOM_SEED_BASE = 42
TEST_SIZE = 0.30


# ---------------------------------------------------------------------------
# Data loading (same as LR/MLP)
# ---------------------------------------------------------------------------
def load_metadata() -> tuple[list[str], list[str], list[str], np.ndarray]:
    flickr_ids, scene_types, bodies, labels = [], [], [], []
    with open(VERIFIED_CSV, "r") as f:
        for row in csv.DictReader(f):
            flickr_ids.append(row["flickr_id"])
            scene_types.append(row["scene_type"] or "other")
            bodies.append(row["body"] or "unknown")
            labels.append(1 if row["class"] == "positive" else 0)
    return flickr_ids, scene_types, bodies, np.array(labels, dtype=np.int64)


def stratify_groups(scene_types: list[str], bodies: list[str]) -> np.ndarray:
    major_scenes = {"portrait", "landscape", "street", "night", "architecture", "macro"}
    collapsed_scenes = [s if s in major_scenes else "other" for s in scene_types]
    leica_bodies = {
        "LEICA M10", "LEICA M10-R", "LEICA M10-P", "LEICA M11",
        "LEICA SL (Typ 601)", "LEICA Q2", "LEICA Q3", "LEICA SL2", "LEICA SL3",
    }
    collapsed_bodies = []
    for b in bodies:
        if b in leica_bodies:
            collapsed_bodies.append("leica")
        elif b.startswith("Canon"):
            collapsed_bodies.append("canon")
        elif b.startswith("ILCE-"):
            collapsed_bodies.append("sony")
        elif b.startswith("NIKON"):
            collapsed_bodies.append("nikon")
        else:
            collapsed_bodies.append("other")

    groups = [f"{s}|{b}" for s, b in zip(collapsed_scenes, collapsed_bodies)]
    group_counts = Counter(groups)
    rare_groups = {g for g, c in group_counts.items() if c < 3}
    group_ids = []
    mapping = {}
    next_id = 0
    for g in groups:
        if g in rare_groups:
            g_key = "rare"
        else:
            g_key = g
        if g_key not in mapping:
            mapping[g_key] = next_id
            next_id += 1
        group_ids.append(mapping[g_key])
    return np.array(group_ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              n_per_class: int, seed: int) -> dict:
    """Sample, split, evaluate k-NN for all k values."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)

    unique_groups = np.unique(groups)
    pos_sampled, neg_sampled = [], []
    for g in unique_groups:
        pos_in_g = pos_idx[groups[pos_idx] == g]
        neg_in_g = neg_idx[groups[neg_idx] == g]
        prop_pos = len(pos_in_g) / len(pos_idx) if len(pos_idx) > 0 else 0
        prop_neg = len(neg_in_g) / len(neg_idx) if len(neg_idx) > 0 else 0
        n_pos_sample = max(0, min(len(pos_in_g), int(round(n_per_class * prop_pos))))
        n_neg_sample = max(0, min(len(neg_in_g), int(round(n_per_class * prop_neg))))
        if n_pos_sample > 0:
            pos_sampled.extend(rng.choice(pos_in_g, size=n_pos_sample, replace=False))
        if n_neg_sample > 0:
            neg_sampled.extend(rng.choice(neg_in_g, size=n_neg_sample, replace=False))

    if len(pos_sampled) < n_per_class:
        remaining = list(set(pos_idx) - set(pos_sampled))
        if remaining:
            extra = rng.choice(remaining, size=min(n_per_class - len(pos_sampled), len(remaining)), replace=False)
            pos_sampled.extend(extra)
    pos_sampled = pos_sampled[:n_per_class]

    if len(neg_sampled) < n_per_class:
        remaining = list(set(neg_idx) - set(neg_sampled))
        if remaining:
            extra = rng.choice(remaining, size=min(n_per_class - len(neg_sampled), len(remaining)), replace=False)
            neg_sampled.extend(extra)
    neg_sampled = neg_sampled[:n_per_class]

    sampled_idx = np.concatenate([pos_sampled, neg_sampled])
    X_sub = X[sampled_idx]
    y_sub = y[sampled_idx]
    g_sub = groups[sampled_idx]

    unique_in_sample = np.unique(g_sub)
    min_group_count = min(np.sum(g_sub == ug) for ug in unique_in_sample) if len(unique_in_sample) > 0 else 0
    if len(unique_in_sample) >= 2 and min_group_count >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_sub, y_sub, test_size=TEST_SIZE, stratify=g_sub, random_state=seed)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_sub, y_sub, test_size=TEST_SIZE, stratify=y_sub, random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=TEST_SIZE, stratify=y_sub, random_state=seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}
    for k in K_VALUES:
        knn = KNeighborsClassifier(
            n_neighbors=k, metric="cosine", weights="distance",
            algorithm="brute", n_jobs=-1)
        knn.fit(X_train, y_train)
        y_prob = knn.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
        results[k] = round(auc, 6)

    return {
        "test_pos": int(np.sum(y_test == 1)),
        "test_neg": int(np.sum(y_test == 0)),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_CSV}")

    flickr_ids, scene_types, bodies, labels = load_metadata()
    groups = stratify_groups(scene_types, bodies)
    print(f"Loaded {len(labels)} images ({labels.sum()} pos, {(labels==0).sum()} neg)")

    fieldnames = [
        "model", "pooling", "dataset_size", "k", "auc",
        "test_pos", "test_neg", "embedding_dim", "runtime_sec"
    ]

    existing = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "r") as f:
            for row in csv.DictReader(f):
                existing.add((row["model"], row["pooling"], row["dataset_size"], row["k"]))
        print(f"Found {len(existing)} existing results — will skip")

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()

        total_configs = len(MODELS) * len(POOLING) * len(DATASET_SIZES) * len(K_VALUES)
        completed = len(existing)
        print(f"\nStarting: {total_configs} total, {completed} done")

        for model_name in MODELS:
            model_dir = EMBEDDINGS_DIR / model_name

            for pool_name in POOLING:
                emb_path = model_dir / f"{pool_name}.npy"
                if not emb_path.exists():
                    print(f"  ⚠️  Missing: {emb_path}")
                    continue

                t0 = time.time()
                X = np.load(emb_path).astype(np.float32)
                embedding_dim = X.shape[1]

                for n_per_class in DATASET_SIZES:
                    n_pos = int(labels.sum())
                    n_neg = int((labels == 0).sum())
                    if n_pos < n_per_class or n_neg < n_per_class:
                        continue

                    seed_offset = DATASET_SIZES.index(n_per_class) * 1000
                    seed = RANDOM_SEED_BASE + seed_offset

                    split_info = run_split(X, labels, groups, n_per_class, seed)
                    test_pos = split_info["test_pos"]
                    test_neg = split_info["test_neg"]

                    for k, auc_val in split_info["results"].items():
                        key = (model_name, pool_name, str(n_per_class), str(k))
                        if key in existing:
                            continue

                        row = {
                            "model": model_name,
                            "pooling": pool_name,
                            "dataset_size": str(n_per_class),
                            "k": str(k),
                            "auc": auc_val,
                            "test_pos": test_pos,
                            "test_neg": test_neg,
                            "embedding_dim": embedding_dim,
                            "runtime_sec": round(time.time() - t0, 2),
                        }
                        writer.writerow(row)
                        completed += 1

                elapsed = time.time() - t0
                print(f"  {model_name}/{pool_name}: dim={embedding_dim}, "
                      f"{len(DATASET_SIZES)}×{len(K_VALUES)} configs in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"Done. {completed}/{total_configs} evaluations.")
    print(f"Results: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
