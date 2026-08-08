#!/usr/bin/env python3
"""
Content-Matching Experiment — Issue #11
========================================
Match Leica↔non-Leica images by CLIP content similarity and aspect ratio,
then re-run LR probes on matched pairs to determine whether the Phase 1
signal is LENS RENDERING or CONTENT.

Matching strategies:
1. CLIP-matched: for each Leica image, pick the non-Leica image with
   highest cosine similarity in CLIP CLS embedding space.
2. Aspect-ratio-matched: for each Leica image, pick the non-Leica image
   with closest aspect ratio (width/height).
3. Random pairs (baseline): shuffle neg indices.

Output: experiments/discriminator-content-matched/results.csv
"""

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_logistic_probes import (
    load_metadata,
    stratify_groups,
    run_split,
    EMBEDDINGS_DIR,
    C_VALUES,
)

OUTPUT_DIR = REPO_ROOT / "experiments/discriminator-content-matched"
OUTPUT_CSV = OUTPUT_DIR / "results.csv"

MODELS = [
    "dinov2-small",
    "dinov2-base",
    "dinov2-large",
    "dinov2-giant",
    "dinov3-vitl16",
    "siglip-so400m",
    "clip-vitl14",
]

POOLING = ["cls", "patch_mean", "patch_gem", "cls_patch"]
MATCH_STRATEGIES = ["clip_matched", "aspect_matched", "random"]
DATASET_SIZE = 250  # Most reliable per seed sweep
SEEDS = [42, 7, 17, 23, 99, 1234]  # Include seed=42 + 5 others


def load_aspect_ratios():
    """Return aspect ratios (w/h) for all images in verified.csv order."""
    ratios = []
    csv_path = REPO_ROOT / "data/registry/verified.csv"
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            w = float(row["width"])
            h = float(row["height"])
            ratios.append(w / h if h > 0 else 1.0)
    return np.array(ratios)


def match_pairs(
    strategy: str,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    clip_embeddings: np.ndarray,
    aspect_ratios: np.ndarray,
) -> np.ndarray:
    """
    For each positive image, select ONE negative image as its match.
    Returns array of (pos_idx, neg_idx) pairs of shape (n_pairs, 2).
    """
    if strategy == "random":
        rng = np.random.RandomState(42)
        neg_shuffled = neg_idx.copy()
        rng.shuffle(neg_shuffled)
        # If more neg than pos, take only as many as needed
        n_pairs = min(len(pos_idx), len(neg_shuffled))
        pairs = np.column_stack([pos_idx[:n_pairs], neg_shuffled[:n_pairs]])
        return pairs

    pairs = []
    used_neg = set()

    for pi in pos_idx:
        if strategy == "clip_matched":
            # Compute cosine similarity between this pos image and all neg images
            pos_vec = clip_embeddings[pi:pi+1]  # shape (1, D)
            neg_vecs = clip_embeddings[neg_idx]  # shape (N_neg, D)
            sims = cosine_similarity(pos_vec, neg_vecs)[0]

            # Pick the best UNUSED neg
            sorted_indices = np.argsort(-sims)  # descending similarity
            best_ni = None
            for ni in sorted_indices:
                candidate = neg_idx[ni]
                if candidate not in used_neg:
                    best_ni = candidate
                    break

            if best_ni is not None:
                pairs.append([pi, best_ni])
                used_neg.add(best_ni)

        elif strategy == "aspect_matched":
            # Find neg with closest aspect ratio
            pos_ar = aspect_ratios[pi]
            neg_ars = aspect_ratios[neg_idx]
            diffs = np.abs(neg_ars - pos_ar)

            sorted_indices = np.argsort(diffs)
            best_ni = None
            for ni in sorted_indices:
                candidate = neg_idx[ni]
                if candidate not in used_neg:
                    best_ni = candidate
                    break

            if best_ni is not None:
                pairs.append([pi, best_ni])
                used_neg.add(best_ni)

    return np.array(pairs)


def evaluate_matched(
    pairs: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_per_class: int,
    C_values: list[float],
    seeds: list[int],
) -> list[dict]:
    """
    For each seed, create a matched subset from the pairs, then run LR.
    The matched subset uses exactly the paired images.

    Since pairs are already 50/50 (one pos, one neg per pair), we sample
    n_per_class//2 pairs (giving n_per_class images from each class).
    Returns list of result dicts.
    """
    n_pairs = len(pairs)
    results = []

    for seed in seeds:
        rng = np.random.RandomState(seed)

        # Sample pairs
        n_sample_pairs = min(n_per_class, n_pairs)
        sampled_pair_idx = rng.choice(n_pairs, size=n_sample_pairs, replace=False)

        # Build X_sub, y_sub from sampled pairs
        sampled_indices = []
        for pi in sampled_pair_idx:
            pos_i, neg_i = pairs[pi]
            sampled_indices.extend([pos_i, neg_i])

        sampled_indices = np.array(sampled_indices)
        X_sub = X[sampled_indices]
        y_sub = labels[sampled_indices]
        g_sub = groups[sampled_indices]

        if len(y_sub) < 4:
            continue

        # Use the same split logic as run_split but adapted
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        unique_in_sample = np.unique(g_sub)
        min_group_count = min(np.sum(g_sub == ug) for ug in unique_in_sample) if len(unique_in_sample) > 0 else 0

        test_size = 0.30
        if len(unique_in_sample) >= 2 and min_group_count >= 2:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sub, y_sub, test_size=test_size,
                    stratify=g_sub, random_state=seed)
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sub, y_sub, test_size=test_size,
                    stratify=y_sub, random_state=seed)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_sub, y_sub, test_size=test_size,
                stratify=y_sub, random_state=seed)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        for C in C_values:
            try:
                clf = LogisticRegression(
                    C=C, penalty="l2", solver="lbfgs",
                    max_iter=5000, random_state=seed,
                    class_weight="balanced",
                )
                clf.fit(X_train_s, y_train)
                y_prob = clf.predict_proba(X_test_s)[:, 1]
                auc = float(roc_auc_score(y_test, y_prob))

                results.append({
                    "seed": seed,
                    "C": C,
                    "auc": round(auc, 6),
                    "test_pos": int(np.sum(y_test == 1)),
                    "test_neg": int(np.sum(y_test == 0)),
                    "n_pairs_sampled": n_sample_pairs,
                })
            except Exception as e:
                results.append({
                    "seed": seed,
                    "C": C,
                    "auc": f"ERROR:{e}",
                    "test_pos": int(np.sum(y_test == 1)),
                    "test_neg": int(np.sum(y_test == 0)),
                    "n_pairs_sampled": n_sample_pairs,
                })

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_CSV}")

    # Load metadata
    flickr_ids, scene_types, bodies, labels = load_metadata()
    groups = stratify_groups(scene_types, bodies)
    print(f"Loaded {len(labels)} images ({labels.sum()} pos, {(labels==0).sum()} neg)")

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    print(f"Positive: {len(pos_idx)}, Negative: {len(neg_idx)}")

    # Load CLIP CLS embeddings for matching
    clip_path = EMBEDDINGS_DIR / "clip-vitl14" / "cls.npy"
    if not clip_path.exists():
        print(f"ERROR: CLIP embeddings not found at {clip_path}")
        return
    clip_embeddings = np.load(clip_path).astype(np.float32)
    print(f"CLIP embeddings: {clip_embeddings.shape}")

    # Load aspect ratios
    aspect_ratios = load_aspect_ratios()

    # ── Build pairs for each strategy ──
    all_pairs = {}
    for strategy in MATCH_STRATEGIES:
        pairs = match_pairs(strategy, pos_idx, neg_idx,
                           clip_embeddings, aspect_ratios)
        all_pairs[strategy] = pairs
        print(f"\n{strategy}: {len(pairs)} matched pairs")

    # ── Run evaluations ──
    fieldnames = [
        "model", "pooling", "match_strategy", "C", "seed",
        "auc", "test_pos", "test_neg", "embedding_dim",
        "n_pairs_sampled", "n_pairs_total", "runtime_sec",
    ]

    existing = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                existing.add((
                    row["model"], row["pooling"], row["match_strategy"],
                    row["C"], row["seed"],
                ))

    total_configs = (len(MODELS) * len(POOLING) * len(MATCH_STRATEGIES)
                     * len(C_VALUES) * len(SEEDS))
    completed = len(existing)
    print(f"\nTotal configs: {total_configs}, already done: {completed}")

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()

        for model_name in MODELS:
            for pool_name in POOLING:
                emb_path = EMBEDDINGS_DIR / model_name / f"{pool_name}.npy"
                if not emb_path.exists():
                    print(f"  ⚠️  Missing: {emb_path}")
                    continue

                t0 = time.time()
                X = np.load(emb_path).astype(np.float32)
                embedding_dim = X.shape[1]

                for strategy in MATCH_STRATEGIES:
                    pairs = all_pairs[strategy]
                    n_pairs_total = len(pairs)
                    if n_pairs_total < 10:
                        print(f"  ⚠️  {model_name}/{pool_name}/{strategy}: "
                              f"only {n_pairs_total} pairs — skipping")
                        continue

                    # Skip already-completed configs
                    skip_count = 0
                    for C in C_VALUES:
                        for seed in SEEDS:
                            key = (
                                model_name, pool_name, strategy,
                                str(C), str(seed),
                            )
                            if key in existing:
                                skip_count += 1

                    if skip_count == len(C_VALUES) * len(SEEDS):
                        continue

                    # Run evaluation
                    eval_results = evaluate_matched(
                        pairs, X, labels, groups,
                        n_per_class=DATASET_SIZE,
                        C_values=C_VALUES,
                        seeds=SEEDS,
                    )

                    for er in eval_results:
                        key = (
                            model_name, pool_name, strategy,
                            str(er["C"]), str(er["seed"]),
                        )
                        if key in existing:
                            continue

                        row = {
                            "model": model_name,
                            "pooling": pool_name,
                            "match_strategy": strategy,
                            "C": str(er["C"]),
                            "seed": str(er["seed"]),
                            "auc": er["auc"] if isinstance(er["auc"], float) else "",
                            "test_pos": er["test_pos"],
                            "test_neg": er["test_neg"],
                            "embedding_dim": embedding_dim,
                            "n_pairs_sampled": er["n_pairs_sampled"],
                            "n_pairs_total": n_pairs_total,
                            "runtime_sec": round(time.time() - t0, 2),
                        }
                        writer.writerow(row)
                        completed += 1

                elapsed = time.time() - t0
                print(f"  {model_name}/{pool_name}: {len(MATCH_STRATEGIES)}×"
                      f"{len(C_VALUES)}×{len(SEEDS)} configs in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"Done. {completed}/{total_configs} evaluations.")
    print(f"Results: {OUTPUT_CSV}")

    # ── Summary ──
    if OUTPUT_CSV.exists():
        print(f"\n{'='*60}")
        print("SUMMARY: Mean AUC by Model × Strategy (best C, average across seeds)")
        print(f"{'='*60}")

        from collections import defaultdict
        strat_model_pool_data = defaultdict(lambda: defaultdict(list))

        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                if row["auc"] and row["auc"] != "" and "ERROR" not in str(row["auc"]):
                    key = (row["match_strategy"], row["model"], row["pooling"])
                    strat_model_pool_data[key].append(float(row["auc"]))

        for strategy in MATCH_STRATEGIES:
            print(f"\n--- {strategy} ---")
            print(f"{'Model':<25s} {'Best Pool':<15s} {'Mean AUC':>10s} {'σ':>8s} {'Best':>10s} {'Worst':>10s}")
            print("-" * 80)

            model_best = {}
            for (s, model, pool), aucs in strat_model_pool_data.items():
                if s != strategy:
                    continue
                mean_auc = np.mean(aucs)
                key = model
                if key not in model_best or mean_auc > model_best[key][1]:
                    model_best[key] = (pool, mean_auc, np.std(aucs),
                                       max(aucs), min(aucs))

            model_order = MODELS
            for model in model_order:
                if model in model_best:
                    pool, mean_auc, std_auc, best, worst = model_best[model]
                    print(f"  {model:<23s} {pool:<15s} {mean_auc:10.4f} {std_auc:8.4f} {best:10.4f} {worst:10.4f}")

        # CLIP vs DINOv2 gap
        print(f"\n--- CLIP vs DINOv2 gap per strategy ---")
        for strategy in MATCH_STRATEGIES:
            clip_aucs = []
            dino_aucs = []
            for (s, model, pool), aucs in strat_model_pool_data.items():
                if s != strategy:
                    continue
                if model == "clip-vitl14":
                    clip_aucs.extend(aucs)
                elif model.startswith("dinov"):
                    dino_aucs.extend(aucs)

            if clip_aucs and dino_aucs:
                clip_mean = np.mean(clip_aucs)
                dino_mean = np.mean(dino_aucs)
                gap = clip_mean - dino_mean
                print(f"  {strategy:<20s} CLIP={clip_mean:.4f}  DINOv2={dino_mean:.4f}  gap={gap:+.4f}")


if __name__ == "__main__":
    main()
