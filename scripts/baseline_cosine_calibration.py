#!/usr/bin/env python3
"""
Baseline cosine-distance calibration for Phase 2a gate threshold.

Computes DINOv2-g embedding cosine distances to establish:
  - Same-class noise floor (Leica↔Leica, NonLeica↔NonLeica)
  - Cross-class signal range (Leica↔NonLeica)
  - Matched-pair distances (uses content-matched pairs from #11)

These baselines inform whether the gate threshold of Δ ≥ 0.02
is reasonable — it must sit above the noise floor and within
reach of the cross-class signal.

CPU-only. Runs in ~30 seconds on 592 embeddings.
"""

import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict

EMBEDDINGS_DIR = Path("/mnt/nas-ai-models/training-data/leica-look/embeddings")
DATA_DIR = Path("/home/tim/source/activity/leica-look/data")
MATCHED_DIR = Path("/home/tim/source/activity/leica-look/experiments/discriminator-content-matched")

MODEL = "dinov2-giant"
POOLING = "cls"  # CLS token (standard for DINOv2)


def cosine_distance(a, b):
    """Cosine distance = 1 - cosine_similarity. Range [0, 2] for non-negative features."""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 1.0 - np.dot(a_norm, b_norm.T)


def cosine_similarity(a, b):
    """Pairwise cosine similarity."""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b_norm).sum(axis=-1)


def compute_distributions(leica_embs, nonleica_embs, n_sample=5000):
    """Compute distance distributions with sampling to avoid O(N²) memory."""
    rng = np.random.RandomState(42)

    # Leica↔Leica (same-class noise floor) — sample pairs
    n_leica = len(leica_embs)
    indices = np.triu_indices(n_leica, k=1)
    n_pairs = len(indices[0])
    if n_pairs > n_sample:
        idx = rng.choice(n_pairs, n_sample, replace=False)
        pairs_i, pairs_j = indices[0][idx], indices[1][idx]
    else:
        pairs_i, pairs_j = indices

    leica_leica_dists = cosine_similarity(leica_embs[pairs_i], leica_embs[pairs_j])
    leica_leica_dists = 1.0 - leica_leica_dists  # convert to distance

    # NonLeica↔NonLeica (other same-class noise floor)
    n_nl = len(nonleica_embs)
    indices_nl = np.triu_indices(n_nl, k=1)
    n_pairs_nl = len(indices_nl[0])
    if n_pairs_nl > n_sample:
        idx = rng.choice(n_pairs_nl, n_sample, replace=False)
        pairs_i, pairs_j = indices_nl[0][idx], indices_nl[1][idx]
    else:
        pairs_i, pairs_j = indices_nl

    nl_nl_dists = cosine_similarity(nonleica_embs[pairs_i], nonleica_embs[pairs_j])
    nl_nl_dists = 1.0 - nl_nl_dists

    # Leica↔NonLeica (cross-class signal) — sample pairs
    n_cross = n_leica * n_nl
    if n_cross > n_sample:
        idx_i = rng.choice(n_leica, n_sample, replace=True)
        idx_j = rng.choice(n_nl, n_sample, replace=True)
    else:
        idx_i, idx_j = np.meshgrid(np.arange(n_leica), np.arange(n_nl), indexing='ij')
        idx_i = idx_i.flatten()
        idx_j = idx_j.flatten()

    cross_dists = cosine_similarity(leica_embs[idx_i], nonleica_embs[idx_j])
    cross_dists = 1.0 - cross_dists

    return {
        "leica_leica": leica_leica_dists,
        "nonleica_nonleica": nl_nl_dists,
        "leica_nonleica": cross_dists,
    }


def compute_matched_distances(leica_embs, nonleica_embs, leica_ids, nonleica_ids, matched_pairs):
    """Compute distances for content-matched pairs specifically."""
    leica_id_to_idx = {iid: idx for idx, iid in enumerate(leica_ids)}
    nonleica_id_to_idx = {iid: idx for idx, iid in enumerate(nonleica_ids)}

    dists = []
    unmatched = 0
    for pair in matched_pairs:
        leica_idx = leica_id_to_idx.get(pair["leica_id"])
        nl_idx = nonleica_id_to_idx.get(pair["nonleica_id"])
        if leica_idx is not None and nl_idx is not None:
            d = 1.0 - cosine_similarity(leica_embs[leica_idx], nonleica_embs[nl_idx])
            dists.append(float(d))
        else:
            unmatched += 1

    return np.array(dists), unmatched


def print_stats(name, arr):
    if len(arr) == 0:
        print(f"  {name}: NO DATA")
        return
    print(f"  {name}:")
    print(f"    n={len(arr)}, mean={np.mean(arr):.6f}, std={np.std(arr):.6f}")
    print(f"    median={np.median(arr):.6f}")
    print(f"    min={np.min(arr):.6f}, max={np.max(arr):.6f}")
    pcts = [5, 10, 25, 50, 75, 90, 95]
    vals = np.percentile(arr, pcts)
    print(f"    percentiles: " + ", ".join(f"P{p}={v:.6f}" for p, v in zip(pcts, vals)))


def main():
    print(f"=== Baseline Cosine-Distance Calibration ===")
    print(f"Model: {MODEL}, Pooling: {POOLING}")
    print(f"Data: {EMBEDDINGS_DIR}\n")

    # Load embeddings
    emb_path = EMBEDDINGS_DIR / MODEL / f"{POOLING}.npy"
    ids_path = EMBEDDINGS_DIR / "image_ids.txt"
    labels_path = EMBEDDINGS_DIR / "labels.npy"

    embs = np.load(emb_path).astype(np.float32)
    labels = np.load(labels_path).astype(int)
    with open(ids_path) as f:
        ids = [line.strip() for line in f]

    # Normalize for cosine computation
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)

    leica_mask = labels == 1
    nonleica_mask = labels == 0

    leica_embs = embs[leica_mask]
    nonleica_embs = embs[nonleica_mask]
    leica_ids = [ids[i] for i in range(len(ids)) if leica_mask[i]]
    nonleica_ids = [ids[i] for i in range(len(ids)) if nonleica_mask[i]]

    print(f"Total images: {len(embs)}")
    print(f"  Leica: {len(leica_embs)}")
    print(f"  Non-Leica: {len(nonleica_embs)}")
    print()

    # Compute distributions
    print("Computing distance distributions (sampled to 5000 pairs each)...")
    dists = compute_distributions(leica_embs, nonleica_embs)

    print("\n--- Same-Class Noise Floors ---")
    print_stats("Leica ↔ Leica", dists["leica_leica"])
    print()
    print_stats("NonLeica ↔ NonLeica", dists["nonleica_nonleica"])

    print("\n--- Cross-Class Signal ---")
    print_stats("Leica ↔ NonLeica (random)", dists["leica_nonleica"])

    # Matched pairs
    print("\n--- Content-Matched Pairs (#11) ---")
    for matched_file in ["pairs_train.json", "pairs_val.json", "matched_pairs.json", "pairs.json"]:
        mp = MATCHED_DIR / matched_file
        if mp.exists():
            with open(mp) as f:
                pairs = json.load(f)
            if isinstance(pairs, dict):
                pairs = list(pairs.values()) if "pairs" in pairs else pairs.get("pairs", [])
            matched_dists, unmatched = compute_matched_distances(
                leica_embs, nonleica_embs, leica_ids, nonleica_ids, pairs
            )
            print(f"  Source: {matched_file}")
            print(f"  Total pairs: {len(pairs)}, unmatched: {unmatched}")
            print_stats(f"  Matched Leica↔NonLeica", matched_dists)
            break
    else:
        print("  No matched-pair file found (check experiments/discriminator-content-matched/)")

    # Key analysis for gate threshold
    print("\n=== Threshold Analysis ===")
    
    # --- Centroid-based distances (the actual gate metric) ---
    leica_centroid = leica_embs.mean(axis=0)
    leica_centroid = leica_centroid / np.linalg.norm(leica_centroid)
    leica_to_centroid = 1.0 - (leica_embs * leica_centroid).sum(axis=-1)
    nonleica_to_centroid = 1.0 - (nonleica_embs * leica_centroid).sum(axis=-1)
    
    leica_centroid_mean = np.mean(leica_to_centroid)
    leica_centroid_std = np.std(leica_to_centroid)
    nonleica_centroid_mean = np.mean(nonleica_to_centroid)
    nonleica_centroid_std = np.std(nonleica_to_centroid)
    centroid_gap = nonleica_centroid_mean - leica_centroid_mean
    
    print(f"\n--- Centroid-Based Distances (Gate Metric) ---")
    print(f"Leica → Leica centroid:          mean={leica_centroid_mean:.6f}, std={leica_centroid_std:.6f}")
    print(f"NonLeica → Leica centroid:       mean={nonleica_centroid_mean:.6f}, std={nonleica_centroid_std:.6f}")
    print(f"Centroid gap (cross - same):     {centroid_gap:.6f}")
    print(f"Effect size (gap / Leica σ):     {centroid_gap / leica_centroid_std:.2f}")
    
    # --- Proposed threshold Δ ≥ 0.02 ---
    proposed_threshold = 0.02
    pct_of_centroid_gap = proposed_threshold / centroid_gap * 100
    pct_nl_near_centroid = np.mean(nonleica_to_centroid <= leica_centroid_mean + proposed_threshold) * 100
    
    print(f"\n--- Gate Threshold Δ ≥ {proposed_threshold} ---")
    print(f"Δ as % of centroid gap:           {pct_of_centroid_gap:.1f}%")
    print(f"NonLeica within Δ of Leica mean:  {pct_nl_near_centroid:.1f}%")
    
    if pct_of_centroid_gap < 25:
        print(f"✓ Δ requires closing {pct_of_centroid_gap:.0f}% of gap — reasonable for 1500-step feasibility")
    elif pct_of_centroid_gap < 50:
        print(f"⚠ Δ requires closing {pct_of_centroid_gap:.0f}% of gap — ambitious for feasibility")
    else:
        print(f"✗ Δ requires closing {pct_of_centroid_gap:.0f}% of gap — too strict for feasibility")
    
    # --- Pairwise-based analysis (for context) ---
    leica_leica_mean = np.mean(dists["leica_leica"])
    leica_leica_std = np.std(dists["leica_leica"])
    cross_mean = np.mean(dists["leica_nonleica"])
    cross_std = np.std(dists["leica_nonleica"])
    raw_gap = cross_mean - leica_leica_mean
    noise_to_signal_ratio = raw_gap / leica_leica_std
    
    print(f"\n--- Pairwise Distances (Context Only) ---")
    print(f"Same-class (Leica↔Leica) mean:    {leica_leica_mean:.6f} ± {leica_leica_std:.6f}")
    print(f"Cross-class (Leica↔NonLeica):     {cross_mean:.6f} ± {cross_std:.6f}")
    print(f"Raw gap:                           {raw_gap:.6f}")
    print(f"Cohen's d:                         {raw_gap / ((leica_leica_std + cross_std)/2):.3f}")
    print(f"NOTE: Pairwise gap ({raw_gap:.3f}) << centroid gap ({centroid_gap:.3f})")
    print(f"      because Leica images are diverse — the centroid removes scene variance.")
    print(f"      Use CENTROID gap for gate calibration, not pairwise.")

    # Output JSON for machine consumption
    output = {
        "model": MODEL,
        "pooling": POOLING,
        "n_total": int(len(embs)),
        "n_leica": int(len(leica_embs)),
        "n_nonleica": int(len(nonleica_embs)),
        "same_class_leica": {
            "mean": float(leica_leica_mean),
            "std": float(leica_leica_std),
            "median": float(np.median(dists["leica_leica"])),
            "p5": float(np.percentile(dists["leica_leica"], 5)),
            "p95": float(np.percentile(dists["leica_leica"], 95)),
        },
        "same_class_nonleica": {
            "mean": float(np.mean(dists["nonleica_nonleica"])),
            "std": float(np.std(dists["nonleica_nonleica"])),
            "median": float(np.median(dists["nonleica_nonleica"])),
        },
        "cross_class": {
            "mean": float(cross_mean),
            "std": float(cross_std),
            "median": float(np.median(dists["leica_nonleica"])),
            "p5": float(np.percentile(dists["leica_nonleica"], 5)),
            "p95": float(np.percentile(dists["leica_nonleica"], 95)),
        },
        "raw_gap": float(raw_gap),
        "noise_to_signal_ratio": float(noise_to_signal_ratio),
        "centroid_analysis": {
            "leica_to_centroid_mean": float(leica_centroid_mean),
            "leica_to_centroid_std": float(leica_centroid_std),
            "nonleica_to_centroid_mean": float(nonleica_centroid_mean),
            "nonleica_to_centroid_std": float(nonleica_centroid_std),
            "centroid_gap": float(centroid_gap),
            "effect_size": float(centroid_gap / leica_centroid_std),
            "proposed_delta_0_02_pct_of_gap": float(pct_of_centroid_gap),
            "pct_nl_within_delta_of_leica_mean": float(pct_nl_near_centroid),
        },
    }

    out_path = Path("experiments/lora-transfer/baseline_cosine_distances.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
