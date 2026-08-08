#!/usr/bin/env python3
"""
Color-transfer baseline control arm for Phase 2a.

Purpose: a pure color/LUT shift would pass the DINOv2 embedding-shift gate
trivially WITHOUT transferring any lens rendering. This script computes that
color-only baseline so the gate can report LoRA vs color-baseline side-by-side.

If LoRA Δ ≈ color-baseline Δ → the 'Leica look' is color science, not optics.

Method: Reinhard et al. 2001 color transfer in LAB color space.
  - Compute per-channel mean/std of Leica reference images
  - For each non-Leica image: shift mean → Leica mean, scale std → Leica std
  - Extract DINOv2 embeddings from color-transferred images
  - Compute centroid shift metric (identical to gate #1)

Output:
  runs/color_baseline/{timestamp}/
    color_transfer_results.json     — metrics for the color baseline
    samples/                         — side-by-side before/after images

USAGE:
  # CPU-only (uses pre-extracted embeddings, but generates images)
  python3 color_transfer_baseline.py --output-dir runs/color_baseline/2026-08-08_0000

  # Or compute and save images only (no embedding extraction)
  python3 color_transfer_baseline.py --images-only
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add experiment src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation import (
    cosine_distance,
    embedding_shift,
    load_embeddings,
    load_leica_centroid,
)

PROJECT_ROOT = Path(os.environ.get("LEICA_LOOK_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent)))
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "lora-transfer"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "verified.csv"
RAW_DIR = Path("/mnt/nas-ai-models/training-data/leica-look/raw")


def load_image_paths(csv_path: Path, class_filter: Optional[str] = None) -> list:
    """Load image file paths from the verified registry."""
    paths = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if class_filter is not None and row["class"] != class_filter:
                continue
            file_path = row.get("file_path", "")
            if file_path and Path(file_path).exists():
                paths.append(file_path)
    return paths


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB (H, W, 3) uint8 to LAB float32.
    Uses skimage-style conversion for reproducibility.
    """
    from skimage.color import rgb2lab
    return rgb2lab(rgb)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB float32 to RGB uint8."""
    from skimage.color import lab2rgb
    rgb = lab2rgb(lab)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def compute_lab_stats(image_paths: list, max_images: int = 100) -> dict:
    """
    Compute per-channel LAB mean and std across Leica reference images.

    Uses a subsample for efficiency (100 images is plenty for stable stats).
    """
    import random
    random.seed(42)

    paths = random.sample(image_paths, min(max_images, len(image_paths)))

    l_vals, a_vals, b_vals = [], [], []
    for p in tqdm(paths, desc="Computing Leica LAB stats"):
        img = Image.open(p).convert("RGB")
        arr = np.array(img)
        lab = rgb_to_lab(arr)
        l_vals.append(lab[..., 0].mean())
        a_vals.append(lab[..., 1].mean())
        b_vals.append(lab[..., 2].mean())

    stats = {
        "n_images": len(paths),
        "L_mean": float(np.mean(l_vals)),
        "L_std": float(np.std(l_vals)),
        "A_mean": float(np.mean(a_vals)),
        "A_std": float(np.std(a_vals)),
        "B_mean": float(np.mean(b_vals)),
        "B_std": float(np.std(b_vals)),
    }
    return stats


def apply_color_transfer(
    source_path: str,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
) -> np.ndarray:
    """
    Apply Reinhard color transfer: match source mean/std to reference.

    Args:
        source_path: Path to source image
        ref_mean: (3,) LAB mean of reference
        ref_std: (3,) LAB std of reference

    Returns:
        (H, W, 3) uint8 RGB image with transferred color
    """
    src_img = Image.open(source_path).convert("RGB")
    src_arr = np.array(src_img)
    src_lab = rgb_to_lab(src_arr)

    # Per-channel stats
    src_mean = np.array([src_lab[..., c].mean() for c in range(3)])
    src_std = np.array([src_lab[..., c].std() for c in range(3)])

    # Transfer: (src - src_mean) * (ref_std / src_std) + ref_mean
    # Handle zero std gracefully
    epsilon = 1e-8
    scale = ref_std / (src_std + epsilon)

    result_lab = src_lab.copy().astype(np.float64)
    for c in range(3):
        result_lab[..., c] = (src_lab[..., c] - src_mean[c]) * scale[c] + ref_mean[c]

    # Clip LAB to valid ranges
    result_lab[..., 0] = result_lab[..., 0].clip(0, 100)
    result_lab[..., 1] = result_lab[..., 1].clip(-128, 127)
    result_lab[..., 2] = result_lab[..., 2].clip(-128, 127)

    return lab_to_rgb(result_lab.astype(np.float32))


def extract_dinov2_embedding_cpu(image: Image.Image, model, processor) -> np.ndarray:
    """Extract DINOv2 CLS embedding on CPU."""
    import torch
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].numpy()
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb[0].astype(np.float32)


def compute_color_baseline_metrics(
    nonleica_paths: list,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    output_dir: Path,
    compute_embeddings: bool = True,
    n_eval_images: int = 50,
) -> dict:
    """
    Run color transfer on non-Leica images and compute gate metrics.

    Returns dict with embedding shift and sample paths.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    import random
    random.seed(42)
    eval_paths = random.sample(nonleica_paths, min(n_eval_images, len(nonleica_paths)))

    # Load DINOv2 for embedding extraction (CPU is fine for 50 images)
    results = {"method": "reinhard_lab_color_transfer", "n_images": len(eval_paths)}

    if compute_embeddings:
        print("Loading DINOv2-giant for embedding extraction...")
        HF_CACHE = Path("/mnt/nas-ai-models/huggingface-cache")
        processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-giant", cache_dir=str(HF_CACHE), local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            "facebook/dinov2-giant", cache_dir=str(HF_CACHE), local_files_only=True,
        ).eval()

        leica_centroid = load_leica_centroid("dinov2-giant", "cls")
        pre_embs = []
        post_embs = []

    # Process images
    sample_idx = 0
    for src_path in tqdm(eval_paths, desc="Color transfer"):
        # Load source
        src_img = Image.open(src_path).convert("RGB")

        # Apply color transfer
        transferred = apply_color_transfer(src_path, ref_mean, ref_std)

        # Save sample side-by-side (first 10)
        if sample_idx < 10:
            trans_pil = Image.fromarray(transferred)
            # Side-by-side: source | transferred
            combined = Image.new("RGB", (src_img.width * 2, src_img.height))
            combined.paste(src_img, (0, 0))
            combined.paste(trans_pil, (src_img.width, 0))
            combined.save(samples_dir / f"sample_{sample_idx:03d}.png")
            sample_idx += 1

        # Extract embeddings
        if compute_embeddings:
            # Pre (source)
            pre_emb = extract_dinov2_embedding_cpu(src_img, model, processor)
            pre_embs.append(pre_emb)
            # Post (color-transferred)
            post_emb = extract_dinov2_embedding_cpu(Image.fromarray(transferred), model, processor)
            post_embs.append(post_emb)

    if compute_embeddings:
        pre_embs = np.array(pre_embs)
        post_embs = np.array(post_embs)

        emb_result = embedding_shift(pre_embs, post_embs, leica_centroid)
        results["embedding_shift"] = emb_result

        print(f"\nColor baseline DINOv2 Δ: {emb_result['delta_mean']:.4f}")
        print(f"  Pre mean dist: {emb_result['pre_mean_distance']:.4f}")
        print(f"  Post mean dist: {emb_result['post_mean_distance']:.4f}")
        print(f"  % with positive Δ: {emb_result['delta_positive_pct']:.1f}%")
        print(f"  % with Δ ≥ 0.02: {emb_result['delta_significant_pct']:.1f}%")
    else:
        results["embedding_shift"] = None

    # Save results
    results_path = output_dir / "color_transfer_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults: {results_path}")
    print(f"Samples: {samples_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Color-transfer baseline for Phase 2a")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for color baseline (e.g., runs/color_baseline/2026-08-08_0000)")
    parser.add_argument("--images-only", action="store_true",
                        help="Only generate color-transferred images, skip embedding extraction")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Number of non-Leica images to process")
    parser.add_argument("--max-ref-images", type=int, default=100,
                        help="Max Leica images for computing reference stats")
    args = parser.parse_args()

    print("=" * 60)
    print("Color-Transfer Baseline Control Arm")
    print("=" * 60)

    # Load Leica reference images
    print("\nLoading Leica reference images...")
    leica_paths = load_image_paths(REGISTRY_PATH, class_filter="positive")
    print(f"  {len(leica_paths)} Leica images available")

    # Compute reference LAB stats
    print("\nComputing Leica LAB reference stats...")
    lab_stats = compute_lab_stats(leica_paths, max_images=args.max_ref_images)
    ref_mean = np.array([lab_stats["L_mean"], lab_stats["A_mean"], lab_stats["B_mean"]])
    ref_std = np.array([lab_stats["L_std"], lab_stats["A_std"], lab_stats["B_std"]])
    print(f"  L: mean={ref_mean[0]:.1f}, std={ref_std[0]:.1f}")
    print(f"  A: mean={ref_mean[1]:.1f}, std={ref_std[1]:.1f}")
    print(f"  B: mean={ref_mean[2]:.1f}, std={ref_std[2]:.1f}")

    # Load non-Leica images
    print("\nLoading non-Leica eval images...")
    nonleica_paths = load_image_paths(REGISTRY_PATH, class_filter="negative")
    print(f"  {len(nonleica_paths)} non-Leica images available")

    # Run color transfer baseline
    results = compute_color_baseline_metrics(
        nonleica_paths, ref_mean, ref_std,
        output_dir=args.output_dir,
        compute_embeddings=not args.images_only,
        n_eval_images=args.n_eval,
    )

    # Save LAB stats
    stats_path = args.output_dir / "lab_reference_stats.json"
    with open(stats_path, "w") as f:
        json.dump(lab_stats, f, indent=2)
    print(f"LAB reference: {stats_path}")

    print("\nDone. Color baseline ready for gate comparison.")


if __name__ == "__main__":
    main()
