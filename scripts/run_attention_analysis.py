#!/usr/bin/env python3
"""
Attention-map analysis — Issue #12
====================================
Extract DINOv2 CLS-to-patch attention maps for a sample of Leica/non-Leica images.
Compute radial attention profiles to determine if classification is driven by
image center (content) or edges/corners (lens rendering: bokeh, vignetting).

Usage:
    python3 scripts/run_attention_analysis.py [--model dinov2-base] [--n-samples 50]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path("/mnt/nas-ai-models/training-data/leica-look")
# When running from repo, use this:
# REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = Path("/mnt/nas-ai-models/training-data/leica-look/experiments/discriminator-attention")
OUTPUT_JSON = OUTPUT_DIR / "attention_profiles.json"
OUTPUT_MD = OUTPUT_DIR / "analysis.md"

HF_HOME = os.environ.get("HF_HOME", "/mnt/nas-ai-models/huggingface-cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DINOv2 model IDs
MODEL_IDS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "dinov2-giant": "facebook/dinov2-giant",
}

IMAGE_SIZE = 518  # DINOv2 standard size
PATCH_SIZE = 14
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 37×37 = 1369 patches


def load_verified_metadata():
    """Return list of (flickr_id, class_label, file_path) from verified.csv."""
    entries = []
    # Look for verified.csv in NAS training-data root (where it was copied)
    csv_path = Path("/mnt/nas-ai-models/training-data/leica-look/verified.csv")
    if not csv_path.exists():
        csv_path = Path("/home/tim/source/activity/leica-look/data/registry/verified.csv")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            entries.append({
                "flickr_id": row["flickr_id"],
                "label": 1 if row["class"] == "positive" else 0,
                "file_path": row["file_path"],
                "class_name": row["class"],
            })
    return entries


def compute_radial_profile(attn_map: np.ndarray, n_bins: int = 10):
    """
    Compute average attention vs. normalized distance from image center.

    attn_map: (grid_size, grid_size) CLS-to-patch attention (averaged across heads)
    n_bins: number of radial bins

    Returns: (bin_centers, mean_attention_per_bin, std_attention_per_bin)
    """
    grid_size = attn_map.shape[0]
    h, w = grid_size, grid_size
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0

    # Compute distance from center for each patch, normalized to [0, 1]
    yy, xx = np.mgrid[0:h, 0:w]
    distances = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    max_dist = np.sqrt(center_y ** 2 + center_x ** 2)
    norm_distances = distances / max_dist  # [0, 1]

    # Bin by normalized distance
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mean_attn = np.zeros(n_bins)
    std_attn = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (norm_distances >= bin_edges[i]) & (norm_distances < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_attn[i] = attn_map[mask].mean()
            std_attn[i] = attn_map[mask].std()

    return bin_centers, mean_attn, std_attn


def extract_attention(model, processor, image_path: str):
    """
    Extract CLS-to-patch attention from the last transformer block.
    Returns (grid_size, grid_size) attention map averaged across heads.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  ⚠️  Cannot load {image_path}: {e}")
        return None

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (batch, n_heads, n_patches+1, n_patches+1)
    # Last layer, average across heads, extract CLS token attention to patches
    last_attn = outputs.attentions[-1]  # (1, n_heads, N+1, N+1)
    cls_attn = last_attn[0, :, 0, 1:].mean(dim=0)  # (N,) — average across heads, CLS→patches
    cls_attn = cls_attn.cpu().numpy()

    # Reshape to 2D grid — compute grid size dynamically
    n_patches = len(cls_attn)
    grid_size = int(np.sqrt(n_patches))
    if grid_size * grid_size != n_patches:
        # If not a perfect square, pad or handle differently
        print(f"  ⚠️  Non-square patches: {n_patches} patches (grid_size ~{grid_size})")
        return None
    attn_map = cls_attn.reshape(grid_size, grid_size)

    return attn_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dinov2-base",
                       choices=list(MODEL_IDS.keys()))
    parser.add_argument("--n-samples", type=int, default=50,
                       help="Number of images per class to analyze")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}")

    # Load model
    model_id = MODEL_IDS[args.model]
    print(f"Loading {model_id}...")
    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=HF_HOME)
    # Use eager attention to get attention weights (SDPA doesn't return them)
    model = AutoModel.from_pretrained(
        model_id, cache_dir=HF_HOME,
        attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    # Load metadata
    entries = load_verified_metadata()
    pos_entries = [e for e in entries if e["label"] == 1]
    neg_entries = [e for e in entries if e["label"] == 0]
    print(f"Dataset: {len(pos_entries)} Leica, {len(neg_entries)} non-Leica")

    # Sample
    rng = np.random.RandomState(42)
    n_per_class = min(args.n_samples, len(pos_entries), len(neg_entries))
    pos_sample = rng.choice(pos_entries, size=n_per_class, replace=False)
    neg_sample = rng.choice(neg_entries, size=n_per_class, replace=False)

    # Extract attention maps
    all_profiles = {"positive": [], "negative": []}

    for class_name, sample in [("positive", pos_sample), ("negative", neg_sample)]:
        print(f"\nProcessing {class_name} ({len(sample)} images)...")
        for i, entry in enumerate(sample):
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(sample)}")

            attn_map = extract_attention(model, processor, entry["file_path"])
            if attn_map is None:
                continue

            bin_centers, mean_attn, std_attn = compute_radial_profile(attn_map)

            all_profiles[class_name].append({
                "flickr_id": entry["flickr_id"],
                "bin_centers": bin_centers.tolist(),
                "mean_attention": mean_attn.tolist(),
                "std_attention": std_attn.tolist(),
            })

    # Save raw data
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_profiles, f, indent=2)
    print(f"\nSaved attention profiles to {OUTPUT_JSON}")

    # ── Aggregate analysis ──
    pos_agg = np.array([p["mean_attention"] for p in all_profiles["positive"]])
    neg_agg = np.array([p["mean_attention"] for p in all_profiles["negative"]])

    if len(pos_agg) == 0 or len(neg_agg) == 0:
        print("ERROR: No attention maps extracted")
        return

    pos_mean = pos_agg.mean(axis=0)
    pos_std = pos_agg.std(axis=0)
    neg_mean = neg_agg.mean(axis=0)
    neg_std = neg_agg.std(axis=0)

    bin_centers = np.array(all_profiles["positive"][0]["bin_centers"])

    # Center-to-edge ratio: compare inner 2 bins vs outer 2 bins
    inner_mask = bin_centers < 0.3
    outer_mask = bin_centers > 0.7

    pos_center_ratio = pos_mean[inner_mask].mean() / (pos_mean[outer_mask].mean() + 1e-10)
    neg_center_ratio = neg_mean[inner_mask].mean() / (neg_mean[outer_mask].mean() + 1e-10)

    # ── Write analysis ──
    with open(OUTPUT_MD, "w") as f:
        f.write(f"# DINOv2 Attention-Map Analysis\n\n")
        f.write(f"**Date:** 2026-08-07  \n")
        f.write(f"**Model:** {args.model} ({model_id})  \n")
        f.write(f"**Images analyzed:** {len(all_profiles['positive'])} Leica, "
                f"{len(all_profiles['negative'])} non-Leica  \n\n")

        f.write(f"## Radial Attention Profile\n\n")
        f.write(f"Average CLS-to-patch attention (last layer, averaged across heads) "
                f"vs. normalized distance from image center.\n\n")

        f.write(f"| Radial bin | Leica mean | Leica σ | Non-Leica mean | Non-Leica σ | Δ (Leica − non) |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for i in range(len(bin_centers)):
            delta = pos_mean[i] - neg_mean[i]
            f.write(f"| {bin_centers[i]:.2f} | {pos_mean[i]:.6f} | {pos_std[i]:.6f} | "
                    f"{neg_mean[i]:.6f} | {neg_std[i]:.6f} | {delta:+.6f} |\n")

        f.write(f"\n## Center-to-Edge Ratio\n\n")
        f.write(f"Ratio of mean attention in inner 30% vs outer 30% of image:\n\n")
        f.write(f"- **Leica:** {pos_center_ratio:.2f}× (center-weighted)\n")
        f.write(f"- **Non-Leica:** {neg_center_ratio:.2f}× (center-weighted)\n\n")

        # Interpretation
        if pos_center_ratio > neg_center_ratio + 0.1:
            verdict = "Leica images are MORE center-weighted — the model attends more to the subject in Leica images. This supports the content-confound interpretation: Leica photographers' subject choices differ."
        elif neg_center_ratio > pos_center_ratio + 0.1:
            verdict = "Non-Leica images are MORE center-weighted — Leica images receive more edge/corner attention. This supports the lens-rendering interpretation: bokeh, vignetting, and focus falloff are driving the classification."
        else:
            verdict = "No strong difference in center-weighting between Leica and non-Leica images. The DINOv2 CLS token attends similarly to both classes, suggesting the signal is distributed across the image rather than concentrated in center (subject) or edges (lens artifacts)."

        f.write(f"**Verdict:** {verdict}\n\n")

        # Additional stats
        f.write(f"## Full-Radius Comparison\n\n")
        total_pos = pos_mean.sum()
        total_neg = neg_mean.sum()
        # Normalize to compare distribution shape
        pos_norm = pos_mean / total_pos
        neg_norm = neg_mean / total_neg

        f.write(f"### Normalized attention distribution\n\n")
        f.write(f"| Radial bin | Leica (norm) | Non-Leica (norm) | Δ |\n")
        f.write(f"|---|---|---|---|\n")
        for i in range(len(bin_centers)):
            delta = pos_norm[i] - neg_norm[i]
            f.write(f"| {bin_centers[i]:.2f} | {pos_norm[i]:.4f} | {neg_norm[i]:.4f} | {delta:+.4f} |\n")

        f.write(f"\n## Data\n\n")
        f.write(f"- Raw profiles: `{OUTPUT_JSON.name}`\n")
        f.write(f"- Model: `{args.model}`\n")
        f.write(f"- Images per class: {n_per_class}\n")

    print(f"\nAnalysis written to {OUTPUT_MD}")
    print(open(OUTPUT_MD).read())


if __name__ == "__main__":
    main()
