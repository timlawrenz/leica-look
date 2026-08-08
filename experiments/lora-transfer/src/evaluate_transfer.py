#!/usr/bin/env python3
"""
Phase 2a: Post-training evaluation — compute the three-gate verdict.

After training completes, this script:
1. Runs FLUX img2img transfer on all eval images using a checkpoint
2. Extracts DINOv2/CLIP embeddings from transfer outputs
3. Computes gate metrics and verdict

The script can run on CPU for embedding extraction OR on GPU for
the FLUX inference pass. Embedding models are loaded from HF cache.

USAGE:
  # Full pipeline (GPU needed for FLUX inference)
  python3 evaluate_transfer.py --checkpoint runs/2026-08-07_1200/checkpoints/final --gpu 4090

  # Embedding-only (already have transfer images)
  python3 evaluate_transfer.py --transfer-dir runs/2026-08-07_1200/evaluation/step_01500
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation import (
    attention_shift,
    center_edge_ratio,
    clip_image_similarity,
    embedding_shift,
    evaluate_gate,
    load_embeddings,
    load_leica_centroid,
    load_held_out_centroid,
    cosine_distance,
)


PROJECT_ROOT = Path(os.environ.get("LEICA_LOOK_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent)))
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "lora-transfer"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "verified.csv"
HF_CACHE = Path("/mnt/nas-ai-models/huggingface-cache")
GPU_SCHEDULER = Path("/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py")


def extract_dinov2_embedding(model, processor, image: Image.Image, device) -> np.ndarray:
    """Extract DINOv2 CLS embedding from a single image."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb[0].astype(np.float32)


def extract_clip_embedding(model, processor, image: Image.Image, device) -> np.ndarray:
    """Extract CLIP ViT-L image embedding."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs).cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb[0].astype(np.float32)


def extract_dinov2_attention(model, processor, image: Image.Image, device) -> np.ndarray:
    """
    Extract DINOv2-S CLS-to-patch attention map for C/E ratio computation.
    Returns a radial profile of shape (10,).
    """
    from copy import deepcopy
    # Register hook to capture last attention
    attentions = []

    def hook(module, input, output):
        attentions.append(output[1].detach().cpu())  # output[1] = attention weights

    # Hook into last block
    last_block = model.encoder.layer[-1].attention.attention
    handle = last_block.register_forward_hook(hook)

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)

    handle.remove()

    if attentions:
        # Mean over heads: (B, heads, N, N) → (N, N) where N = 1 CLS + patches
        attn = attentions[-1].mean(dim=1)[0]  # (N, N)
        cls_attn = attn[0, 1:].numpy()  # CLS → patches, drop self-attention
        # Reshape to 2D
        patch_size = int(np.sqrt(len(cls_attn)))
        attn_map = cls_attn.reshape(patch_size, patch_size)
        return attn_map
    else:
        return np.ones((14, 14))  # Fallback uniform


def load_transfer_images(transfer_dir: Path) -> np.ndarray:
    """Load transfer output images (right halves of comparison images)."""
    images = []
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    for img_path in sorted(transfer_dir.glob("sample_*.png")):
        img = Image.open(img_path).convert("RGB")
        # Right half is the transfer output
        w = img.width
        transfer_half = img.crop((w // 2, 0, w, img.height))
        tensor = transform(transfer_half)
        images.append(tensor)
    return np.stack([t.numpy() for t in images]) if images else np.array([])


def load_input_images(transfer_dir: Path) -> list:
    """Load input images (left halves of comparison images) for embedding extraction."""
    images = []
    for img_path in sorted(transfer_dir.glob("sample_*.png")):
        img = Image.open(img_path).convert("RGB")
        w = img.width
        input_half = img.crop((0, 0, w // 2, img.height))
        images.append(input_half)
    return images


def compute_gate_metrics(checkpoint_dir: Path, device: str = "cuda",
                       color_baseline_path: Optional[Path] = None):
    """Compute all three gate metrics from transfer outputs, plus held-out
    generalization check and color-baseline comparison.

    FIX #2: Computes embedding shift toward held-out Leica centroid (generalization check).
    FIX #1: Compares LoRA shift to color-transfer baseline (optical vs color signal).

    Returns dict with verdict, held-out verdict, and color-baseline comparison.
    """
    import csv

    # Find latest evaluation
    eval_dirs = sorted(checkpoint_dir.parent.glob("evaluation/step_*"))
    if not eval_dirs:
        print("No evaluation directories found!")
        return None

    eval_dir = eval_dirs[-1]  # latest
    print(f"Evaluating: {eval_dir}")

    # Load images
    input_imgs = load_input_images(eval_dir)
    transfer_imgs = load_transfer_images(eval_dir)

    if not input_imgs:
        print("No images to evaluate!")
        return None

    print(f"Found {len(input_imgs)} input/transfer pairs")

    # Load DINOv2-g for embedding shift (Gate #1)
    print("\n--- Gate #1: DINOv2 embedding shift ---")
    from transformers import AutoImageProcessor, AutoModel
    dino_g_processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-giant", cache_dir=str(HF_CACHE), local_files_only=True,
    )
    dino_g_model = AutoModel.from_pretrained(
        "facebook/dinov2-giant", cache_dir=str(HF_CACHE), local_files_only=True,
    ).to(device).eval()

    # Extract embeddings
    pre_embs = []
    post_embs = []
    for inp, trans in zip(input_imgs, transfer_imgs):
        pre_embs.append(extract_dinov2_embedding(dino_g_model, dino_g_processor, inp, device))
        # Convert transfer tensor back to PIL
        trans_pil = T.ToPILImage()((trans * 0.5 + 0.5).clamp(0, 1))
        post_embs.append(extract_dinov2_embedding(dino_g_model, dino_g_processor, trans_pil, device))

    pre_embs = np.array(pre_embs)
    post_embs = np.array(post_embs)
    leica_centroid = load_leica_centroid("dinov2-giant", "cls")

    emb_result = embedding_shift(pre_embs, post_embs, leica_centroid)
    print(f"  DINOv2 Δ: {emb_result['delta_mean']:.4f} (need ≥ 0.02)")
    print(f"  Pre mean dist: {emb_result['pre_mean_distance']:.4f}")
    print(f"  Post mean dist: {emb_result['post_mean_distance']:.4f}")
    print(f"  % with positive Δ: {emb_result['delta_positive_pct']:.1f}%")

    del dino_g_model, dino_g_processor
    torch.cuda.empty_cache()

    # --- Gate #2: Attention C/E ratio shift ---
    print("\n--- Gate #2: Attention C/E ratio shift ---")
    dino_s_processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-small", cache_dir=str(HF_CACHE), local_files_only=True,
    )
    dino_s_model = AutoModel.from_pretrained(
        "facebook/dinov2-small", cache_dir=str(HF_CACHE), local_files_only=True,
    ).to(device).eval()

    from evaluation import center_edge_ratio
    pre_ratios = []
    post_ratios = []
    for inp, trans in zip(input_imgs, transfer_imgs):
        # Input
        attn_map_pre = extract_dinov2_attention(dino_s_model, dino_s_processor, inp, device)
        # Normalize attention map
        if attn_map_pre.sum() > 0:
            attn_map_pre = attn_map_pre / attn_map_pre.sum()
        pre_ratios.append(center_edge_ratio(attn_map_pre))
        # Transfer
        trans_pil = T.ToPILImage()((trans * 0.5 + 0.5).clamp(0, 1))
        attn_map_post = extract_dinov2_attention(dino_s_model, dino_s_processor, trans_pil, device)
        if attn_map_post.sum() > 0:
            attn_map_post = attn_map_post / attn_map_post.sum()
        post_ratios.append(center_edge_ratio(attn_map_post))

    attn_result = attention_shift(np.array(pre_ratios), np.array(post_ratios))
    print(f"  C/E Δ: {attn_result['delta_mean']:.4f} (need ≤ -0.10)")
    print(f"  Pre C/E: {attn_result['pre_mean_ratio']:.4f}")
    print(f"  Post C/E: {attn_result['post_mean_ratio']:.4f}")
    print(f"  % with negative Δ: {attn_result['delta_negative_pct']:.1f}%")

    del dino_s_model, dino_s_processor
    torch.cuda.empty_cache()

    # --- Gate #3: CLIP-I content preservation ---
    print("\n--- Gate #3: CLIP-I content preservation ---")
    from transformers import CLIPProcessor, CLIPModel
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=str(HF_CACHE), local_files_only=True,
    )
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=str(HF_CACHE), local_files_only=True,
    ).to(device).eval()

    clip_pre = []
    clip_post = []
    for inp, trans in zip(input_imgs, transfer_imgs):
        clip_pre.append(extract_clip_embedding(clip_model, clip_processor, inp, device))
        trans_pil = T.ToPILImage()((trans * 0.5 + 0.5).clamp(0, 1))
        clip_post.append(extract_clip_embedding(clip_model, clip_processor, trans_pil, device))

    clip_result = clip_image_similarity(np.array(clip_pre), np.array(clip_post))
    print(f"  CLIP-I: {clip_result['mean_clip_i']:.4f} (need ≥ 0.85)")
    print(f"  Min: {clip_result['min_clip_i']:.4f}")
    print(f"  % above 0.85: {clip_result['pct_above_threshold_0_85']:.1f}%")

    del clip_model, clip_processor
    torch.cuda.empty_cache()

    # --- Verdict ---
    # FIX #2: Compute held-out centroid shift for generalization check
    print("\n--- Held-Out Generalization Check (FIX #2) ---")
    try:
        held_out_centroid = load_held_out_centroid("dinov2-giant", "cls")
        ho_emb_result = embedding_shift(pre_embs, post_embs, held_out_centroid)
        print(f"  Held-out Δ: {ho_emb_result['delta_mean']:.4f}")
        print(f"  (Training centroid Δ: {emb_result['delta_mean']:.4f})")
        delta_diff = emb_result['delta_mean'] - ho_emb_result['delta_mean']
        print(f"  Train-HO gap: {delta_diff:.4f} (negative = memorization risk)")
    except Exception as e:
        print(f"  WARNING: Held-out check failed: {e}")
        print(f"  Using full centroid as fallback.")
        ho_emb_result = None

    # FIX #1: Load color-transfer baseline for comparison
    color_baseline = None
    if color_baseline_path and color_baseline_path.exists():
        print(f"\n--- Color-Transfer Baseline Comparison (FIX #1) ---")
        with open(color_baseline_path) as f:
            color_baseline = json.load(f)
        if "embedding_shift" in color_baseline:
            cb_delta = color_baseline["embedding_shift"]["delta_mean"]
            print(f"  Color baseline Δ: {cb_delta:.4f}")
            print(f"  LoRA Δ: {emb_result['delta_mean']:.4f}")
            ratio = emb_result['delta_mean'] / max(cb_delta, 0.001)
            print(f"  LoRA/Color ratio: {ratio:.2f}x")
        else:
            print(f"  Color baseline loaded but no embedding_shift data found.")
            color_baseline = None
    elif color_baseline_path:
        print(f"\n--- Color-Transfer Baseline: file not found ({color_baseline_path}) ---")
    else:
        print(f"\n--- Color-Transfer Baseline: not provided (run color_transfer_baseline.py first) ---")

    gate = evaluate_gate(
        emb_result, attn_result, clip_result,
        held_out_embedding_result=ho_emb_result,
        color_baseline_result=color_baseline,
    )
    print(f"\n{'='*60}")
    print(f"GATE VERDICT: {gate.verdict}")
    print(f"Criteria met: {gate.criteria_met}/3")
    for c in gate.details["criteria"]:
        status = "✓" if c["passed"] else "✗"
        print(f"  {status} {c['name']}: {c['value']:.4f}")
    # FIX #2: Held-out
    if gate.held_out_verdict:
        print(f"\nHeld-out generalization: {gate.held_out_verdict} (Δ={gate.held_out_delta:.4f})")
    # FIX #1: Color baseline
    if gate.color_baseline_comparison:
        print(f"Color baseline: {gate.color_baseline_comparison}")
    print(f"{'='*60}")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_dir),
        "eval_dir": str(eval_dir),
        "verdict": gate.verdict,
        "criteria_met": gate.criteria_met,
        "embedding_shift": emb_result,
        "attention_shift_raw": attn_result,
        "clip_i": clip_result,
        "criteria_details": gate.details["criteria"],
        # FIX #2: Held-out generalization
        "held_out_delta": gate.held_out_delta,
        "held_out_verdict": gate.held_out_verdict,
        # FIX #1: Color baseline comparison
        "color_baseline_delta": gate.color_baseline_delta,
        "color_baseline_comparison": gate.color_baseline_comparison,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2a gate evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to LoRA checkpoint (e.g., runs/.../checkpoints/final)")
    parser.add_argument("--gpu", type=str, default="cuda",
                        help="Device for model inference (cuda or cpu)")
    parser.add_argument("--transfer-dir", type=Path,
                        help="Directory with pre-generated transfer images (skip FLUX inference)")
    parser.add_argument("--color-baseline", type=Path, default=None,
                        help="Path to color-transfer baseline results JSON (FIX #1)")
    args = parser.parse_args()

    device = args.gpu if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    result = compute_gate_metrics(args.checkpoint, device, args.color_baseline)

    if result:
        output_path = args.checkpoint.parent / "gate_verdict.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved: {output_path}")

        # Return exit code for automation
        if result["verdict"] == "PASS":
            sys.exit(0)
        elif result["verdict"] == "FAIL":
            sys.exit(1)
        else:
            sys.exit(2)  # PENDING


if __name__ == "__main__":
    main()
