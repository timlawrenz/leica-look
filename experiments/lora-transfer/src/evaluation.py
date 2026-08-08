"""
Evaluation metrics for Phase 2a LoRA transfer.

Implements the three pre-registered gate criteria:
  1. DINOv2 embedding shift: cosine distance pre-transfer vs post-transfer
  2. Attention-map C/E ratio: center/edge attention ratio shift
  3. CLIP-I: content preservation (cosine similarity)

All metrics run on CPU using pre-extracted embeddings from Phase 1.
No GPU required for evaluation — only for the FLUX inference that produces
the post-transfer images.

Pre-registered gate (from provenance.yaml):
  PASS if: (1) Δ ≥ 0.02, (2) C/E Δ ≤ -0.10, (3) CLIP-I ≥ 0.85
  PENDING if any 2 of 3 met. FAIL if ≤ 1 met.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

EMBEDDINGS_DIR = Path("/mnt/nas-ai-models/training-data/leica-look/embeddings")
MATCHED_DIR = Path("/home/tim/source/activity/leica-look/experiments/discriminator-content-matched")
ATTENTION_DIR = Path("/home/tim/source/activity/leica-look/experiments/discriminator-attention")
EXPERIMENT_DIR = Path("/home/tim/source/activity/leica-look/experiments/lora-transfer")
BASELINE_PATH = Path(
    "/home/tim/source/activity/leica-look/experiments/lora-transfer/baseline_cosine_distances.json"
)


def load_embeddings(model: str = "dinov2-giant", pooling: str = "cls"):
    """Load pre-extracted embeddings and labels."""
    emb_path = EMBEDDINGS_DIR / model / f"{pooling}.npy"
    labels_path = EMBEDDINGS_DIR / "labels.npy"
    ids_path = EMBEDDINGS_DIR / "image_ids.txt"

    embs = np.load(emb_path).astype(np.float32)
    labels = np.load(labels_path).astype(int)
    with open(ids_path) as f:
        ids = [line.strip() for line in f]

    # Normalize
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)

    leica_mask = labels == 1
    nonleica_mask = labels == 0

    return {
        "embs": embs,
        "labels": labels,
        "ids": ids,
        "leica_embs": embs[leica_mask],
        "leica_ids": [ids[i] for i in range(len(ids)) if leica_mask[i]],
        "nonleica_embs": embs[nonleica_mask],
        "nonleica_ids": [ids[i] for i in range(len(ids)) if nonleica_mask[i]],
    }


def load_leica_centroid(model: str = "dinov2-giant", pooling: str = "cls") -> np.ndarray:
    """Compute the Leica embedding centroid (reference distribution center) from ALL Leica images."""
    data = load_embeddings(model, pooling)
    centroid = data["leica_embs"].mean(axis=0)
    return centroid / np.linalg.norm(centroid)


def load_held_out_centroid(
    model: str = "dinov2-giant",
    pooling: str = "cls",
    holdout_ids: Optional[set] = None,
) -> np.ndarray:
    """
    Compute the held-out Leica embedding centroid (FIX #2).

    This is the centroid of ONLY the held-out Leica images (not used in training).
    The gate must confirm post-transfer shift toward THIS centroid for generalization,
    not just the full-set centroid (which includes training images — memorization risk).

    Args:
        model: Embedding model name
        pooling: Pooling strategy
        holdout_ids: Set of held-out image IDs. If None, loads from held_out_split.json.

    Returns:
        Normalized centroid vector of held-out images only.
    """
    if holdout_ids is None:
        # Try to load from the held-out split
        split_path = EXPERIMENT_DIR / "held_out_split.json"
        if split_path.exists():
            import json
            with open(split_path) as f:
                split = json.load(f)
            holdout_ids = set(split["holdout_ids"])
        else:
            # Fall back to full centroid
            print("WARNING: held_out_split.json not found. Using full Leica centroid.")
            return load_leica_centroid(model, pooling)

    data = load_embeddings(model, pooling)
    # Match image IDs to find held-out embeddings
    # Image IDs format: "flickr_id" — match against holdout_ids
    holdout_indices = []
    for i, img_id in enumerate(data["ids"]):
        # IDs may be stored as paths or flickr IDs; try both
        img_flickr_id = Path(img_id).stem if "/" in img_id else img_id
        if img_id in holdout_ids or img_flickr_id in holdout_ids:
            # Check it's a Leica image
            if data["labels"][i] == 1:
                holdout_indices.append(i)

    if not holdout_indices:
        print(f"WARNING: Could not match any held-out IDs to embeddings. Using full centroid.")
        return load_leica_centroid(model, pooling)

    heldout_embs = data["embs"][holdout_indices]
    print(f"Held-out centroid: {len(holdout_indices)} images (of {len(holdout_ids)} requested)")
    centroid = heldout_embs.mean(axis=0)
    return centroid / np.linalg.norm(centroid)


def load_baseline() -> dict:
    """Load pre-computed baseline calibration from #14 gate calibration."""
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Metric 1: DINOv2 embedding shift
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two normalized vectors."""
    return float(1.0 - np.dot(a, b))


def embedding_shift(
    pretransfer_embs: np.ndarray,
    posttransfer_embs: np.ndarray,
    leica_centroid: np.ndarray,
) -> dict:
    """
    Measure how much the post-transfer embeddings shift toward the Leica
    reference distribution.

    Metric: Δ = mean(cosine_distance(pre, centroid)) - mean(cosine_distance(post, centroid))

    Positive Δ means post-transfer images are closer to the Leica centroid.

    Args:
        pretransfer_embs: (N, D) normalized embeddings of non-Leica images BEFORE transfer
        posttransfer_embs: (N, D) normalized embeddings of the same images AFTER transfer
        leica_centroid: (D,) normalized Leica embedding centroid

    Returns:
        dict with per-image distances and summary statistics
    """
    pre_to_centroid = np.array([
        cosine_distance(pretransfer_embs[i], leica_centroid)
        for i in range(len(pretransfer_embs))
    ])
    post_to_centroid = np.array([
        cosine_distance(posttransfer_embs[i], leica_centroid)
        for i in range(len(posttransfer_embs))
    ])

    delta = pre_to_centroid - post_to_centroid  # per-image shift (positive = closer to Leica)

    return {
        "n_images": len(pretransfer_embs),
        "pre_mean_distance": float(np.mean(pre_to_centroid)),
        "pre_std_distance": float(np.std(pre_to_centroid)),
        "post_mean_distance": float(np.mean(post_to_centroid)),
        "post_std_distance": float(np.std(post_to_centroid)),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
        "delta_positive_pct": float(np.mean(delta > 0) * 100),
        "delta_significant_pct": float(np.mean(delta >= 0.02) * 100),
        "pre_distances": pre_to_centroid.tolist(),
        "post_distances": post_to_centroid.tolist(),
    }


# ---------------------------------------------------------------------------
# Metric 2: Attention-map C/E ratio
# ---------------------------------------------------------------------------

def center_edge_ratio(
    attention_map: np.ndarray,
    radial_bins: int = 10,
    crossover_bin: int = 6,
) -> float:
    """
    Compute the center/edge attention ratio from a 2D attention map.

    Lower ratio = more edge attention = more lens-rendering signal (per #12).

    Args:
        attention_map: (H, W) or (H*W,) attention weights
        radial_bins: Number of radial bins
        crossover_bin: Bin index dividing center (0:crossover) from edge (crossover:end)

    Returns:
        center/edge ratio (dimensionless)
    """
    if attention_map.ndim == 1:
        # Already a radial profile
        profile = attention_map
    else:
        # 2D map: convert to radial profile
        h, w = attention_map.shape
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        yy, xx = np.meshgrid(y, x)
        r = np.sqrt(xx**2 + yy**2) / np.sqrt(2.0)  # normalized [0, 1]

        bin_idx = np.clip((r * radial_bins).astype(int), 0, radial_bins - 1)
        profile = np.array([
            attention_map[bin_idx == b].mean()
            for b in range(radial_bins)
        ])

    center_attn = profile[:crossover_bin].mean()
    edge_attn = profile[crossover_bin:].mean()

    if edge_attn == 0:
        return float("inf")

    return float(center_attn / edge_attn)


def attention_shift(
    pretransfer_ratios: np.ndarray,
    posttransfer_ratios: np.ndarray,
) -> dict:
    """
    Measure the shift in center/edge attention ratio after transfer.

    Negative delta means post-transfer images have lower C/E ratio
    (more edge attention) — consistent with the #12 hypothesis.

    Args:
        pretransfer_ratios: (N,) C/E ratios before transfer
        posttransfer_ratios: (N,) C/E ratios after transfer

    Returns:
        dict with per-image ratios and summary statistics
    """
    delta = posttransfer_ratios - pretransfer_ratios  # negative = more edge attention

    return {
        "n_images": len(pretransfer_ratios),
        "pre_mean_ratio": float(np.mean(pretransfer_ratios)),
        "pre_std_ratio": float(np.std(pretransfer_ratios)),
        "post_mean_ratio": float(np.mean(posttransfer_ratios)),
        "post_std_ratio": float(np.std(posttransfer_ratios)),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
        "delta_negative_pct": float(np.mean(delta < 0) * 100),
        "delta_significant_pct": float(np.mean(delta <= -0.10) * 100),
        "pre_ratios": pretransfer_ratios.tolist(),
        "post_ratios": posttransfer_ratios.tolist(),
    }


# ---------------------------------------------------------------------------
# Metric 3: CLIP-I content preservation
# ---------------------------------------------------------------------------

def clip_image_similarity(
    pretransfer_embs: np.ndarray,
    posttransfer_embs: np.ndarray,
) -> dict:
    """
    Compute CLIP image-to-image cosine similarity (CLIP-I).
    High similarity means content is preserved during transfer.

    Uses CLIP ViT-L/14 embeddings (pre-extracted in Phase 1).

    Args:
        pretransfer_embs: (N, D) normalized CLIP embeddings of input images
        posttransfer_embs: (N, D) normalized CLIP embeddings of transfer outputs

    Returns:
        dict with per-image similarities and summary statistics
    """
    similarities = np.array([
        float(np.dot(pretransfer_embs[i], posttransfer_embs[i]))
        for i in range(len(pretransfer_embs))
    ])

    return {
        "n_images": len(pretransfer_embs),
        "mean_clip_i": float(np.mean(similarities)),
        "std_clip_i": float(np.std(similarities)),
        "min_clip_i": float(np.min(similarities)),
        "max_clip_i": float(np.max(similarities)),
        "p5_clip_i": float(np.percentile(similarities, 5)),
        "pct_above_threshold_0_85": float(np.mean(similarities >= 0.85) * 100),
        "per_image_similarities": similarities.tolist(),
    }


# ---------------------------------------------------------------------------
# Combined gate evaluation
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Result of evaluating the three-gate criteria."""
    embedding_delta: float
    attention_delta: float
    clip_i: float
    verdict: str  # PASS, PENDING, or FAIL
    criteria_met: int
    details: dict = field(default_factory=dict)
    # FIX #2: Held-out centroid comparison
    held_out_delta: Optional[float] = None
    held_out_verdict: Optional[str] = None  # generalization check
    # FIX #1: Color-transfer baseline comparison
    color_baseline_delta: Optional[float] = None
    color_baseline_comparison: Optional[str] = None  # "LoRA > color", "≈ color", etc.


def evaluate_gate(
    embedding_result: dict,
    attention_result: dict,
    clip_result: dict,
    held_out_embedding_result: Optional[dict] = None,
    color_baseline_result: Optional[dict] = None,
) -> GateResult:
    """
    Evaluate the pre-registered gate.

    PASS if: (1) Δ ≥ 0.02 AND (2) C/E Δ ≤ -0.10 AND (3) CLIP-I ≥ 0.85
    PENDING if any 2 of 3 met.
    FAIL if ≤ 1 met.

    FIX #2: If held_out_embedding_result is provided, also check generalization:
      held_out_delta (shift toward held-out centroid) ≥ 0.02 for generalization.
      NOTE: held-out check is reported but does NOT change the primary PASS/FAIL.
      It's a diagnostic: if full-centroid passes but held-out fails, the LoRA may
      be memorizing the training set rather than learning a general look.

    FIX #1: If color_baseline_result is provided, compare LoRA Δ vs color Δ:
      - If LoRA Δ >> color Δ → the look is more than color science
      - If LoRA Δ ≈ color Δ → the 'Leica look' is primarily color science
    """
    emb_delta = embedding_result["delta_mean"]
    attn_delta = attention_result["delta_mean"]
    clip_i = clip_result["mean_clip_i"]

    criteria = [
        ("embedding_shift", emb_delta >= 0.02, emb_delta),
        ("attention_shift", attn_delta <= -0.10, attn_delta),
        ("clip_i_preservation", clip_i >= 0.85, clip_i),
    ]

    met = sum(1 for _, passed, _ in criteria if passed)
    failed = [name for name, passed, _ in criteria if not passed]

    if met == 3:
        verdict = "PASS"
    elif met >= 2:
        verdict = "PENDING"
    else:
        verdict = "FAIL"

    result = GateResult(
        embedding_delta=emb_delta,
        attention_delta=attn_delta,
        clip_i=clip_i,
        verdict=verdict,
        criteria_met=met,
        details={
            "criteria": [
                {"name": n, "passed": p, "value": round(v, 6)}
                for n, p, v in criteria
            ],
            "failed": failed,
        },
    )

    # FIX #2: Held-out generalization check
    if held_out_embedding_result is not None:
        ho_delta = held_out_embedding_result["delta_mean"]
        result.held_out_delta = ho_delta
        if ho_delta >= 0.02:
            result.held_out_verdict = "GENERALIZES"
        elif ho_delta >= 0.01:
            result.held_out_verdict = "WEAK_GENERALIZATION"
        else:
            result.held_out_verdict = "MEMORIZATION_RISK"
        result.details["held_out"] = {
            "delta_mean": round(ho_delta, 6),
            "verdict": result.held_out_verdict,
            "pre_mean_distance": round(held_out_embedding_result["pre_mean_distance"], 6),
            "post_mean_distance": round(held_out_embedding_result["post_mean_distance"], 6),
        }

    # FIX #1: Color-transfer baseline comparison
    if color_baseline_result is not None:
        cb_result = color_baseline_result
        if isinstance(cb_result, dict) and "embedding_shift" in cb_result:
            cb_delta = cb_result["embedding_shift"]["delta_mean"]
        elif isinstance(cb_result, dict) and "delta_mean" in cb_result:
            cb_delta = cb_result["delta_mean"]
        else:
            cb_delta = None

        result.color_baseline_delta = cb_delta

        if cb_delta is not None:
            # Compare LoRA vs color baseline
            ratio = emb_delta / max(cb_delta, 0.001)  # avoid div by 0
            if ratio >= 2.0:
                result.color_baseline_comparison = "LoRA >> color (2x+) — optical signal dominant"
            elif ratio >= 1.3:
                result.color_baseline_comparison = "LoRA > color — optical signal present, color contributes"
            elif ratio >= 0.7:
                result.color_baseline_comparison = "LoRA ≈ color — look is primarily color science"
            else:
                result.color_baseline_comparison = "LoRA < color — color transfer outperforms LoRA"

            result.details["color_baseline"] = {
                "delta_mean": round(cb_delta, 6),
                "lora_to_color_ratio": round(ratio, 3),
                "comparison": result.color_baseline_comparison,
            }
        else:
            result.color_baseline_comparison = "no_color_baseline_data"
            result.details["color_baseline"] = {"status": "unavailable"}

    return result
