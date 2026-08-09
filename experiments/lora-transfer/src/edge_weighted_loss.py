"""
Edge-weighted loss for FLUX LoRA training.

Operationalizes the Phase 1 attention-map finding (#12, GO verdict):
  - Correct classifications use more edge/peripheral features (C/E ratio 2.79)
  - Incorrect classifications over-attend to center (C/E ratio 3.51)
  - Radial profile crossover at bin 6 (~40% from center to corner)

This module generates a spatial weight map that amplifies loss in peripheral
regions relative to center regions, biasing the LoRA toward learning from
image regions where the lens rendering signal is strongest.

The default crossover_radius=0.40 maps to bin 6/10 from the DINOv2-S/14
attention analysis at experiments/discriminator-attention/attention_analysis.json.

No GPU required — pure numpy/torch utility.
"""

import json
from pathlib import Path

import numpy as np
import torch


# Default parameters derived from attention analysis (#12)
DEFAULT_CROSSOVER_RADIUS = 0.40  # bin 6/10 = 40% from center to corner
DEFAULT_EDGE_AMPLIFICATION = 2.0  # edge regions weighted 2× vs center
DEFAULT_CENTER_WEIGHT = 1.0


def radial_weight_map(
    height: int = 1024,
    width: int = 1024,
    crossover_radius: float = 0.40,
    edge_amplification: float = DEFAULT_EDGE_AMPLIFICATION,
    center_weight: float = DEFAULT_CENTER_WEIGHT,
    transition_width: float = 0.15,
) -> torch.Tensor:
    """
    Generate a 2D spatial weight map for edge-weighted loss.

    The map amplifies loss in peripheral regions (outer ~60% of the image)
    where lens rendering characteristics (bokeh, vignetting, microcontrast
    falloff at edges) are most visible.

    Uses a smooth sigmoid transition from center_weight → edge_amplification
    at crossover_radius (derived from attention analysis bin-6 crossover at
    ~40% from center to corner). Unlike binning + Gaussian convolution, this
    is numerically stable for corner pixels and produces the correct
    center=1.0, edge=2.0 profile.

    Args:
        height, width: Output resolution (typically 1024×1024 for FLUX)
        crossover_radius: Normalized radius (0-1) where weight transition
                          reaches 50%. Default 0.40 = bin 6/10 from #12.
        edge_amplification: Multiplier for edge-region loss weight (outer bins)
        center_weight: Multiplier for center-region loss weight (inner bins)
        transition_width: Width of the sigmoid transition zone in normalized
                          radius units. Smaller = sharper transition.

    Returns:
        weight_map: (height, width) float tensor, centered at [0,0].
                   Center ≈ 1.0, corners ≈ edge_amplification.
    """
    # Create radial distance grid (normalized 0=center, 1=corner)
    y = torch.linspace(-1, 1, height)
    x = torch.linspace(-1, 1, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2) / np.sqrt(2.0)  # normalized [0, 1]

    # Smooth sigmoid transition: maps r in [0,1] to weight in [center, edge_amp]
    # sigmoid(t) where t = (r - crossover) / (transition_width / 4)
    # At r=crossover: sigmoid(0) = 0.5 → weight = center + 0.5*(edge - center)
    # At r << crossover: sigmoid(-large) → 0 → weight ≈ center
    # At r >> crossover: sigmoid(+large) → 1 → weight ≈ edge_amp
    steepness = 4.0 / max(transition_width, 0.01)
    t = steepness * (r - crossover_radius)
    blend = torch.sigmoid(t)  # [0, 1] blend factor

    weight_map = center_weight + (edge_amplification - center_weight) * blend

    return weight_map


def edge_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute edge-weighted MSE loss.

    Args:
        pred: (B, C, H, W) predicted latents
        target: (B, C, H, W) target latents
        weight_map: (H, W) spatial weight map from radial_weight_map()
        reduction: 'mean' (default) or 'none'

    Returns:
        Scalar loss if reduction='mean', else (B, C, H, W) per-element loss
    """
    # Per-element squared error: (B, C, H, W)
    # Compute in fp32 to prevent fp16 overflow of (pred-target)^2 — the
    # low-noise timestep regime (sigma<0.2) produces large residuals that
    # overflow fp16's max 65504 when squared, NaNing the loss (observed at
    # step 5 with timestep ranges that include t<200).
    se = (pred.float() - target.float()) ** 2

    # Broadcast spatial weights: (H, W) → (1, 1, H, W)
    w = weight_map.unsqueeze(0).unsqueeze(0).to(pred.device)

    # Weighted MSE
    weighted = se * w

    if reduction == "none":
        return weighted

    # Mean over all weighted elements (preserving the mean-of-valid convention)
    return weighted.mean()


def edge_weighted_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Edge-weighted Huber loss (smooth L1). More robust to outliers than MSE.

    Args:
        pred: (B, C, H, W) predicted latents
        target: (B, C, H, W) target latents
        weight_map: (H, W) spatial weight map
        delta: Huber delta threshold
        reduction: 'mean' or 'none'
    """
    abs_error = torch.abs(pred - target)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic**2 + delta * linear

    w = weight_map.unsqueeze(0).unsqueeze(0).to(pred.device)
    weighted = huber * w

    if reduction == "none":
        return weighted
    return weighted.mean()


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

def validate_weight_map(weight_map: torch.Tensor, expected_center: float = 1.0, expected_edge: float = 2.0):
    """Sanity-check a generated weight map."""
    h, w = weight_map.shape
    cy, cx = h // 2, w // 2

    center_val = weight_map[cy, cx].item()
    corner_vals = [
        weight_map[0, 0].item(),
        weight_map[0, w - 1].item(),
        weight_map[h - 1, 0].item(),
        weight_map[h - 1, w - 1].item(),
    ]
    mean_corner = sum(corner_vals) / len(corner_vals)

    print(f"Center weight: {center_val:.4f} (expected ≈ {expected_center})")
    print(f"Corner weights: {corner_vals}")
    print(f"Mean corner weight: {mean_corner:.4f} (expected ≈ {expected_edge})")
    print(f"Corner/Center ratio: {mean_corner / center_val:.4f}")

    assert abs(center_val - expected_center) < 0.01, f"Center {center_val} ≠ {expected_center}"
    assert mean_corner > center_val, "Edge should be weighted higher than center"
    return True
