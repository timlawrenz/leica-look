# Attention-Map Analysis

**Hypothesis:** DINOv2 attention maps can spatially localize the Leica/non-Leica classification signal. Center-weighted = content confound; edge-weighted = lens rendering (bokeh, vignetting).

**Differs from baseline by:** This is not a training experiment — it's a diagnostic analysis of the frozen DINOv2 model's attention patterns comparing correctly vs incorrectly classified images.

**Expected outcome:** Edge-weighted attention for correct classifications, center-weighted for incorrect. This would confirm the lens signal exists and is spatially localizable.

## Results

**Verdict: GO** — moderate support. Correct classifications show lower center/edge ratio (2.79) than incorrect (3.51). DINOv2-g validation replicates the pattern (1.14 vs 1.60). Effect is real but modest with large per-image variance.

| Metric | Correct | Incorrect |
|--------|---------|-----------|
| Center/Edge Ratio (S) | 2.79 | 3.51 |
| Center/Edge Ratio (g) | 1.14 | 1.60 |
| Full CV AUC | — | 0.906 |

## Runs

| Timestamp | Images | Model | Result | Notes |
|-----------|--------|-------|--------|-------|
| 2026-08-07_2145 | 100 | DINOv2-S + DINOv2-g | GO | Edge signal detected |

## Artifacts
- `attention_analysis.json` — Full results with per-image attention maps
- `analysis.md` — Comprehensive analysis report
- `src/extract_attention.py` — Extraction script
