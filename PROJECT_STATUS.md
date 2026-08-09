# Project Status — leica-look

**Last updated:** 2026-08-09
**Phase / status:** Phase 2a — CONCLUDED (FAIL)

## Current state

Phase 2a LoRA transfer training completed and evaluated. Verdict: **FAIL** — 1/3 gate criteria met.

The FLUX.1-dev LoRA (rank=32, 1500 steps, edge-weighted loss) preserved content well (CLIP-I 0.90) but produced **no measurable shift** toward the Leica rendering distribution:
- DINOv2 embedding shift: -0.0008 (need ≥ 0.02) — effectively zero
- Attention C/E ratio: +0.014 (need ≤ -0.10) — wrong direction
- CLIP-I preservation: 0.8977 ✓
- Effect indistinguishable from color-only baseline (both Δ ≈ 0)
- Held-out generalization: MEMORIZATION_RISK

Full evaluation at `gate_verdict.json`. Issue #17 updated.

## Headline result so far

| Metric | Value | Model | Verdict |
|---|---|---|---|
| Best overall AUC | 0.9867 | CLIP ViT-L/14 (LR, multicrop, n=50) | Content confound |
| Best DINOv2 AUC | 0.9344 | DINOv2 ViT-g (k-NN, patch_mean, n=100) | Signal present |
| CLIP vs DINOv2 gap | +5.5 pts mean LR AUC | — | Confirms content confound |
| MLP vs LR delta | ≤0 for 6/7 models | — | Signal is linearly separable |
| **Attention edge signal** | **C/E ratio 2.79 correct vs 3.51 incorrect** | **DINOv2-S/14** | **GO — lens signal at edges** |
| Matched-pair DINOv2-g | 0.925–0.942 AUC | DINOv2-g (matched pairs) | Signal survives content-matching |
| **Phase 2a LoRA transfer** | **0/2 rendering gates passed** | **FLUX.1-dev LoRA** | **FAIL — no lens transfer effect** |

## Immediate next action

**Human decision required: PIVOT or KILL Phase 2.**

Options to consider:
1. **PIVOT — Longer training:** 1500 steps at batch=1 may be insufficient. Try 5000+ steps, larger batch, higher rank.
2. **PIVOT — Different architecture:** LoRA on FLUX may be fundamentally wrong for lens rendering. Consider ControlNet, IP-Adapter, or direct image-to-image translation.
3. **PIVOT — Better loss signal:** Edge-weighted loss didn't work. Try perceptual loss (LPIPS), discriminator-guided loss, or feature-matching loss against Leica DINOv2 embeddings.
4. **KILL — Insufficient signal:** The DINOv2 AUC of 0.93 on matched pairs may not be strong enough to guide generation. If the discriminator can't reliably tell Leica from non-Leica on content-matched images, a generator trained to fool it won't learn rendering-specific features.

## Baseline calibration (2026-08-07)

Gate threshold calibration completed (CPU-only, no GPU needed):
- DINOv2-g CLS centroid gap: 0.131 (Leica → centroid: 0.763, NonLeica → centroid: 0.895)
- Δ ≥ 0.02 = 15.2% of gap — aggressive but feasible for 1500-step experiment
- Only 6.7% of non-Leica images naturally fall within Δ of Leica centroid
- Effect size: 1.01 (gap / Leica σ) — moderate-to-strong
- Results: `experiments/lora-transfer/baseline_cosine_distances.json`

## Governance docs

- Experiment tree: `docs/EXPERIMENT_TREE.md`
- Permanent ledger: `docs/EXPERIMENTS_AND_RESULTS.md`
- Full results: `docs/discriminator-results.md`
- Experiment design: `docs/experiment-design.md`
- Research plan: `docs/research-plan.md`
- Phase 2a arm: `experiments/lora-transfer/`
