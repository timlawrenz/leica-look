# LoRA Transfer — Phase 2a Feasibility

**Hypothesis:** A FLUX.1-dev LoRA trained on 230 Leica images (35 held-out) can
measurably shift non-Leica images toward the Leica rendering distribution without
destroying content.

**Differs from:** Phase 1 (discriminative → generative). First experiment to attempt
actual rendering transfer rather than classification.

**Expected outcome:** Modest but measurable shift in DINOv2 embedding space, with
increased edge attention (lower C/E ratio). CLIP-I preservation ≥ 0.85. If the
transfer is invisible to both metrics, the discriminative signal may not translate
to a generative one — a negative result that would redirect the project.

## Design

- **Model:** FLUX.1-dev fp16 (cached at `/mnt/nas-ai-models/huggingface-cache/`)
- **Adapter:** LoRA rank=32, α=16, targeting all attention layers
- **Training:** 230 Leica images × 1500 steps, 1024×1024, lr=5e-5, cosine schedule
- **Signal isolation:** Edge-weighted loss (radial weight map from attention-bin
  profile: bins 6-9 amplified 2× relative to bins 0-5, crossover at bin 6 from #12)
- **Caption (FIX #4):** Single trigger token `lctx photo` — standard Dreambooth/LoRA
  approach for rendering transfer. Per-image captions are NOT needed because
  composition control is not the goal; the LoRA only needs to learn the rendering
  look, not to follow content prompts.
- **Held-out split (FIX #2):** 35 Leica images (~13.2%) reserved in
  `held_out_split.json`. Not used in training. The gate must confirm shift toward
  the held-out centroid (generalization), not just the training-set centroid
  (memorization risk).
- **Color baseline (FIX #1):** `src/color_transfer_baseline.py` applies Reinhard
  LAB color transfer as a control. If LoRA Δ ≈ color baseline Δ, the "Leica look"
  is primarily color science, not optics.
- **Dry-run (FIX #3):** `--dry-run` validates: pack/unpack roundtrip, prompt
  consistency, add_noise path, edge-weighted loss, CFG interface.
- **Hardware:** RTX 4090 (24GB) via GPU scheduler

## Evaluation (on 327 non-Leica images)

1. **DINOv2 embedding shift:** cosine distance to Leica reference distribution
   (pre-transfer vs post-transfer) — both full-centroid AND held-out centroid
2. **Attention-map C/E ratio:** does the transfer increase edge attention?
3. **CLIP-I:** content preservation (must stay ≥ 0.85)
4. **Color baseline comparison:** LoRA Δ vs color-transfer Δ (optical vs color signal)
5. **Manual visual inspection:** 10 side-by-side triples (input → transfer → Leica ref)

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | Canonical experiment configuration |
| `provenance.yaml` | Pre-registered gate, hypothesis, fix log |
| `held_out_split.json` | 230 train / 35 held-out split (seed=42) |
| `src/train_lora.py` | FLUX LoRA training script (with dry-run mode) |
| `src/evaluation.py` | Gate metrics (embedding shift, C/E ratio, CLIP-I) |
| `src/evaluate_transfer.py` | Post-training gate evaluation |
| `src/edge_weighted_loss.py` | Spatial weight map + weighted loss functions |
| `src/held_out_split.py` | Train/holdout split creation utility |
| `src/color_transfer_baseline.py` | Reinhard LAB color-transfer control arm |

## Runs

| Timestamp | Steps | DINOv2 Δ | C/E Δ | CLIP-I | HO Δ | Color vs LoRA | Status | Notes |
|-----------|-------|----------|-------|--------|-------|---------------|--------|-------|
| — | — | — | — | — | — | — | PENDING | Awaiting human GO after hardening fixes (#15) |
| 2026-08-08_090324 | dry-run | — | — | — | — | — | PARTIAL PASS | Dry-run: logic ✓, OOM at 1024² (4090 VRAM limit) |
| 2026-08-08_091121 | color-baseline | -0.0007 | — | — | — | — | COMPLETE | Reinhard LAB: no shift. Color alone doesn't explain look. |
