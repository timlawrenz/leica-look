# LoRA Transfer — Phase 2a Feasibility

**Hypothesis:** A FLUX.1-dev LoRA trained on 270 Leica images can measurably shift
non-Leica images toward the Leica rendering distribution without destroying content.

**Differs from:** Phase 1 (discriminative → generative). First experiment to attempt
actual rendering transfer rather than classification.

**Expected outcome:** Modest but measurable shift in DINOv2 embedding space, with
increased edge attention (lower C/E ratio). CLIP-I preservation ≥ 0.85. If the
transfer is invisible to both metrics, the discriminative signal may not translate
to a generative one — a negative result that would redirect the project.

## Design

- **Model:** FLUX.1-dev fp16 (cached at `/mnt/nas-ai-models/huggingface-cache/`)
- **Adapter:** LoRA rank=32, α=16, targeting all attention layers
- **Training:** 270 Leica images × 1500 steps, 1024×1024, lr=5e-5, cosine schedule
- **Signal isolation:** Edge-weighted loss (radial weight map from attention-bin
  profile: bins 6-9 amplified 2× relative to bins 0-5, crossover at bin 6 from #12)
- **Trigger token:** `lctx` (Leica context) — model must learn rendering independently
  of content captions
- **Hardware:** RTX 4090 (24GB) via GPU scheduler

## Evaluation (on 265 matched non-Leica images)

1. **DINOv2 embedding shift:** cosine distance to Leica reference distribution
   (pre-transfer vs post-transfer)
2. **Attention-map C/E ratio:** does the transfer increase edge attention?
3. **CLIP-I:** content preservation (must stay ≥ 0.85)
4. **Manual visual inspection:** 10 side-by-side triples (input → transfer → Leica ref)

## Runs

| Timestamp | Steps | DINOv2 Δ | C/E Δ | CLIP-I | Status | Notes |
|-----------|-------|----------|-------|--------|--------|-------|
| — | — | — | — | — | PENDING | Awaiting human design review |
