# Project Status — leica-look

**Last updated:** 2026-08-07
**Phase / status:** Phase 1 CONCLUDED — PIVOT

## Current state

Phase 1 discriminator ablation study is complete: 1,008 evaluations across 7 vision models, 6 pooling strategies, 3 classifiers, and 3 dataset sizes. The raw AUC numbers clear the GO threshold (>0.90 for CLIP, DINOv2-g, DINOv2-S, SigLIP), but the adversarial pass is incomplete (1/4 checks) and CLIP's dominance over DINOv2 (+5.5 pts) confirms the pre-registered content-confound hypothesis — the classifier is detecting *what Leica photographers shoot*, not *how Leica lenses render*.

The DINOv2 family (self-supervised, no text alignment) achieves AUC 0.86–0.93, confirming a real visual signal exists, but we cannot attribute it to lens rendering vs. body sensor characteristics vs. residual content.

## Headline result so far

| Metric | Value | Model | Verdict |
|---|---|---|---|
| Best overall AUC | 0.9867 | CLIP ViT-L/14 (LR, multicrop, n=50) | Content confound |
| Best DINOv2 AUC | 0.9344 | DINOv2 ViT-g (k-NN, patch_mean, n=100) | Signal present, unvalidated |
| CLIP vs DINOv2 gap | +5.5 pts mean LR AUC | — | Confirms content confound |
| MLP vs LR delta | ≤0 for 6/7 models | — | Signal is linearly separable |

## Immediate blockers / next action

**PIVOT required before Phase 2.** Three concrete next steps (in priority order):

1. **Seed sweep (fastest):** Re-run top-10 LR configurations with 5 different random seeds to measure AUC variance. CLIP's 0.987 at n=50 (only 30 test images) may be a lucky split. This can run on CPU in ~10 minutes.

2. **Content-matching experiment:** Pair-match Leica and non-Leica images by scene type (portrait vs. portrait, landscape vs. landscape). Re-run probes on matched pairs. If the CLIP-DINO gap closes, the signal shifts from content-confounded to potentially lens-based.

3. **Attention-map analysis:** Identify which image regions drive DINOv2 classification decisions. Center-weighted = content; edge/corner-weighted = lens (bokeh, vignetting).

**Do NOT proceed to Phase 2 (LoRA training) until at least items 1 and 2 are complete.** Training now would learn to distinguish Leica photographers' subject choices, not Leica lens rendering.

## Governance docs

- Experiment tree: `docs/EXPERIMENT_TREE.md`
- Permanent ledger: `docs/EXPERIMENTS_AND_RESULTS.md`
- Full results: `docs/discriminator-results.md`
- Experiment design: `docs/experiment-design.md`
- Research plan: `docs/research-plan.md`
