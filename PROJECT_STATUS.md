# Project Status — leica-look

**Last updated:** 2026-08-07
**Phase / status:** Phase 2a — PENDING HUMAN REVIEW

## Current state

All Phase 1/1.5 tickets (#1–#12) are closed. The evidence supports proceeding to Phase 2:
- DINOv2 discriminators achieve AUC 0.86–0.93 on Leica vs non-Leica (signal present)
- Content confound partially resolved: CLIP gap narrows 24–34% on matched pairs
- Attention-map analysis (GO): correct classifications use more edge features (C/E ratio 2.79 vs 3.51 for incorrect)
- Seed sweep confirms n=250/class is stable (σ ≤ 0.02); n=50 are lucky splits

Phase 2a governance infrastructure is in place at `experiments/lora-transfer/`:
- Provenance with pre-registered gate
- Config with edge-weighted loss (radial profile from attention-map data)
- README with evaluation strategy
- Registered in tree and ledger

**Training is BLOCKED on human design review of issue #14.**

## Headline result so far

| Metric | Value | Model | Verdict |
|---|---|---|---|
| Best overall AUC | 0.9867 | CLIP ViT-L/14 (LR, multicrop, n=50) | Content confound |
| Best DINOv2 AUC | 0.9344 | DINOv2 ViT-g (k-NN, patch_mean, n=100) | Signal present |
| CLIP vs DINOv2 gap | +5.5 pts mean LR AUC | — | Confirms content confound |
| MLP vs LR delta | ≤0 for 6/7 models | — | Signal is linearly separable |
| **Attention edge signal** | **C/E ratio 2.79 correct vs 3.51 incorrect** | **DINOv2-S/14** | **GO — lens signal at edges** |
| Matched-pair DINOv2-g | 0.925–0.942 AUC | DINOv2-g (matched pairs) | Signal survives content-matching |

## Immediate blockers / next action

1. **⛔ BLOCKED: Issue #14 design review.** Phase 2a LoRA transfer experiment needs human go/no-go decision.
   - Review the pre-registered gate thresholds
   - Approve or modify the edge-weighted loss approach
   - Decide whether to proceed with training
2. **⛔ BLOCKED: Issue #13 (controlled capture).** Requires physical camera — skip until human unblocks.

**Do NOT start training until the human reviews and approves the Phase 2a design.**

## Governance docs

- Experiment tree: `docs/EXPERIMENT_TREE.md`
- Permanent ledger: `docs/EXPERIMENTS_AND_RESULTS.md`
- Full results: `docs/discriminator-results.md`
- Experiment design: `docs/experiment-design.md`
- Research plan: `docs/research-plan.md`
- Phase 2a arm: `experiments/lora-transfer/`
