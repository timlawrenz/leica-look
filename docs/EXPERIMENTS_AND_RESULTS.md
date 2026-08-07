# Experiments & Results

This ledger documents all empirical findings from the leica-look project. Negative results are recorded permanently to prevent re-running dead ends. Every entry includes a pre-registered gate stated BEFORE results.

---

## Phase 1: Discriminator Ablation Study — `[CONCLUDED — PIVOT]`

**Date:** 2026-08-07
**Goal:** Determine whether the Leica lens rendering look is a learnable, separable signal in frozen vision model embeddings. Run 7 models × 6 pooling × 3 classifiers × 3 sizes (n=50/100/250) = 1,008 total evaluations.

**Pre-registered gate (from docs/experiment-design.md):**
- AUC < 0.65 → KILL (signal too weak)
- 0.65–0.80 → PIVOT (needs more data/better features)
- 0.80–0.90 → GO (data engine viable)
- > 0.90 → GO (proceed to Phase 2 immediately)
- **Key negative result:** if CLIP performs significantly better than DINOv2, the classifier is likely cheating on content/semantics.

### Empirical Evidence

**Data:** 270 EXIF-verified Leica images (positive), 334 EXIF-verified non-Leica premium images (negative). Both classes have zero flickr_id overlap. Split: 70/30 train/test, stratified by scene type + body manufacturer groups.

**Key metrics:**

| Model | Best LR AUC | Best MLP AUC | Best k-NN AUC | Best Overall |
|---|---|---|---|---|
| CLIP ViT-L/14 | **0.9867** | 0.9755 | 0.9562 | **0.9867 (LR, multicrop, n=50)** |
| SigLIP SO400M | 0.9378 | 0.8889 | 0.8872 | 0.9378 (LR, cls_patch, n=50) |
| DINOv2 ViT-g/14 | 0.9156 | 0.9099 | 0.9344 | 0.9344 (k-NN, patch_mean, n=100) |
| DINOv2 ViT-S/14 | 0.9289 | 0.9289 | 0.9333 | 0.9333 (k-NN, patch_gem, n=100) |
| DINOv3 ViT-L/16 | 0.9156 | 0.9289 | 0.9311 | 0.9311 (k-NN, cls_patch, n=100) |
| DINOv2 ViT-B/14 | 0.9138 | 0.9107 | 0.9113 | 0.9138 (LR, patch_gem, n=250) |
| DINOv2 ViT-L/14 | 0.9067 | 0.9026 | 0.9122 | 0.9122 (k-NN, patch_mean, n=100) |

**Critical findings:**
- CLIP dominates DINOv2 by +5.5 pts mean LR AUC (0.918 vs. 0.863). Top 10 LR results are ALL CLIP.
- MLP does NOT beat LR for 6/7 models — signal is linearly separable.
- k=1 is noisy (mean AUC 0.777) vs k=5 (0.863) — embeddings have local structure but high variance.
- No monotonic model scaling: ViT-S (21M) outperforms ViT-B (86M) and ViT-L (304M).
- Dataset size scaling is modest: mean LR AUC 0.853→0.885 from n=50→250. The best CLIP result comes at n=50 (overfit concern).

### Adversarial Pass

- [ ] Metric code has unit tests — **NO.** Zero test files in repo. All scripts use sklearn's roc_auc_score directly.
- [x] Metric definition unchanged across comparisons — Yes. All use identical roc_auc_score(y_test, y_prob).
- [ ] Winner reproduced with different seed — **NO.** All runs use seed=42. CLIP's 0.987 at n=50 untested with seed sweep.
- [ ] Extremes + edge cases inspected — **NO.** No per-image predictions saved.

**Adversarial pass: 1/4 complete.** Verdict is qualified — the raw numbers pass the GO threshold but the measurement infrastructure is untested and unreproduced.

### Verdict

**PIVOT** — not GO because CLIP's dominance (+5.5 pts over DINOv2) confirms the pre-registered content-confound hypothesis. The signal in DINOv2 (0.86–0.93) confirms a real visual signal exists, but we cannot attribute it to lens rendering vs. body sensor characteristics vs. residual content. Proceeding to LoRA training now would learn to distinguish Leica photographers' subject choices, not Leica lens rendering.

**Pivot direction:** Content-matching experiment (pair-match by scene type), controlled capture (same scene, Leica + non-Leica lenses on same body), lens-only signal extraction via attention maps, and seed sweep on top configurations.

### Artifacts
- LR results: `experiments/discriminator-lr/results.csv` (504 rows)
- MLP results: `experiments/discriminator-mlp/results.csv` (126 rows)
- k-NN results: `experiments/discriminator-knn/results.csv` (378 rows)
- Full analysis: `docs/discriminator-results.md`
- Analysis script: `scripts/compile_phase1_results.py`

---

## Issue #1: Download missing vision models — `[CONCLUDED — DONE]`

**Date:** 2026-08-07 (before this ledger)
**Goal:** Download missing vision models to NAS cache for embedding extraction.

**Verdict:** DONE. Models cached at `/mnt/nas-ai-models/huggingface-cache/`.

---

## Issue #2: Scrape seed dataset from Flickr — `[CONCLUDED — DONE]`

**Date:** 2026-08-07 (before this ledger)
**Goal:** Scrape Leica and non-Leica images from Flickr with EXIF verification.

**Verdict:** DONE. 270 positive + 334 negative EXIF-verified images. Manifests at `data/registry/`.

---

## Issue #3: Download & verify dataset images — `[CONCLUDED — DONE]`

**Date:** 2026-08-07 (before this ledger)
**Goal:** Download original-size images and verify EXIF lens match, min resolution, no monochrome.

**Verdict:** DONE. Verified.csv written. Images at `/mnt/nas-ai-models/training-data/leica-look/raw/`.

---

## Issue #4: Extract all vision model embeddings — `[CONCLUDED — DONE]`

**Date:** 2026-08-07 (before this ledger)
**Goal:** Extract frozen embeddings (7 models × 6 pooling) from all verified images.

**Verdict:** DONE. Embeddings cached at `/mnt/nas-ai-models/training-data/leica-look/embeddings/`.
