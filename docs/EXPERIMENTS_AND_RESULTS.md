# Experiments & Results

This ledger documents all empirical findings from the leica-look project. Negative results are recorded permanently to prevent re-running dead ends. Every entry includes a pre-registered gate stated BEFORE results.

---

## Phase 2a: LoRA Transfer Feasibility — `[CONCLUDED — FAIL]`

**Date:** 2026-08-08 (training), 2026-08-09 (evaluation)
**Goal:** Determine whether a FLUX.1-dev LoRA trained on 270 Leica images with edge-weighted loss can measurably shift non-Leica images toward the Leica rendering distribution in DINOv2 embedding space.

**Pre-registered gate (stated BEFORE results):**
> PASS if: (1) DINOv2 embedding cosine distance to Leica centroid decreases by ≥0.02, AND (2) attention C/E ratio decreases by ≥0.10 (more edge attention), AND (3) CLIP-I ≥ 0.85.
> PENDING if 2/3 met. FAIL if ≤ 1 met. 

### Empirical Evidence

- **Run:** `runs/2026-08-08_165323`, 1500 steps, batch=1, grad_accum=4, lr=5e-5, rank=32, 1024²
- **Eval:** 20 held-out non-Leica images at step_01500

| Gate | Threshold | Actual | Pass? |
|------|-----------|--------|-------|
| DINOv2 embedding shift | Δ ≥ 0.02 | -0.00075 | **FAIL** |
| Attention C/E ratio | Δ ≤ -0.10 | +0.01422 | **FAIL** |
| CLIP-I preservation | ≥ 0.85 | 0.8977 | **PASS** |

- **Held-out generalization:** Δ = -0.0015 (MEMORIZATION_RISK — no generalization to held-out Leica images)
- **Color baseline:** Color transfer Δ = -0.0007; LoRA Δ = -0.0008. LoRA/Color ratio: -0.75× (indistinguishable from color-only transfer)
- **DINOv2 pre-mean distance:** 0.8844 → post: 0.8851 (no movement)
- **Attention C/E:** 1.41 → 1.42 (attention shifted slightly toward center, not edges)
- **CLIP-I:** mean 0.90, min 0.75, 80% above threshold (content preserved)

### Adversarial Pass

- [x] Gate #3 metric (CLIP-I) verified working: 20 image pairs, plausible values 0.75–0.97
- [x] DINOv2 g embeddings verified: distances in expected range (0.73–0.98)
- [ ] Metric code has unit tests — NO (known project gap)
- [ ] Result reproduced (2nd seed) — NO (only one training run)
- [x] Edge cases inspected — per-image deltas show random scatter, no systematic shift. The positive Δ fraction (45%) is indistinguishable from coin flips.

**Verdict: FAIL.** The LoRA learned content preservation but produced zero measurable shift toward the Leica rendering distribution. The effect is indistinguishable from color-only transfer. This suggests either insufficient training signal (batch=1, 1500 steps), ineffective edge-weighted loss, or LoRA-on-FLUX is fundamentally insufficient for lens-rendering transfer.

### Artifacts
- Config: `experiments/lora-transfer/config.yaml`
- Provenance: `experiments/lora-transfer/provenance.yaml`
- Checkpoint: `experiments/lora-transfer/runs/2026-08-08_165323/checkpoints/final/`
- Evaluation: `experiments/lora-transfer/runs/2026-08-08_165323/evaluation/step_01500/`
- Verdict JSON: `experiments/lora-transfer/runs/2026-08-08_165323/checkpoints/gate_verdict.json`
- Issue: [#17](https://github.com/timlawrenz/leica-look/issues/17)

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

---

## Issue #12: Attention-Map Analysis — `[CONCLUDED — GO]`

**Date:** 2026-08-07
**Goal:** Use DINOv2 attention maps to determine WHERE in the image the Leica/non-Leica classification signal lives. Center-weighted attention = content confound; edge-weighted = lens rendering (bokeh, vignetting, focus falloff).

**Pre-registered gate (stated BEFORE results):**
> PASS if correctly classified images show EDGE/CORNER-weighted attention (center/edge ratio < 1.5) while incorrectly classified images show CENTER-weighted attention (center/edge ratio > 2.0), indicating the model relies on lens-rendering features at edges for correct decisions. PENDING if the pattern is present but effect size is small or variance is large. FAIL if no difference between correct and incorrect attention profiles.

### Empirical Evidence

**Data:** 592 verified images (265 Leica + 327 non-Leica). DINOv2-S/14 cls_patch embeddings, 5-fold CV LR (C=1.0), full CV AUC = 0.906, accuracy = 83.3%. 100 images selected for attention analysis (50 most-confident correct + 50 most-confident incorrect).

**Key metrics (DINOv2-S/14, last-layer CLS-to-patch attention):**

| Group | Mean Center | Mean Edge | Center/Edge Ratio | Std Ratio |
|-------|-------------|-----------|-------------------|-----------|
| Correct (n=50) | 0.00644 | 0.00321 | **2.79** | ±3.48 |
| Incorrect (n=50) | 0.00634 | 0.00261 | **3.51** | ±4.38 |

**Key metrics (DINOv2-g/14 validation, n=10+10):**

| Group | Mean Center | Mean Edge | Center/Edge Ratio | Std Ratio |
|-------|-------------|-----------|-------------------|-----------|
| Correct (n=10) | 0.00242 | 0.00269 | **1.14** | ±1.13 |
| Incorrect (n=10) | 0.00425 | 0.00339 | **1.60** | ±1.47 |

**Radial profile (DINOv2-S, 10 bins center→edge):** Inner bins (0–5) show higher attention for INCORRECT predictions. Outer bins (6–9) show higher attention for CORRECT predictions. Crossover at bin 6 (~40% from center to corner). Correct classifications use more peripheral features.

**Scene type breakdown:** Street photography shows no center/edge distinction (ratio ~1.25 for both correct and incorrect). Other scene types show the expected pattern (correct < incorrect). Scene tagging is sparse (67/100 = "other").

### Adversarial Pass

- [x] Metric unchanged across comparisons — same attention extraction for all images
- [x] Result reproduced (2nd model) — DINOv2-g validates the center/edge ratio pattern (correct < incorrect)
- [x] Extremes + edge cases inspected — per-image profiles examined; high-variance outliers confirmed
- [x] Null baseline checked — uniform attention = 0.00391; observed center = 1.64× uniform, edge = 0.75× uniform
- [ ] Unit tests for attention extraction — NOT YET. The extract_attention.py script is single-use analysis code.

**Adversarial pass: 4/5 complete.** The one missing check (unit tests for attention extraction code) is low-risk since the raw attention tensors come directly from the model with no intermediate computation.

### Verdict

**GO** — moderate support. Both DINOv2-S and DINOv2-g show a consistent pattern: correct classifications use more edge/peripheral features, incorrect classifications rely more on the image center. This supports the hypothesis that lens rendering characteristics (bokeh, vignetting, edge sharpness) contribute to the DINOv2 classification signal. However, the effect size is modest (Δ ratio = 0.72 for DINOv2-S) and per-image variance is large (±3.5), so this evidence is suggestive rather than definitive.

### Artifacts
- Full results: `experiments/discriminator-attention/attention_analysis.json`
- Analysis report: `experiments/discriminator-attention/analysis.md`
- Script: `experiments/discriminator-attention/src/extract_attention.py`
- Config: `experiments/discriminator-attention/config.yaml`
- Provenance: `experiments/discriminator-attention/provenance.yaml`
