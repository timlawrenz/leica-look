# Phase 1 Discriminator Results

**Date:** 2026-08-07
**Status:** Complete — all 1,008 evaluations run across 3 classifiers

## Executive Summary

We ran a comprehensive 7-model × 6-pooling × 3-classifier ablation study to determine whether the Leica lens rendering look is a learnable, separable signal in frozen vision model embeddings. The answer is nuanced: **the signal exists but is confounded with semantic content.**

- **CLIP ViT-L/14** dominates everything (AUC 0.987), but this is a **content confound** — CLIP's text-aligned training lets it detect *what Leica photographers shoot*, not *how Leica lenses render*.
- **DINOv2 family** achieves AUC 0.86–0.93, confirming a real visual signal exists independent of text alignment.
- **SigLIP** is unstable (0.938 LR → 0.887 k-NN), suggesting its text-aligned features interact differently with the classification decision boundary.
- **MLP does not beat LR** for 6/7 models — the Leica/non-Leica signal in these embeddings is **linearly separable**, a simple shift rather than a nonlinear texture composition.
- **k=1 is noisy** (mean AUC 0.777 vs. k=5 at 0.863) — the signal requires local averaging.

---

## 1. AUC Matrix

### Best AUC per Model × Classifier

| Model | Best LR | Best MLP | Best k-NN | Best Overall | Best Classifier | Best Pooling | n |
|---|---|---|---|---|---|---|---|
| **CLIP ViT-L/14** | **0.9867** | 0.9755 | 0.9562 | **0.9867** | LR | multicrop | 50 |
| SigLIP SO400M | 0.9378 | 0.8889 | 0.8872 | 0.9378 | LR | cls_patch | 50 |
| DINOv2 ViT-g/14 | 0.9156 | 0.9099 | **0.9344** | 0.9344 | k-NN | patch_mean | 100 |
| DINOv2 ViT-S/14 | 0.9289 | 0.9289 | 0.9333 | 0.9333 | k-NN | patch_gem | 100 |
| DINOv3 ViT-L/16 | 0.9156 | 0.9289 | 0.9311 | 0.9311 | k-NN | cls_patch | 100 |
| DINOv2 ViT-B/14 | 0.9138 | 0.9107 | 0.9113 | 0.9138 | LR | patch_gem | 250 |
| DINOv2 ViT-L/14 | 0.9067 | 0.9026 | 0.9122 | 0.9122 | k-NN | patch_mean | 100 |

### Full Grid Statistics

- **LR:** 504 evaluations (7 models × 6 pooling × 3 sizes × 4 C values) — mean AUC 0.865
- **MLP:** 126 evaluations (7 × 6 × 3) — mean AUC 0.858
- **k-NN:** 378 evaluations (7 × 6 × 3 × 3 k values) — mean AUC 0.832

### CLIP vs. DINO Analysis

| Metric | CLIP ViT-L/14 | DINOv2 mean (S/B/L/g) | Δ |
|---|---|---|---|
| Best LR AUC | 0.9867 | 0.9162 | +0.0705 |
| Mean LR AUC | 0.9181 | 0.8631 | +0.0550 |
| Best k-NN AUC | 0.9562 | 0.9228 | +0.0334 |
| Best MLP AUC | 0.9755 | 0.9131 | +0.0624 |

**Top 10 LR results are ALL CLIP.** This is the experiment design's pre-registered negative result: CLIP's dominance indicates the classifier is cheating on semantic content, not lens rendering.

---

## 2. Key Comparisons

### DINOv2 vs. DINOv3 — Does newer pretraining help?

| Model | Best LR | Best k-NN | Best MLP |
|---|---|---|---|
| DINOv2 ViT-L/14 | 0.9067 | 0.9122 | 0.9026 |
| DINOv3 ViT-L/16 | 0.9156 | 0.9311 | 0.9289 |

DINOv3 slightly edges out DINOv2-L across all classifiers (+0.9–2.5 pts). The improvement is modest but consistent — suggesting newer pretraining captures marginally more rendering-relevant features.

### DINOv2 vs. SigLIP — Is text-alignment confounded?

| Model | Best LR | Best k-NN | Best MLP | Stability (LR→kNN drop) |
|---|---|---|---|---|
| DINOv2-g | 0.9156 | 0.9344 | 0.9099 | +0.0188 (gain!) |
| SigLIP | 0.9378 | 0.8872 | 0.8889 | −0.0506 (drop) |

SigLIP's best LR (0.938) comes from cls_patch (2,304-dim concatenation) at n=50 — a high-dimensional, small-sample regime prone to overfitting. It drops hard with k-NN and MLP. DINOv2 is stable across classifiers. This suggests SigLIP's LR win is fragile, possibly exploiting the text-aligned subspace that collapses without linear decision boundaries.

### S → B → L → g — Scaling law?

| Model | Params | Best AUC |
|---|---|---|
| DINOv2 ViT-S | 21M | 0.9333 |
| DINOv2 ViT-B | 86M | 0.9138 |
| DINOv2 ViT-L | 304M | 0.9122 |
| DINOv2 ViT-g | 1.1B | 0.9344 |

**No monotonic scaling.** ViT-S (smallest) outperforms ViT-B and ViT-L. ViT-g recovers but doesn't dramatically exceed ViT-S. This is counterintuitive — the smallest model has the most discriminative features. Possible explanations: (1) smaller models compress more relevant information into fewer dimensions (dimensionality as implicit regularization), or (2) the signal is simple enough that 384 dimensions suffice, and extra capacity captures noise.

### CLS vs. Patch Pooling — Global or local signal?

| Pooling | Mean LR AUC | Max LR AUC |
|---|---|---|
| cls_patch (concat) | 0.8852 | 0.9721 |
| patch_gem | 0.8708 | 0.9429 |
| multicrop | 0.8671 | 0.9867 |
| patch_mean | 0.8663 | 0.9545 |
| patch_max | 0.8512 | 0.9292 |
| cls | 0.8490 | 0.9778 |

**cls_patch (concatenated CLS + patch mean) wins on average**, suggesting the signal spans both global and local features. Multicrop (test-time augmentation with 5 random crops) produces the single best result (CLIP 0.987) but is computationally expensive. Patch max underperforms — the "strongest local texture" hypothesis doesn't hold.

### Linear vs. MLP — Separability structure?

| Model | Δ (MLP − LR) |
|---|---|
| DINOv3 ViT-L/16 | +0.0133 |
| DINOv2 ViT-S/14 | ±0.0000 |
| DINOv2 ViT-B/14 | −0.0030 |
| DINOv2 ViT-L/14 | −0.0041 |
| DINOv2 ViT-g/14 | −0.0057 |
| CLIP ViT-L/14 | −0.0112 |
| SigLIP SO400M | −0.0489 |

**The signal is linearly separable.** For 6/7 models, MLP ≤ LR. The 2-layer MLP (256→64, dropout 0.3) provides zero benefit over a linear decision boundary. This strongly suggests the Leica/non-Leica distinction in embedding space is a simple shift (e.g., a dominant principal component like global color cast or contrast), not a nonlinear texture composition.

### 50 vs. 100 vs. 250 images — Dataset size scaling

| n per class | Mean LR AUC | Max LR AUC | n configs |
|---|---|---|---|
| 50 | 0.8530 | 0.9867 | 168 |
| 100 | 0.8572 | 0.9278 | 168 |
| 250 | 0.8847 | 0.9721 | 168 |

**More data helps on average** (+3.2 pts from 50→250), but the best individual results come from n=50 (CLIP at 0.987). The pattern suggests that small-sample regimes can produce spuriously high AUC when the model overfits to a few easy-to-separate images — this is the "lucky split" problem.

### k-NN: k-value comparison

| k | Mean AUC | Max AUC |
|---|---|---|
| 1 | 0.7769 | 0.8994 |
| 5 | 0.8627 | 0.9562 |
| 11 | 0.8577 | 0.9504 |

**k=1 is too noisy** — single nearest-neighbor matching has high variance. k=5 is the sweet spot, with k=11 slightly behind. The embedding space has local structure (k-NN works) but individual point estimates are unreliable — exactly what you'd expect from noisy, uncurated Flickr images.

---

## 3. Adversarial Pass

| Check | Status | Detail |
|---|---|---|
| Metric code has unit tests | ❌ | Zero test files in repo. All three scripts use `sklearn.metrics.roc_auc_score` directly without test harness. |
| Metric definition unchanged | ✅ | All scripts use identical `roc_auc_score(y_test, y_prob)` — metric version is consistent. |
| Winner reproduced (different seed) | ❌ | All runs use deterministic seed=42. No seed sweep or reproduction run performed. |
| Extremes + edge cases inspected | ❌ | No per-image predictions saved. No manual inspection of top/bottom predictions. |

**Adversarial pass: INCOMPLETE** — 3 of 4 checks are not met. The verdict must be qualified.

---

## 4. Known Confounds

### Content Confound (CLIP dominance)

CLIP ViT-L/14 dominates all classifiers by +5-7 pts AUC over the best DINOv2 model. CLIP is trained with language supervision — its features encode *what objects are in the image*, not *how the lens renders*. DINOv2 (self-supervised, no text) encodes mostly visual texture. The CLIP-DINO gap directly measures the content confound: Leica photographers shoot different subject matter.

### Body/Brand Confound (inherent)

By definition, all positive-class images come from Leica bodies and all negative-class images come from Canon/Sony/Nikon bodies. The stratification by body model within the positive class doesn't eliminate the brand confound — the class IS the brand. A classifier could be learning:
- Sensor color science differences (Leica CCD/CMOS vs. Canon/Sony)
- JPEG engine differences (Leica's in-camera processing)
- Lens mount metadata in EXIF leaked into JPEG headers

None of these are lens rendering, but all are detectable signals.

### Variance Confound (small test sets)

At n=50, the test set has only 30 images (15 per class, 30% test split). An AUC of 0.987 from 30 images has a wide confidence interval. The top CLIP result at n=50 (multicrop, C=10.0, AUC 0.987) may not generalize. CLIP at n=250 (cls_patch, C=1.0, AUC 0.971) is more credible but still confounded.

---

## 5. Verdict

### By the pre-registered criteria:

| AUC range | Interpretation | Applies to |
|---|---|---|
| > 0.90 | Strong signal — proceed to LoRA | CLIP (0.987), DINOv2-g (0.934), DINOv2-S (0.933), SigLIP (0.938) |
| 0.80–0.90 | Real signal — data engine viable | DINOv2-B (0.914), DINOv2-L (0.912) |

### Actual verdict: **PIVOT**

The raw numbers clear the GO threshold (>0.90) for several models, but the adversarial pass is incomplete and the content confound is severe. **Proceeding directly to LoRA training would learn to distinguish Leica photographers' subject choices, not Leica lens rendering.** The signal in DINOv2 (AUC 0.86–0.93) is real but unvalidated — we don't know if it's lens rendering, body sensor characteristics, or residual content.

### Recommended pivot:
1. **Content-matching experiment:** Pair-match Leica and non-Leica images by scene type, composition, and subject matter. Re-run probes on matched pairs.
2. **Controlled capture:** Photograph the same scene with Leica + non-Leica lenses on the same body (adapter mount) to isolate lens rendering from sensor/body.
3. **Lens-only signal extraction:** Use DINOv2 attention maps to identify which image regions drive the classification — if it's center-weighted (subject), it's content; if it's edge/corner-weighted (bokeh, vignetting), it's lens.
4. **Seed sweep:** Re-run the top-10 configurations with 5 different random seeds to measure AUC variance and confirm the results aren't lucky splits.

---

## 6. Artifacts

- **LR results:** `experiments/discriminator-lr/results.csv` (504 rows)
- **MLP results:** `experiments/discriminator-mlp/results.csv` (126 rows)
- **k-NN results:** `experiments/discriminator-knn/results.csv` (378 rows)
- **Analysis script:** `scripts/compile_phase1_results.py`
- **Probe scripts:** `src/run_logistic_probes.py`, `src/run_mlp_probes.py`, `src/run_knn_probes.py`
