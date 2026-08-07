# Experiment Design: Discriminator Ablation Study (Phase 1)

## Objective

Determine whether the Leica lens rendering look is a learnable, separable signal in visual embeddings by training classifiers on frozen features from multiple vision models.

## Ablation Matrix

### Models (embedding backbones)

| Model | Dim | Params | Notes |
|---|---|---|---|
| DINOv2 ViT-S/14 | 384 | 21M | Baseline — is the signal there at all? |
| DINOv2 ViT-B/14 | 768 | 86M | Registers enabled |
| DINOv2 ViT-L/14 | 1024 | 304M | Registers enabled — likely sweet spot |
| DINOv2 ViT-g/14 | 1536 | 1.1B | Upper bound for v2 |
| DINOv3 ViT-L/16 | 1024 | ~300M | Newer pretraining, might capture different features |
| SigLIP ViT-L/16 | 1024 | ~300M | Text-aligned — **negative control** (does language bias help or hurt?) |
| CLIP ViT-L/14 | 768 | ~300M | Weaker negative control |

### Feature pooling strategies

For each model, extract:

| Pooling | Dim | What it captures |
|---|---|---|
| **CLS token** | model_dim | Global image-level feature |
| **Patch mean pool** | model_dim | Average texture across image |
| **Patch max pool** | model_dim | Strongest local texture signal (bokeh highlights, edge contrast) |
| **Patch GeM pool** (p=3) | model_dim | Interpolates mean/max — standard for texture retrieval |
| **CLS + patch (concat)** | 2× model_dim | Combined global + local |
| **Multi-crop avg** (5 crops) | model_dim | Test-time augmentation — averages CLS over 5 random crops |

### Classifier architectures

| Classifier | Hyperparams | Purpose |
|---|---|---|
| **Logistic regression** | C ∈ {0.01, 0.1, 1, 10} | Linear separability test — tightest signal check |
| **2-layer MLP** | hidden=[256, 64], dropout=0.3 | Slight nonlinearity — is the signal slightly curved? |
| **k-NN** | k ∈ {1, 5, 11} | Non-parametric baseline — sensitive to embedding quality |

### Dataset size sweep

| Train size per class | Purpose |
|---|---|
| 50 | Floor — is the signal learnable at all? |
| 100 | Practical minimum |
| 250 | Upper bound (capped by verified positive supply ≈ 271) |

> **Note (2026-08-07):** the original `{50, 100, 200, 500}` sweep was narrowed to
> `{50, 100, 250}` because the EXIF-verified positive class tops out at 271, so
> the 500/class cell is unreachable. 250 is comfortably above the 200 "expected
> sweet spot" and the ±SE analysis confirms 271/class gives decisively
> separable AUC at the 0.80 gate.

## Full Experiment Grid

```
7 models × 6 pooling × 3 classifiers × 3 dataset sizes (50/100/250) = 378 combinations
```

Embeddings are cached per model, so 378 evaluations complete in ~30 minutes once embeddings are computed. Embedding extraction: ~10–15 minutes for all models.

## Stratification

Train/test splits must be stratified by:
- **Scene category**: portrait, landscape, street, macro, architecture, night
- **Leica body model**: M10, M11, SL2, Q2, other

This ensures the classifier isn't learning content or body-model confounds.

## Key Comparisons

| Comparison | Question |
|---|---|
| DINOv2 vs. DINOv3 | Does newer pretraining capture more rendering nuance? |
| DINOv2 vs. SigLIP | Is text-alignment confounded with content, or does it help? |
| S → B → L → g | Scaling law for lens-rendering features — does bigger help? |
| CLS vs. patch pooling | Is the signal global (color, contrast) or local (texture, bokeh)? |
| Linear vs. MLP | Is the Leica look linearly separable, or does it need nonlinear composition? |
| 50 vs. 500 images | How many EXIF-verified images do you actually need? |

## Success Criteria

| AUC range | Interpretation |
|---|---|
| < 0.65 | Signal too weak — Leica look not learnable from unpaired images at this scale |
| 0.65–0.80 | Signal present but noisy — needs more data or better features |
| 0.80–0.90 | Real, learnable signal — data engine viable |
| > 0.90 | Strong signal — proceed to LoRA training immediately |

## Key Negative Result

If DINOv2 achieves high AUC but SigLIP/CLIP don't, the signal is **visual-texture-based** (lens rendering) rather than **semantic** (Leica shooters photograph different things). If CLIP performs equally well, the classifier is likely cheating on content and the whole approach needs rethink.

## Seed Dataset Construction

### Positive class (Leica)

**Source:** Flickr API, filtered by EXIF lens model

Target lenses:
- Leica Summilux-M 50mm f/1.4 ASPH
- Leica Summilux-M 35mm f/1.4 ASPH
- Leica APO-Summicron-M 50mm f/2 ASPH
- Leica Summilux-M 28mm f/1.4 ASPH

Target bodies: Leica M10, M10-R, M11, SL2, SL3, Q2, Q3

**Size:** 300–500 images, balanced across lens models, focal lengths, and scene types

### Negative class (non-Leica premium)

**Source:** Flickr API, EXIF-confirmed high-end glass

Target lenses:
- Canon EF/RF L series (50mm f/1.2L, 85mm f/1.2L, 35mm f/1.4L II)
- Sony FE GM (50mm f/1.2 GM, 85mm f/1.4 GM, 35mm f/1.4 GM)
- Zeiss Otus (55mm f/1.4, 85mm f/1.4)
- Nikon Z S-line (50mm f/1.2 S, 85mm f/1.2 S)

**Size:** 500–1000 images

### Quality filters
- Minimum resolution: 1024px short edge
- Download original size (not Flickr-compressed)
- Exclude monochrome (Leica Monochrom has different signal)
- Exclude obvious heavy edits (check histogram for crushed blacks)
- Creative Commons licensed preferred
