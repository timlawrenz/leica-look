# Research Plan: Leica Lens Rendering Transfer

## Background

### What is the "Leica look"?

The Leica look is a constellation of optical characteristics associated with Leica M-mount lenses (particularly ASPH designs):

| Quality | Optical correlate | Measurable? |
|---|---|---|
| Microcontrast | High MTF at mid-high spatial frequencies, clean edge transitions | Yes — MTF curves, LensFun |
| Smooth focus falloff | Gradual spherical aberration, controlled coma | Partially — PSF measurements |
| 3D "pop" | Subject-background separation via contrast + bokeh | Indirect — combination of above |
| Bokeh character | Specific point spread function shape, no onion rings, smooth OOF highlights | Yes — PSF at different apertures |
| Color rendering | Sensor + JPEG engine color science | Yes — color profiles, 3D LUTs |

The optical component is the hard part. Color science is trivially transferable via 3D LUT or RAW profiles. Lens rendering lives in texture, edge behavior, and spatially-varying defocus — higher-order signals.

### Has it been quantified?

**Not comprehensively.** Individual optical characteristics are measured (MTF, distortion, vignetting) and catalogued in the LensFun database. But no one has produced a unified "Leica rendering vector" or demonstrated that it's a learnable signal separable from content.

### Prior art

| Paper | Year | Relevance |
|---|---|---|
| NeuroLens (Zheng & Zheng) | 2017 | Data-driven lens simulation via neural networks — learns ray-transfer function |
| Neural Lens Modeling (Xian et al.) | CVPR 2023 | Invertible neural network for lens distortion + vignetting from LensFun |
| Lens-to-Lens Bokeh Transformation | NTIRE 2023 | Supervised transfer of bokeh between lenses with paired data |
| Bokehlicious | 2025 | Photorealistic bokeh with aperture-aware transformer attention |
| Deep Photo Style Transfer (Luan et al.) | CVPR 2017 | Photorealistic style transfer via locally affine color constraints |
| EBokehNet (Seizinger et al.) | CVPRW 2023 | Multi-lens bokeh rendering and transformation from all-in-focus images |
| Canon Neural Lens Optimizer | 2023 | Deep learning to *remove* lens aberrations — the inverse of our goal |

## Three-Phase Pipeline

### Phase 1: Discriminator (Prove the signal exists)

**Goal:** Determine if the Leica rendering look is a learnable, separable signal in visual embeddings.

**Method:** Train classifiers on frozen embeddings from vision models (DINOv2, DINOv3, SigLIP, CLIP) to distinguish Leica ASPH photos from other high-end lens photos.

**Seed dataset:**
- Positive: 300–500 EXIF-confirmed Leica ASPH photos from Flickr
- Negative: 500–1000 photos from other premium lenses (Canon L, Sony GM, Zeiss Otus, Nikon S-line)
- Stratified by scene type (portrait, landscape, street, macro, architecture, night) and body model

**Success criteria:** AUC > 0.80 on held-out test set

→ See [`experiment-design.md`](experiment-design.md) for full ablation matrix

### Phase 2: Style Transfer (Apply the look)

**Goal:** Transfer Leica rendering to arbitrary photos while preserving content.

**Method:**
1. Train a LoRA on FLUX.1-dev using Leica-rendered images with trigger token
2. Apply via img2img at low denoising strength (0.2–0.4) with 5–15 steps
3. The LoRA biases denoising toward the Leica distribution; low step count prevents hallucination

**Key design decisions:**
- **FLUX.1-dev** (not schnell): full scheduler needed for fine-grained control
- **Trigger token** (`lctx`) + content captions: model must learn rendering, not content
- **LoRA rank 16–32**: style transfer needs more capacity than face LoRAs
- **Learning rate 5e-5 to 1e-4**: FLUX is LR-sensitive

**Training:**
- Kohya sd-scripts on Strix Halo (128GB unified memory)
- 1024×1024 resolution, 1000–2000 steps
- Dataset: 500–2000 Leica photos (initially curated, later expanded via Phase 3)

**Inference pipeline:**
```
Input → VAE encode → add noise (strength 0.2–0.4) → LoRA denoise (5–15 steps) → VAE decode → output
```

### Phase 3: Data Engine (Automate the pipeline)

**Goal:** Use the Phase 1 discriminator to harvest unlabeled Leica-style photos at scale.

**Method:**
1. Crawl Flickr for images from high-end cameras
2. Score each with Phase 1 classifier
3. Top-k (e.g., top 5%) become pseudo-labeled positives
4. Iterate: train stronger classifier on expanded set, re-harvest

**This closes the loop:** discriminator → data engine → LoRA trainset → style transfer model

## Research Questions

1. **Is the Leica look learnable from unpaired data?** If Phase 1 achieves high accuracy, the signal exists independently of content.
2. **What's the effective dimensionality?** LoRA rank ablation tells us whether the look is mostly a color/LUT shift (low rank sufficient) or higher-order texture (high rank needed).
3. **Is it separable from content bias?** Test on images unlike the training distribution — macro, astro, studio product shots. If rendering transfers, we've captured lens character, not scene priors.
4. **Is the signal visual-texture-based or semantic?** DINOv2 vs. CLIP comparison in Phase 1 answers this: if CLIP performs as well as DINOv2, the classifier is likely cheating on content.

## Key Risks

| Risk | Mitigation |
|---|---|
| Signal too weak (AUC < 0.65) | Increase seed dataset size; try full fine-tuning instead of linear probe |
| Content confound (classifier learns "European street photography") | Stratified sampling by scene type; CLIP vs. DINOv2 comparison as diagnostic |
| LoRA learns color science, not optics | Train on monochrome-converted images as control; if LoRA still transfers, signal is optical |
| Low denoising strength produces artifacts | Test with FLUX.2 klein (4B, faster iteration); use ControlNet for structure preservation |
| Flickr dataset has post-processing contamination | Filter for near-SOOC images via EXIF Software tag; use RAW files where possible |
