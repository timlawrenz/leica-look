# Literature Review: Lens Rendering & Style Transfer

## Lens Rendering Quantification

### NeuroLens: Data-Driven Camera Lens Simulation Using Neural Networks
**Zheng & Zheng, Eurographics 2017**

Learns a lens's ray-transfer function (IRF) from ray-trace data, mapping incident rays to emergent rays via neural networks. Can simulate how a specific lens renders a 3D scene. Uses kd-tree spatial subdivision + ensemble of local neural networks for accuracy.

**Relevance:** First demonstration that lens rendering can be captured by a learned function. Ray-transfer approach, not image-to-image — useful as a rendering tool, not for post-processing.

### Neural Lens Modeling
**Xian et al., CVPR 2023**

Models lens distortion and vignetting using invertible neural networks (INNs). Uses the LensFun database (400+ real lens profiles) to create SynLens, a synthetic dataset for lens model evaluation. Achieves subpixel accuracy in calibration.

**Relevance:** Shows that LensFun profiles can serve as a bridge between optical measurements and learned models. The invertible residual network approach could potentially model the full rendering pipeline.

### Ray-Transfer Functions for Camera Simulation
**Goossens et al., Optics Express 2022**

Polynomial ray-transfer functions that characterize how rays enter and exit a lens at any position. Software provided. Combines image sensor simulation with physically-based ray tracing.

**Relevance:** A bridge between pure optical simulation and learned models — the RTF polynomial could serve as a physics-based prior for neural rendering transfer.

---

## Lens-to-Lens Bokeh Transfer

### Lens-to-Lens Bokeh Effect Transformation (NTIRE 2023 Challenge)
**Conde et al., CVPRW 2023**

A competition specifically on transforming the bokeh effect of one lens to another. Given an image shot with Lens A, produce the same scene as if shot with Lens B. Uses paired datasets (same scene, multiple lenses).

**Relevance: THIS IS THE MOST DIRECTLY RELATED WORK.** Exact same goal — lens rendering transfer. Key difference: uses paired data (same scene, different lenses), which we're trying to avoid.

### EBokehNet: Efficient Multi-Lens Bokeh Effect Rendering and Transformation
**Seizinger et al., CVPRW 2023**

State-of-the-art solution from the NTIRE 2023 challenge. Takes an all-in-focus image (or lens-A image) plus lens metadata (focal length, aperture) and outputs the lens-B rendering. Efficient enough for real-time use.

**Relevance:** Architecture reference for Phase 2. Demonstrates that lens rendering transfer is feasible with current methods.

### Bokehlicious: Photorealistic Bokeh Rendering with Controllable Apertures
**arXiv 2025**

Aperture-aware transformer attention that mimics physical aperture mechanics by adapting attention mask width to f-stop. Works directly from all-in-focus images.

**Relevance:** Modern transformer-based approach to bokeh rendering. Could be adapted for multi-lens character transfer beyond just bokeh.

### Selective Bokeh Effect Transformation
**Peng et al., CVPR 2023**

Allows per-region lens effect control — specify which regions get which lens treatment. Combines bokeh rendering and defocus deblurring.

**Relevance:** If the Leica look is spatially varying (different behavior at different depths/apertures), selective application could be necessary.

---

## Photorealistic Style Transfer

### Deep Photo Style Transfer
**Luan et al., CVPR 2017**

The canonical photorealistic style transfer paper. Constrains the transform to be locally affine in color space (Matting Laplacian), preventing the "painterly" distortions of Gatys-style neural style transfer. Preserves photorealism while transferring color/texture.

**Relevance:** Foundation for any photorealistic rendering transfer. The locally-affine constraint ensures the output still looks like a photograph.

### BLoRA: Content-Style Disentanglement for Diffusion Style Transfer
**2024**

Uses SDXL + LoRA-based fine-tuning to explicitly separate content and style representations. Achieves better content preservation than prompt-based methods.

**Relevance:** Direct architectural reference for Phase 2 — using LoRA for style disentanglement in diffusion models.

---

## Industrial Applications

### Canon Neural Network Lens Optimizer
**Canon, 2023**

Deep learning to *remove* lens aberrations and diffraction blur. Trained on paired data (aberrated vs. ideal) using Canon's internal optical design knowledge. Modified architecture to prevent "false corrections."

**Relevance:** The inverse of our goal (removing lens character vs. applying it). Validates that deep learning can manipulate lens rendering at production quality. Their architecture choices for avoiding artifacts are directly applicable.

---

## Dataset & Camera Identification

### Camera Model Identification (Forensic)
**Multiple papers, 2016–2025**

Subfield of digital forensics that identifies camera make/model from image content. Works primarily on sensor-level artifacts: PRNU noise patterns, demosaicing artifacts, JPEG compression signatures.

**Relevance:** Different signal (sensor, not lens). Our approach is novel in targeting lens rendering specifically. The forensic literature provides robust evaluation methodology.

### LensFun Database
**Open-source, community-maintained**

400+ real lens profiles with distortion, vignetting, and TCA measurements. Used by raw processing software (Darktable, RawTherapee) and the SynLens dataset.

**Relevance:** Could provide physics-based priors for neural models, or serve as a bridge between optical measurements and learned representations.

---

## Key Gap

**No existing work attempts unpaired lens rendering transfer** — all current approaches either:
1. Use paired data (same scene, different lenses) — expensive to collect
2. Model the optical pipeline directly (requires lens design specs)
3. Transfer artistic styles, not optical characteristics

Our approach — discriminator-guided unpaired learning + diffusion LoRA — is novel in this specific application.
