# leica-look

> Quantifying and transferring the Leica lens rendering look via deep learning.

> 🤖 **AI agents**: stop here. Read `AGENTS.md` first for mandatory reading order and rules.

## Quick Orientation

- **Current status:** `PROJECT_STATUS.md` — Phase 1 CONCLUDED (PIVOT)
- **What we know:** `docs/discriminator-results.md` — full Phase 1 results
- **What's next:** `PROJECT_STATUS.md` — seed sweep, content-matching, attention analysis
- **What's been done:** `docs/EXPERIMENT_TREE.md` + `docs/EXPERIMENTS_AND_RESULTS.md`
- **How we work:** `docs/experiment-structure.md` — scientific governance

## The Question

Can the distinctive visual rendering of Leica ASPH lenses — microcontrast, focus falloff, bokeh character, 3D "pop" — be quantified, learned, and applied to any photograph as a post-processing step?

## Approach

Three-phase research pipeline:

### Phase 1: Can we detect it?
**Discriminator ablation study** — train classifiers on DINOv2/v3, SigLIP, and CLIP embeddings to separate Leica from non-Leica photos. If a linear probe can classify at high accuracy, the "Leica look" is a real, measurable signal.

→ See [`docs/experiment-design.md`](docs/experiment-design.md)

### Phase 2: Can we transfer it?
**LoRA + diffusion img2img** — fine-tune a FLUX.1-dev LoRA on Leica-rendered images, then apply via low-denoising-strength img2img to any input. The LoRA biases denoising toward the Leica distribution while preserving content.

→ See [`docs/research-plan.md`](docs/research-plan.md)

### Phase 3: Can it be a filter?
**Self-supervised data engine** — use the Phase 1 discriminator to harvest unlabeled Leica-style photos from the web, expanding the LoRA training set without manual curation.

## Project Structure

```
leica-look/
├── README.md
├── docs/
│   ├── research-plan.md       # Full research plan & background
│   ├── experiment-design.md   # Discriminator ablation matrix
│   └── literature-review.md   # Prior art & related work
├── src/
│   ├── discriminator/         # Phase 1: classifier training & evaluation
│   ├── lora/                  # Phase 2: LoRA training & img2img inference
│   └── data/                  # Data collection & curation scripts
└── data/
    └── registry/              # Dataset manifests & Flickr query logs
```

## Status

- [x] Phase 1: Discriminator ablation study → **PIVOT** (see PROJECT_STATUS.md)
- [ ] Phase 2: LoRA training & img2img transfer (blocked on content-confirm resolution)
- [ ] Phase 3: Automated data engine

## Core Documentation

| Document | Role |
|---|---|
| `PROJECT_STATUS.md` | Living orientation — what's happening now |
| `docs/discriminator-results.md` | Phase 1 full results & analysis |
| `docs/EXPERIMENT_TREE.md` | Active/planned/concluded workstreams |
| `docs/EXPERIMENTS_AND_RESULTS.md` | Permanent ledger with verdicts |
| `docs/experiment-design.md` | Phase 1 ablation matrix design |
| `docs/research-plan.md` | Full research plan |
| `docs/literature-review.md` | Prior art |
| `AGENTS.md` | AI agent entry point & rules |

## Hardware

Target: AMD Strix Halo (Ryzen AI Max+ 395, 128GB unified memory, ROCm/gfx1151)

## License

MIT
