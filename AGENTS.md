# AGENTS.md — leica-look

AI agent entry point. Read these in order before starting any work.

## Mandatory reading order

1. **`PROJECT_STATUS.md`** — current phase, blockers, immediate next action
2. **`docs/EXPERIMENT_TREE.md`** — what's active, concluded, TBD
3. **`docs/EXPERIMENTS_AND_RESULTS.md`** — permanent ledger with all verdicts
4. **This file** — project conventions and rules

## Project conventions

### Hardware
- Local GPU: RTX 4090 (24GB) on game box
- Remote GPU: Strix Halo max395 (128GB unified, ROCm) via SSH tim@192.168.86.137
- GPU scheduler: `/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py`
- Model cache: `HF_HOME=/mnt/nas-ai-models/huggingface-cache`
- Training data: `/mnt/nas-ai-models/training-data/leica-look/`

### Data
- Positive manifest: `data/registry/positive_manifest_final.csv` (270 EXIF-verified Leica images)
- Negative manifest: `data/registry/negative_manifest.csv` (334 EXIF-verified non-Leica images)
- Verified CSV: `data/registry/verified.csv`
- Embeddings: `/mnt/nas-ai-models/training-data/leica-look/embeddings/`
- Raw images: `/mnt/nas-ai-models/training-data/leica-look/raw/`

### Code
- Experiment-specific code lives in `experiments/{arm}/src/`, not in `src/`
- Probe scripts: `src/run_logistic_probes.py`, `src/run_mlp_probes.py`, `src/run_knn_probes.py`
- All configs go through `experiments/{arm}/config.yaml`
- GPU work uses the scheduler: request → poll → activate → launch → heartbeat → release

### Scientific governance
- Pre-register gates BEFORE seeing results
- Run adversarial pass before any PASS verdict
- Never re-run a [CONCLUDED — FAIL] or KILLed experiment
- Config is frozen at run start (copy to runs/{timestamp}/)
- Write ledger entry for every experiment

## Critical rules

1. **Do NOT proceed to Phase 2 (LoRA training) without resolving the content confound.** CLIP's +5.5 pt AUC lead over DINOv2 confirms the classifier detects subject matter, not lens rendering.
2. **Never trust a single AUC number at n=50.** Test sets are 30 images — wide confidence intervals.
3. **Always check PROJECT_STATUS.md first.** The project may be PARKed or between phases.
4. **Code has NO unit tests.** Any new metric code needs tests before trusting its output.
5. **The body/brand confound is inherent.** All positives are Leica bodies, all negatives are Canon/Sony/Nikon. Stratification can't fix this.
