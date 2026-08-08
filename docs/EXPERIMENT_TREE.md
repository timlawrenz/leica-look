# Experiment Tree

Living map of active, planned, and concluded workstreams. Status tags: `[ACTIVE]`, `[CONCLUDED]`, `[TBD]`. Each entry links to the experiment arm or issue.

---

## Active

* **[ACTIVE — PENDING HUMAN REVIEW] Phase 2a: LoRA Transfer Feasibility** (`experiments/lora-transfer/`)
  * FLUX.1-dev LoRA (rank=32) on 270 Leica images with edge-weighted loss.
  * Governance only — training blocked on human design review (issue #14).
  * Pre-registered gate: DINOv2 embedding shift + attention C/E ratio + CLIP-I.
  * See `experiments/lora-transfer/provenance.yaml` for full gate criteria.

---

## Concluded

* **[CONCLUDED — PIVOT] Phase 1: Discriminator Ablation Study** (issues #1–#8)
  * 7 models × 6 pooling × 3 classifiers × 3 sizes = 1,008 evaluations.
  * CLIP dominates (AUC 0.987) — content confound confirmed.
  * DINOv2 achieves 0.86–0.93 AUC — real signal exists but unvalidated.
  * MLP does not beat LR — signal is linearly separable.
  * Verdict: PIVOT. Need content-matching before Phase 2.
  * Full results: `docs/discriminator-results.md`

* **[CONCLUDED — DONE] Issue #1: Download vision models** — Models cached to NAS.

* **[CONCLUDED — DONE] Issue #2: Scrape seed dataset** — 270 pos + 334 neg EXIF-verified images.

* **[CONCLUDED — DONE] Issue #3: Download & verify images** — Verified.csv written, images on NAS.

* **[CONCLUDED — DONE] Issue #4: Extract embeddings** — 7×6 embedding sets cached to NAS.

* **[CONCLUDED — DONE] Issue #5: Logistic regression probes** — 504 evaluations, CLIP 0.987 best.

* **[CONCLUDED — DONE] Issue #6: MLP probes** — 126 evaluations, MLP ≤ LR for 6/7 models.

* **[CONCLUDED — DONE] Issue #7: k-NN probes** — 378 evaluations, k=5 sweet spot.

* **[CONCLUDED — DONE] Issue #8: Compile results & write ledger** — This entry.

---

## TBD

### Phase 1.5: Content Confound Resolution

* **[CONCLUDED — GO] Content-matching experiment (#11)** — CLIP-matched + aspect-matched pairs. DINOv2-g 0.925–0.942 AUC on matched set. CLIP gap narrows 24–34% but doesn't close. See `experiments/discriminator-content-matched/results.csv`.

* **[CONCLUDED — GO] Seed sweep (#10)** — Top-10 LR configs with 5 seeds. Confirms n=50/class are lucky splits (σ 0.04–0.08); n=250/class stable (σ ≤ 0.02). See `experiments/discriminator-seed-sweep/results.csv`.

* **[CONCLUDED — GO] Attention-map analysis (#12)** — DINOv2-S/14 + DINOv2-g/14 CLS-to-patch attention on 100 images. Correct classifications use more edge features (C/E ratio 2.79 vs 3.51 for incorrect). Effect is real but modest (±3.5 std). Supports lens-signal-at-edges hypothesis. See `experiments/discriminator-attention/analysis.md`.

* **[TBD] Controlled capture** — Photograph same scene with Leica + non-Leica lenses on same body (adapter mount) to isolate rendering from sensor/body.

* **[TBD] Per-class accuracy logging** — Needed for understanding what kinds of images are misclassified.

### Phase 2: LoRA Training (Phase 2a pending human review)

* **[ACTIVE — PENDING HUMAN REVIEW] LoRA transfer feasibility** — FLUX.1-dev + edge-weighted loss on 270 Leica images. See `experiments/lora-transfer/`. Blocked on design review (issue #14).

* **[TBD] Scale to full dataset LoRA** — If Phase 2a passes, expand training.

* **[TBD] Human preference study** — If transfer is measurable, validate with human raters.
