# Experiment Tree

Living map of active, planned, and concluded workstreams. Status tags: `[ACTIVE]`, `[CONCLUDED]`, `[TBD]`. Each entry links to the experiment arm or issue.

---

## Active

*None currently. Phase 1 is concluded with a PIVOT verdict. See PROJECT_STATUS.md for next actions.*

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

* **[TBD] Content-matching experiment** — Pair-match Leica/non-Leica images by scene type and composition. Re-run probes on matched pairs to isolate lens signal.

* **[TBD] Seed sweep** — Re-run top-10 LR configurations with 5 random seeds to measure AUC variance and confirm results aren't lucky splits.

* **[TBD] Attention-map analysis** — Identify which image regions drive DINOv2 classification. Center-weighted (subject) = content confound; edge-weighted (bokeh, vignetting) = lens signal.

* **[TBD] Controlled capture** — Photograph same scene with Leica + non-Leica lenses on same body (adapter mount) to isolate rendering from sensor/body.

* **[TBD] Per-class accuracy logging** — Needed for understanding what kinds of images are misclassified.

### Phase 2: LoRA Training (blocked on Phase 1.5)

* **[TBD] LoRA fine-tuning on DINOv2-g** — Train model to reproduce Leica rendering on non-Leica images. Requires validated signal from Phase 1.5.

* **[TBD] Image-to-image translation evaluation** — FID, LPIPS, human preference study.
