# DRIFT-SENSE — Hybrid AI Pipeline (new additions)

This folder adds the missing pieces identified when comparing the
existing repo against `DRIFT-SENSE.docx`: a real CNN feature-extraction
branch, coarse-to-fine matching with Top-K candidates, fine
registration, subpixel localization, and confidence-aware output —
combined ("hybrid") with the classical multi-scale/rotation matching
the repo already had.

## New files

| File | Document section | What it does |
|---|---|---|
| `src/cnn_feature_extractor.py` | 3.1, 3.3 | CNN backbone → reference embedding + dense search feature maps |
| `src/image_pyramid.py` | 3.2 | Builds the multi-scale search-image pyramid |
| `src/coarse_matching.py` | 3.4, 3.5 | Cosine-similarity coarse matching + Top-K candidate selection (CNN branch) |
| `src/classical_coarse_matching.py` | 5 | Classical multi-scale/rotation NCC candidates (classical branch) |
| `src/fine_registration.py` | 3.6 | Per-candidate rotation+translation refinement via NCC |
| `src/subpixel.py` | 3.7 | Parabolic subpixel peak refinement + pixel→physical conversion |
| `src/confidence.py` | 3.8 | Confidence score from coarse score, fine NCC, and candidate margin |
| `src/drift_sense_pipeline.py` | Full Section 3 | Orchestrates every stage above into one `DriftSenseDetector` |
| `src/dataset_generator.py` | 4 | Synthetic dataset: offset, scale, rotation, illumination, blur, noise, occlusion, ground truth |
| `src/evaluate_pipeline.py` | 6, 7 | Runs the full dataset, computes accuracy/error/time, logs an honest failure case |
| `src/compare_methods.py` | 9 | Template Matching, NCC, ORB, CNN-only (ablation), full DRIFT-SENSE |
| `src/generate_graphs.py` | 7 figures | Produces the result plots |
| `src/final_summary.py` | 7, 8, 9 tables | Fills in the `[XX]` placeholders from your document with real numbers |
| `src/train_cnn.py` | 5, Limitations | Triplet-loss training for `DriftSenseCNN` using existing dataset triples |
| `src/calibrate_scale.py` | 3.7, 11 | Derives `s_x`/`s_y` from known-displacement calibration shots |
| `run_pipeline.py` | — | One command to run everything above in order |

## How to run

```bash
pip install -r requirements.txt
python run_pipeline.py --n 30
```

Outputs land in `data/`, `results/`, and `figures/`.

## Honest state of the results (read before writing the paper)

- These numbers come from a **synthetic** wafer-pattern dataset (see
  `src/dataset_generator.py`), not a real wafer inspection tool.
- The CNN backbone (`DriftSenseCNN`) is **randomly initialized**, not
  trained. It's kept in the loop because your document's Section 5
  requires the hybrid architecture, and because training on real wafer
  data is the natural next step — but right now the classical NCC
  branch is doing most of the heavy lifting. Expect the CNN branch's
  contribution (and overall accuracy) to improve substantially once you
  fine-tune it on real or more-realistic data.
- `results/summary_metrics.json` and `results/method_comparison_summary.csv`
  contain the exact numbers to paste into your document's Section 7/8/9
  tables — don't hand-estimate them.
- Do **not** claim "nanometer precision" until `s_x`/`s_y` in
  `DriftSenseDetector.__init__` are calibrated against a real wafer
  stage and you've reported `physical_dx`/`physical_dy`, per Section 11.

## Status update: first real training run completed

`train_cnn.py` was run with 300 generated triplets / 15 epochs. Triplet
loss dropped 0.097 → 0.017 (the CNN branch is genuinely learning), and
using the trained weights cut mean inference time from 771.82ms to
129.23ms. **However, overall localization accuracy on the 30-sample eval
set went from 96.67% (untrained) to 90.0% (trained)** — the classical NCC
branch was already resolving most cases, and 300 triplets isn't enough
for the CNN branch to consistently add value on top of it yet.
`train_cnn.py`'s own docstring says "hundreds-thousands" of samples are
needed for real training — 300 was a first smoke-test run, not the final
number.

`calibrate_scale.py` now exists but has **not** been run yet — it needs
real calibration shots (known physical stage moves) that only exist on
the actual imaging system, not something that can be simulated here.

## Next steps to make this submission-ready

1. Re-run `train_cnn.py` with 800+ samples (`python src/dataset_generator.py --n 800 --out data_train`)
   and more epochs, and confirm the trained CNN branch beats the
   untrained baseline before trusting its numbers.
2. Replace the synthetic dataset with real (or more realistic) wafer
   images if you have access to any, even a small sample.
3. Capture real known-displacement calibration shots and run
   `python src/calibrate_scale.py --calib_dir calibration_shots` to get
   defensible `s_x`, `s_y` — do not hand-estimate these.
4. Regenerate `results/` and `figures/` after each change and copy the
   numbers into the Section 7/8/9 tables of `DRIFT-SENSE.docx` — the
   current README already reflects the latest 300-sample/15-epoch run.
5. Produce the remaining architecture/pipeline diagrams (Figures 1–6, 8, 9, 12)
   — these are conceptual diagrams, not generated from code.
6. Investigate the 100% SIFT/ORB failure rate in `compare_methods.py`'s
   output before publishing that comparison row as-is.
7. Look into why the confidence estimator flagged the sample-23 failure
   (154px error) as `reliable=true` — this weakens the confidence-aware
   claim in Section 3.8 of the document until addressed.
