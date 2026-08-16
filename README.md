# DRIFT-SENSE
## Hybrid AI-Based Navigation Error Recovery and Image Localization

DRIFT-SENSE is a hybrid Artificial Intelligence (AI) and computer-vision pipeline developed for image localization and navigation-error recovery.

The system combines CNN-based feature extraction with classical image matching in a coarse-to-fine architecture. It is designed to locate a reference pattern inside a transformed search image while handling translation, scale, rotation, illumination changes, blur, and noise.

---

## 1. Key Features

- CNN-based feature extraction
- Multi-scale image pyramid
- CNN cosine-similarity matching
- Top-K candidate selection
- Classical multi-scale and rotation-aware NCC matching
- Candidate fusion
- Fine registration
- Subpixel localization
- Confidence-aware output
- Synthetic wafer-pattern dataset generation
- Automated evaluation
- Failure-case analysis
- Comparison with conventional methods
- Result visualization
- CNN training support

---

## 2. System Architecture

```text
                         REFERENCE IMAGE
                                |
                                v
                     CNN Feature Extraction
                                |
                                v
                      Reference Embedding
                                |
                  +-------------+-------------+
                  |                           |
                  v                           v
           CNN Coarse Matching        Classical Matching
                  |                           |
                  v                           v
            Top-K Candidates          Multi-scale / Rotation
                  |                           |
                  +-------------+-------------+
                                |
                                v
                       Candidate Fusion
                                |
                                v
                       Fine Registration
                                |
                                v
                     Subpixel Localization
                                |
                                v
                      Confidence Estimation
                                |
                                v
                       FINAL (X, Y) POSITION
```

The pipeline follows a coarse-to-fine strategy:

1. Extract features from the reference image.
2. Build a multi-scale search-image pyramid.
3. Generate candidates using the CNN branch.
4. Generate additional candidates using classical NCC matching.
5. Fuse the candidate sets.
6. Perform fine registration around each candidate.
7. Refine the detected position at subpixel level.
8. Calculate a confidence score.
9. Output the final localized `(x, y)` position.

---

## 3. Main Modules

| File | Function |
|---|---|
| `src/cnn_feature_extractor.py` | CNN feature extraction and dense feature maps |
| `src/image_pyramid.py` | Multi-scale search-image generation |
| `src/coarse_matching.py` | CNN cosine-similarity matching and Top-K selection |
| `src/classical_coarse_matching.py` | Classical multi-scale and rotation-aware NCC matching |
| `src/fine_registration.py` | Fine rotation and translation refinement |
| `src/subpixel.py` | Subpixel peak refinement and physical-coordinate conversion |
| `src/confidence.py` | Confidence estimation and reliability decision |
| `src/drift_sense_pipeline.py` | Complete DRIFT-SENSE pipeline |
| `src/dataset_generator.py` | Synthetic dataset generation |
| `src/evaluate_pipeline.py` | Accuracy, error, confidence and runtime evaluation |
| `src/compare_methods.py` | Comparison with conventional localization methods |
| `src/generate_graphs.py` | Generates result figures |
| `src/final_summary.py` | Generates final result tables |
| `run_pipeline.py` | Runs the complete evaluation pipeline |

---

## 4. Processing Pipeline

### 4.1 CNN Feature Extraction

The reference image and search image are processed using the CNN feature extractor.

The reference image is converted into a feature embedding, while the search image produces dense feature maps.

### 4.2 Multi-scale Image Pyramid

The search image is evaluated at multiple scales to improve robustness against changes in target size.

### 4.3 Coarse Matching

CNN cosine similarity is used to identify promising candidate locations.

Instead of selecting only one location, the system retains multiple Top-K candidates.

### 4.4 Classical Matching

A classical normalized cross-correlation (NCC) branch performs multi-scale and rotation-aware template matching.

This branch provides additional candidate locations when the CNN branch is uncertain.

### 4.5 Candidate Fusion

CNN and classical candidates are combined into a common candidate pool.

The candidates are evaluated using fine registration.

### 4.6 Fine Registration

Each candidate is refined using full-resolution normalized cross-correlation.

Rotation is also considered during the refinement stage.

### 4.7 Subpixel Localization

The peak of the correlation response is refined to estimate a more accurate image-space position.

### 4.8 Confidence Estimation

The confidence score combines:

- Coarse matching score
- Fine registration NCC score
- Separation from the runner-up candidate

A low-confidence result can therefore be identified as potentially ambiguous.

---

## 5. Dataset

The current evaluation uses a synthetic wafer-pattern dataset.

The dataset generator introduces controlled variations including:

- Translation
- Scale
- Rotation
- Illumination changes
- Blur
- Noise
- Other image transformations

Each generated sample contains a corresponding ground-truth localization coordinate.

The current image resolution is:

```text
1000 × 1000 pixels
```

The synthetic dataset is intended for algorithm development and controlled evaluation. Real wafer-inspection data is required before making industrial performance claims.

---

## 6. Installation

Create and activate the Python virtual environment:

```bash
cd ~/DRIFT-SENSE-build
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 7. Running the Pipeline

The complete pipeline can be executed using:

```bash
python run_pipeline.py --n 30 --weights weights/driftsense_cnn.pt
```

For a stress-test configuration:

```bash
python run_pipeline.py --n 30 --weights weights/driftsense_cnn.pt --stress_prob 0.01
```

The pipeline performs:

1. Dataset generation
2. DRIFT-SENSE evaluation
3. Conventional-method comparison
4. Graph generation
5. Final result summarization

---

## 8. Training the CNN Backbone

`DriftSenseCNN` ships untrained by default. `src/train_cnn.py` trains it with a
triplet loss (reference patch = anchor, true-location crop = positive,
random wrong-location crop = negative) using only the
`(reference, search, ground_truth)` triples `dataset_generator.py` already
produces — no manual labeling required.

```
# 1. Generate a training set (hundreds-thousands of samples; 30 is only
#    enough for evaluation, not training)
python src/dataset_generator.py --n 800 --out data_train --seed 1

# 2. Train
python src/train_cnn.py --data_dir data_train --epochs 15 --out weights/driftsense_cnn.pt

# 3. Point the evaluation pipeline at the trained weights
python run_pipeline.py --n 30 --weights weights/driftsense_cnn.pt
```

**Honest note on training set size:** a 300-sample / 15-epoch run reduced
the triplet loss from 0.097 to 0.017 and cut inference time roughly 6×,
but overall localization accuracy on the 30-sample eval set *decreased*
slightly compared to the untrained baseline (96.67% → 90.0%) — the
classical NCC branch was still resolving most cases correctly on its own,
and 300 triplets isn't enough for the CNN branch to consistently add
value on top of it. Scaling to 800+ samples and more epochs is the
natural next step before trusting the CNN branch's contribution.

## 9. Calibrating Pixel-to-Physical Scale

Before reporting any nanometer/micron-scale correction, `s_x` and `s_y`
(pixels → physical distance) must be calibrated against the real imaging
and stage system — see `src/calibrate_scale.py`. It takes a set of
reference/search image pairs captured after a **known** physical stage
move, measures the pixel shift DRIFT-SENSE recovers, and derives
`s_x = known_physical_dx / measured_pixel_dx` (averaged over several
moves). Until this is done, only pixel-space accuracy is defensible.

```
python src/calibrate_scale.py --calib_dir calibration_shots --weights weights/driftsense_cnn.pt
```

## 10. Results

The latest 30-sample evaluation (trained CNN, 300-sample/15-epoch run)
produced the following results:

| Metric | Result |
|---|---:|
| Number of test cases | 30 |
| Localization accuracy | **90.0%** |
| Tolerance | 20 px |
| Mean localization error | **11.887 px** |
| Maximum localization error | **154.159 px** |
| Average confidence | **0.878** |
| Average inference time | **129.23 ms** |
| Image resolution | **1000 × 1000 pixels** |

One failure case was recorded at sample 23 with an error of approximately
154.16 pixels. Its confidence score was 0.76 and was flagged
**reliable=true** — the confidence estimator did not catch this failure,
which is a real limitation worth stating explicitly rather than implying
confidence reliably filters every failure mode.

This failure is intentionally retained for honest failure analysis rather
than reporting an artificial 100% success rate.

*(Both the pre-training and post-training runs are synthetic-dataset
results with an untrained-vs-trained CNN branch; neither should be read
as a claim about real wafer-inspection performance — see Limitations.)*

---

## 11. Comparison with Conventional Methods

The latest evaluation (trained CNN, 300-sample/15-epoch run) produced the following comparison:

| Method | Mean Error (px) | Median Error (px) | Failure Rate (%) | Runtime (ms) |
|---|---:|---:|---:|---:|
| CNN Matching (no fine reg.) | 101.30 | 113.93 | 0.0 | 36.17 |
| **DRIFT-SENSE (proposed)** | **11.89** | **0.12** | 0.0 | 119.97 |
| NCC Multi-scale | 11.52 | 0.43 | 0.0 | 40.99 |
| SIFT/ORB | — | — | **100.0** | 2.91 |
| Template Matching | 11.81 | 0.54 | 0.0 | 1.69 |

DRIFT-SENSE still provides substantially lower mean localization error than the CNN-only baseline. Its median error (0.12px) is the best of the group, meaning it's usually very precise, but its mean is close to the classical NCC and template-matching baselines — a few harder cases (like sample 23, see Section 10) pull the mean up.

**Known issue:** SIFT/ORB failed on 100% of test cases in this run — almost certainly too few detectable keypoints on the synthetic wafer patterns, not a fundamental SIFT/ORB weakness. This should be investigated (e.g. tune the ORB detector's feature-count/threshold parameters, or use richer synthetic textures) before this row is presented as a fair comparison in the paper.

The classical NCC-based methods remain faster, while the proposed hybrid pipeline combines learned feature extraction, candidate generation, fine registration, subpixel localization and confidence estimation.

---

## 12. Output Files

After running the pipeline, results are generated in:

```text
data/
results/
figures/
```

Important result files include:

```text
results/detection_results.csv
results/summary_metrics.json
results/failure_case.json
results/method_comparison_summary.csv
```

Generated graphs are stored in:

```text
figures/
```

---

## 13. Technology Stack

```text
Programming       : Python 3
Deep Learning     : PyTorch
Image Processing  : OpenCV
Numerical         : NumPy
Data Analysis     : Pandas
Visualization     : Matplotlib
```

The current evaluation was performed on CPU.

---

## 14. Project Structure

```text
DRIFT-SENSE-build/
│
├── src/
│   ├── cnn_feature_extractor.py
│   ├── image_pyramid.py
│   ├── coarse_matching.py
│   ├── classical_coarse_matching.py
│   ├── fine_registration.py
│   ├── subpixel.py
│   ├── confidence.py
│   ├── drift_sense_pipeline.py
│   ├── dataset_generator.py
│   ├── evaluate_pipeline.py
│   ├── compare_methods.py
│   ├── generate_graphs.py
│   ├── final_summary.py
│   ├── train_cnn.py
│   └── calibrate_scale.py
│
├── weights/
│   └── driftsense_cnn.pt
│
├── data_train/          (generated by dataset_generator.py for training)
├── calibration_shots/   (your real known-displacement calibration images)
├── data/
│   ├── reference/
│   ├── test/
│   └── manifest.json
│
├── results/
├── figures/
├── requirements.txt
├── run_pipeline.py
└── README.md
```

---

## 15. Limitations

The current results should be interpreted within the scope of the synthetic dataset.

Important limitations are:

- The evaluation is not yet a validation on real wafer-inspection imagery.
- Pixel-space accuracy does not directly represent physical stage accuracy.
- Physical or nanometer-scale claims require calibration of the pixel-to-distance parameters (see `src/calibrate_scale.py`, Section 9).
- Performance can vary with image quality, pattern repetitiveness and severe transformations.
- The current CPU implementation has higher runtime than purely classical template matching.
- The 300-sample / 15-epoch trained CNN did not outperform the untrained baseline on this dataset — the classical NCC branch still resolves most cases; a larger training set is needed before the CNN branch's contribution can be trusted (Section 8).
- The confidence estimator does not catch every failure: the recorded failure case (sample 23, Section 10) was flagged `reliable=true` despite a 154px error, so confidence should not be treated as a hard reliability guarantee.
- SIFT/ORB failed on 100% of samples in the current comparison run; this is likely a parameter/data issue with the synthetic patterns rather than a fundamental result and should be fixed before publishing the comparison table as-is.

---

## 16. Future Improvements

Future development can include:

1. Scaling the training set to 800+ samples (per `train_cnn.py`'s own guidance) and re-evaluating whether the trained CNN branch beats the untrained baseline.
2. Getting access to even a small real wafer-inspection image set for validation beyond synthetic data.
3. Improving robustness against repetitive patterns.
4. Optimizing inference for GPU execution.
5. Reducing the computational cost of multi-scale and rotation search.
6. Running `src/calibrate_scale.py` against the real imaging/stage setup to obtain defensible `s_x`/`s_y`.
7. Evaluating the system using real navigation/stage measurements.
8. Performing larger-scale statistical validation.
9. Debugging the SIFT/ORB 100% failure rate before including it in the paper's comparison table.
10. Investigating why the confidence estimator missed the sample-23 failure, to make `reliable` a more trustworthy flag.

---

## 17. Research Note

The current results demonstrate **subpixel image localization capability on a synthetic dataset**.

The results should not be interpreted as nanometer-scale physical positioning accuracy unless the image-to-physical calibration has been experimentally established using the actual imaging and stage system.

---

## 18. License

This project is intended for academic, educational, and research purposes.

The source code may be used, modified, and extended for non-commercial research and learning. Any use of the project or results should appropriately acknowledge the DRIFT-SENSE project and its contributors.

For commercial use or redistribution, please contact the project owner for permission.
