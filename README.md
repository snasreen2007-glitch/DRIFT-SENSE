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

## 8. Results

The latest 30-sample evaluation produced the following results:

| Metric | Result |
|---|---:|
| Number of test cases | 30 |
| Localization accuracy | **96.67%** |
| Tolerance | 20 px |
| Mean localization error | **2.216 px** |
| Maximum localization error | **62.973 px** |
| Average confidence | **0.872** |
| Average inference time | **771.82 ms** |
| Image resolution | **1000 × 1000 pixels** |

One failure case was recorded at sample 23 with an error of approximately 62.97 pixels.

This failure is intentionally retained for honest failure analysis rather than reporting an artificial 100% success rate.

---

## 9. Comparison with Conventional Methods

The latest evaluation produced the following comparison:

| Method | Mean Error (px) | Median Error (px) | Runtime (ms) |
|---|---:|---:|---:|
| CNN Matching | 114.43 | 124.33 | 403.13 |
| **DRIFT-SENSE** | **2.22** | **0.10** | 674.20 |
| NCC Multi-scale | 2.50 | 0.45 | 46.26 |
| SIFT/ORB | — | — | 102.35 |
| Template Matching | 2.71 | 0.61 | 1.82 |

DRIFT-SENSE provides substantially lower mean localization error than the CNN-only baseline in the current synthetic evaluation.

The classical NCC-based methods are faster, while the proposed hybrid pipeline combines learned feature extraction, candidate generation, fine registration, subpixel localization and confidence estimation.

---

## 10. Output Files

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

## 11. Technology Stack

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

## 12. Project Structure

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
│   └── final_summary.py
│
├── weights/
│   └── driftsense_cnn.pt
│
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

## 13. Limitations

The current results should be interpreted within the scope of the synthetic dataset.

Important limitations are:

- The evaluation is not yet a validation on real wafer-inspection imagery.
- Pixel-space accuracy does not directly represent physical stage accuracy.
- Physical or nanometer-scale claims require calibration of the pixel-to-distance parameters.
- Performance can vary with image quality, pattern repetitiveness and severe transformations.
- The current CPU implementation has higher runtime than purely classical template matching.

---

## 14. Future Improvements

Future development can include:

1. Training and fine-tuning the CNN on real wafer-inspection data.
2. Increasing the diversity and realism of the synthetic dataset.
3. Improving robustness against repetitive patterns.
4. Optimizing inference for GPU execution.
5. Reducing the computational cost of multi-scale and rotation search.
6. Calibrating pixel-to-physical-distance conversion.
7. Evaluating the system using real navigation/stage measurements.
8. Performing larger-scale statistical validation.

---

## 15. Research Note

The current results demonstrate **subpixel image localization capability on a synthetic dataset**.

The results should not be interpreted as nanometer-scale physical positioning accuracy unless the image-to-physical calibration has been experimentally established using the actual imaging and stage system.

---

## 16. License

This project is intended for academic, educational, and research purposes.

The source code may be used, modified, and extended for non-commercial research and learning. Any use of the project or results should appropriately acknowledge the DRIFT-SENSE project and its contributors.

For commercial use or redistribution, please contact the project owner for permission.
