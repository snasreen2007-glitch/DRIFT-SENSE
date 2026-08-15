"""
compare_methods.py
=====================
Implements Document Section 9 (Comparison with Conventional Methods).

Runs the SAME 30-sample dataset through:
  1. Template Matching        (plain cv2.matchTemplate, no scale/rotation search)
  2. Normalized Cross-Correlation multi-scale (classical_coarse_matching only)
  3. ORB feature matching     (classical keypoint descriptors + homography)
  4. CNN matching, no fine registration (CNN coarse candidate only, skip 3.6/3.7)
  5. Full DRIFT-SENSE          (the complete hybrid pipeline)

and reports localization error + runtime for each, producing the exact
comparison table shape requested in Section 9.
"""

import json
import os
import time
import numpy as np
import pandas as pd
import cv2

from classical_coarse_matching import classical_topk_candidates
from cnn_feature_extractor import CNNFeatureExtractor
from image_pyramid import build_pyramid
from coarse_matching import topk_candidates
from drift_sense_pipeline import DriftSenseDetector


def method_template_matching(reference, search):
    t0 = time.time()
    result = cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    rh, rw = reference.shape[:2]
    x = max_loc[0] + rw / 2
    y = max_loc[1] + rh / 2
    return x, y, time.time() - t0


def method_ncc_multiscale(reference, search):
    t0 = time.time()
    cands = classical_topk_candidates(search, reference, k=1)
    if not cands:
        return -1, -1, time.time() - t0
    return cands[0].x, cands[0].y, time.time() - t0


def method_orb(reference, search):
    t0 = time.time()
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(reference, None)
    kp2, des2 = orb.detectAndCompute(search, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return -1, -1, time.time() - t0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:30]
    if len(matches) < 4:
        return -1, -1, time.time() - t0
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    rh, rw = reference.shape[:2]
    if H is None:
        return -1, -1, time.time() - t0
    center = np.array([[[rw / 2, rh / 2]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(center, H)[0][0]
    return float(mapped[0]), float(mapped[1]), time.time() - t0


def method_cnn_no_fine_registration(reference, search, extractor):
    t0 = time.time()
    ref_emb = extractor.extract_embedding(reference)
    pyramid = build_pyramid(search, scales=(0.85, 1.0, 1.2))
    fmaps = {s: extractor.extract_map(img) for s, img in pyramid.items()}
    cands = topk_candidates(ref_emb, fmaps, stride=extractor.stride, k=1)
    if not cands:
        return -1, -1, time.time() - t0
    return cands[0].x, cands[0].y, time.time() - t0


def run_comparison(data_dir: str = "data", out_csv: str = "results/method_comparison.csv",
                    cnn_weights_path: str = None):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)

    extractor = CNNFeatureExtractor(backbone="driftsense", weights_path=cnn_weights_path)
    drift_sense = DriftSenseDetector()
    if cnn_weights_path:
        drift_sense.extractor = extractor  # reuse the same trained weights

    rows = []
    for entry in manifest:
        ref = cv2.imread(entry["reference"], cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(entry["search"], cv2.IMREAD_GRAYSCALE)
        with open(entry["ground_truth_file"]) as f:
            gt = json.load(f)["ground_truth"]

        methods = {}
        x, y, t = method_template_matching(ref, search)
        methods["Template Matching"] = (x, y, t)

        x, y, t = method_ncc_multiscale(ref, search)
        methods["NCC Multi-scale"] = (x, y, t)

        x, y, t = method_orb(ref, search)
        methods["SIFT/ORB"] = (x, y, t)

        x, y, t = method_cnn_no_fine_registration(ref, search, extractor)
        methods["CNN Matching (no fine reg.)"] = (x, y, t)

        result = drift_sense.locate(ref, search)
        methods["DRIFT-SENSE (proposed)"] = (result.x, result.y, result.inference_time_s)

        for method_name, (x, y, t) in methods.items():
            err = float(np.hypot(x - gt["x"], y - gt["y"])) if x >= 0 else float("nan")
            rows.append({
                "sample_id": entry["sample_id"], "method": method_name,
                "pred_x": x, "pred_y": y, "localization_error_px": err,
                "runtime_ms": t * 1000,
            })
        print(f"sample {entry['sample_id']:02d} done")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = df.groupby("method").agg(
        mean_error_px=("localization_error_px", "mean"),
        median_error_px=("localization_error_px", "median"),
        failure_rate_pct=("localization_error_px", lambda s: 100 * s.isna().mean()),
        mean_runtime_ms=("runtime_ms", "mean"),
    ).round(2)

    print("\n=== Section 9 comparison table ===")
    print(summary)
    summary.to_csv("results/method_comparison_summary.csv")
    return df, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()
    run_comparison(cnn_weights_path=args.weights)
