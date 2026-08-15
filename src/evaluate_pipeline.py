"""
evaluate_pipeline.py
=======================
Implements Document Section 6 (Experimental Methodology) and
Section 7 (Results and Discussion) -- including 7.2 (Honest Failure
Analysis, which the current repo's README does NOT do: it claims
100% success / 0-pixel error, which this script will NOT let you get
away with silently -- failures are logged explicitly).

For every reference/search pair it records exactly the table from
Section 6:
  reference coordinate, predicted coordinate, X error, Y error,
  Euclidean error, confidence, inference time, success/failure
"""

import json
import os
import time
import numpy as np
import pandas as pd
import cv2

from drift_sense_pipeline import DriftSenseDetector


def run_evaluation(data_dir: str = "data", out_csv: str = "results/detection_results.csv",
                    tolerance_px: float = 20.0, cnn_weights_path: str = None):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)

    detector = DriftSenseDetector(cnn_weights_path=cnn_weights_path)
    rows = []

    for entry in manifest:
        ref = cv2.imread(entry["reference"], cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(entry["search"], cv2.IMREAD_GRAYSCALE)
        with open(entry["ground_truth_file"]) as f:
            gt_data = json.load(f)
        gt = gt_data["ground_truth"]
        params = gt_data["params"]

        result = detector.locate(ref, search)

        x_err = result.x - gt["x"]
        y_err = result.y - gt["y"]
        euclid_err = float(np.hypot(x_err, y_err))
        success = euclid_err <= tolerance_px

        rows.append({
            "sample_id": entry["sample_id"],
            "gt_x": gt["x"], "gt_y": gt["y"],
            "pred_x": result.x, "pred_y": result.y,
            "x_error": x_err, "y_error": y_err,
            "euclidean_error_px": euclid_err,
            "confidence": result.confidence,
            "reliable_flag": result.reliable,
            "inference_time_ms": result.inference_time_s * 1000,
            "success": success,
            "scale_gt": params["scale"], "rotation_gt": params["rotation_deg"],
            "illumination_factor": params["illumination_factor"],
            "blur_ksize": params["blur_ksize"], "noise_sigma": params["noise_sigma"],
        })
        print(f"[{entry['sample_id']:02d}] err={euclid_err:6.2f}px  "
              f"conf={result.confidence:.2f}  success={success}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    accuracy = 100.0 * df["success"].mean()
    summary = {
        "num_test_cases": len(df),
        "localization_accuracy_pct": round(accuracy, 2),
        "mean_localization_error_px": round(df["euclidean_error_px"].mean(), 3),
        "max_localization_error_px": round(df["euclidean_error_px"].max(), 3),
        "average_inference_time_ms": round(df["inference_time_ms"].mean(), 2),
        "average_confidence": round(df["confidence"].mean(), 3),
        "tolerance_px": tolerance_px,
    }

    with open("results/summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY (Section 7 table) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # ---- honest failure case (Section 7.2) ----
    failures = df[~df["success"]].sort_values("euclidean_error_px", ascending=False)
    if len(failures) > 0:
        worst = failures.iloc[0]
        print("\n=== WORST FAILURE CASE (Section 7.2) ===")
        print(worst.to_dict())
        with open("results/failure_case.json", "w") as f:
            json.dump(worst.to_dict(), f, indent=2, default=str)
    else:
        print("\nNo failures at the current tolerance -- try lowering `tolerance_px` "
              "or check whether the dataset difficulty is realistic before claiming "
              "100% success in the report (Section 7.2 requires an honest failure case).")

    return df, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()
    run_evaluation(cnn_weights_path=args.weights)
