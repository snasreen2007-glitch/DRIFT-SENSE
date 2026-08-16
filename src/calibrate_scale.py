"""
calibrate_scale.py
====================
Computes the pixel-to-physical calibration factors s_x, s_y that
DriftSenseDetector needs (Section 3.7 / Section 11) before ANY
nanometer/micron claim is defensible.

Method: known-displacement calibration.
  1. On the real imaging system, move the wafer stage by a KNOWN
     physical distance (e.g. 5000 nm in x, 0 nm in y) using the stage
     controller itself -- not DRIFT-SENSE.
  2. Capture a reference image before the move and a search image
     after the move.
  3. Run DRIFT-SENSE (or simple correlation) to measure how many
     pixels the pattern shifted.
  4. s_x = known_physical_dx / measured_pixel_dx  (repeat for y, and
     average multiple moves for a robust estimate).

This script automates steps 3-4 given a folder of
(reference, search, known_dx_nm, known_dy_nm) calibration shots.

Usage:
    python src/calibrate_scale.py --calib_dir calibration_shots

Expects calibration_shots/manifest.json:
    [
      {"reference": "ref.png", "search": "shot_dx5000.png",
       "known_dx": 5000.0, "known_dy": 0.0},
      {"reference": "ref.png", "search": "shot_dy5000.png",
       "known_dx": 0.0, "known_dy": 5000.0},
      ...
    ]
(units of known_dx/known_dy are whatever physical unit you want
s_x/s_y reported in -- nm, um, your choice, just be consistent.)

Output: prints the calibrated s_x, s_y (mean +/- std across shots) and
writes calibration_result.json you can paste into DriftSenseDetector(
s_x=..., s_y=...).
"""

import argparse
import json
import os
import statistics

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from drift_sense_pipeline import DriftSenseDetector


def measure_pixel_shift(detector: DriftSenseDetector, reference_path: str, search_path: str):
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if reference is None or search is None:
        raise FileNotFoundError(f"Could not read {reference_path} or {search_path}")

    result = detector.locate(reference, search)

    # Pixel shift = predicted position minus where the reference patch
    # started (its center, since reference is a small crop). Assumes
    # the reference crop's own center is the nominal zero point --
    # adjust if your capture setup differs.
    ref_h, ref_w = reference.shape
    origin_x, origin_y = ref_w / 2, ref_h / 2
    pixel_dx = result.x - origin_x
    pixel_dy = result.y - origin_y
    return pixel_dx, pixel_dy, result.confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, default=None,
                         help="trained CNN weights, if available")
    args = parser.parse_args()

    with open(os.path.join(args.calib_dir, "manifest.json")) as f:
        manifest = json.load(f)

    detector = DriftSenseDetector(cnn_weights_path=args.weights)

    sx_estimates, sy_estimates = [], []
    for entry in manifest:
        ref_path = os.path.join(args.calib_dir, entry["reference"])
        search_path = os.path.join(args.calib_dir, entry["search"])
        pixel_dx, pixel_dy, conf = measure_pixel_shift(detector, ref_path, search_path)

        known_dx, known_dy = entry["known_dx"], entry["known_dy"]
        print(f"{entry['search']}: measured pixel shift=({pixel_dx:.3f}, {pixel_dy:.3f}) "
              f"px, known physical shift=({known_dx}, {known_dy}), confidence={conf:.3f}")

        if abs(pixel_dx) > 1e-6 and abs(known_dx) > 1e-9:
            sx_estimates.append(known_dx / pixel_dx)
        if abs(pixel_dy) > 1e-6 and abs(known_dy) > 1e-9:
            sy_estimates.append(known_dy / pixel_dy)

    if not sx_estimates or not sy_estimates:
        print("\nWARNING: not enough non-degenerate shots to estimate both s_x and s_y. "
              "Include at least one pure-x move and one pure-y move.")

    result = {
        "s_x_mean": statistics.mean(sx_estimates) if sx_estimates else None,
        "s_x_std": statistics.pstdev(sx_estimates) if len(sx_estimates) > 1 else 0.0,
        "s_y_mean": statistics.mean(sy_estimates) if sy_estimates else None,
        "s_y_std": statistics.pstdev(sy_estimates) if len(sy_estimates) > 1 else 0.0,
        "n_shots": len(manifest),
    }

    print("\n=== Calibration result ===")
    print(json.dumps(result, indent=2))
    print("\nUse these in DriftSenseDetector(s_x=<s_x_mean>, s_y=<s_y_mean>).")
    print("If s_x_std / s_y_std is large relative to the mean, capture more shots "
          "across the field of view -- optical distortion can make the scale factor "
          "position-dependent.")

    out_path = os.path.join(args.calib_dir, "calibration_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
