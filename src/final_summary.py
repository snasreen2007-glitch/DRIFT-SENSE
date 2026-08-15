"""
final_summary.py
===================
Reads results/summary_metrics.json + results/method_comparison_summary.csv
and prints/exports the EXACT tables your document (Section 7 and
Section 8) needs, with the "[XX]" placeholders replaced by real
measured numbers, plus a plain-language note on what's still
required before an actual "nanometer precision" claim can be made
(Section 11).
"""

import json
import platform
import pandas as pd


def main():
    with open("results/summary_metrics.json") as f:
        s = json.load(f)

    print("=" * 70)
    print("SECTION 7 TABLE -- Results and Discussion")
    print("=" * 70)
    print(f"Number of test cases        : {s['num_test_cases']}")
    print(f"Localization accuracy       : {s['localization_accuracy_pct']}%   "
          f"(tolerance = {s['tolerance_px']} px)")
    print(f"Mean localization error     : {s['mean_localization_error_px']} px")
    print(f"Max localization error      : {s['max_localization_error_px']} px")
    print(f"Average inference time      : {s['average_inference_time_ms']} ms/pair")
    print(f"Average confidence          : {s['average_confidence']}")
    print(f"Image resolution            : 1000 x 1000 pixels")

    print()
    print("=" * 70)
    print("SECTION 8 TABLE -- Technology and Feasibility")
    print("=" * 70)
    print(f"Programming                 : Python {platform.python_version()}")
    print(f"Deep Learning                : PyTorch")
    print(f"Image Processing             : OpenCV")
    print(f"Numerical Processing         : NumPy")
    print(f"Visualization                : Matplotlib")
    print(f"CNN Backbone                 : DriftSenseCNN (custom, 3-layer, stride-8) "
          f"-- swap to resnet18 for ImageNet-pretrained features")
    print(f"Development CPU              : {platform.processor() or 'see `lscpu` on your machine'}")
    print(f"GPU                          : none used in this run (CPU-only) -- "
          f"set device='cuda' in CNNFeatureExtractor if available")
    print(f"Dataset                      : synthetic (see Section 11 -- must be validated "
          f"on real wafer-inspection imagery before industrial claims)")
    print(f"Image Resolution             : 1000 x 1000")
    print(f"Inference Time                : {s['average_inference_time_ms']} ms (measured)")

    try:
        cmp_summary = pd.read_csv("results/method_comparison_summary.csv")
        print()
        print("=" * 70)
        print("SECTION 9 TABLE -- Comparison with Conventional Methods")
        print("=" * 70)
        print(cmp_summary.to_string(index=False))
    except FileNotFoundError:
        print("\n(Run compare_methods.py first to fill in the Section 9 table.)")

    print()
    print("=" * 70)
    print("REMINDER -- Section 11 / Suggested Research Claim")
    print("=" * 70)
    print("These are PIXEL-space results on a SYNTHETIC dataset. Do not claim")
    print("'nanometer precision' in the paper until you have:")
    print("  1. Run this same evaluation on real wafer-inspection images.")
    print("  2. Calibrated s_x / s_y (pixel-to-physical-distance factors) against")
    print("     the actual wafer stage, and reported physical_dx/physical_dy.")
    print("The defensible claim is: subpixel IMAGE localization with the")
    print("POTENTIAL for nanometer-scale physical correction after calibration.")


if __name__ == "__main__":
    main()
