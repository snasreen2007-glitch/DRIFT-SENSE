"""
detect_and_visualize.py
==========================
Run DRIFT-SENSE on YOUR OWN images (not the synthetic dataset) and see
the result drawn directly on the image, so you can visually verify
whether the detected location is actually correct.

Usage:
    python src/detect_and_visualize.py \
        --reference path/to/reference_patch.png \
        --search path/to/full_search_image.png \
        --out results/my_test_result.png \
        --weights weights/driftsense_cnn.pt

If you don't have a separate reference patch yet, you can also crop
one interactively from the search image:
    python src/detect_and_visualize.py --search path/to/image.png --crop_reference
"""

import argparse
import os
import sys

import cv2
import numpy as np

from drift_sense_pipeline import DriftSenseDetector


def draw_result(search_bgr, x, y, confidence, reliable, reason, ref_size=64):
    out = search_bgr.copy()
    color = (0, 200, 0) if reliable else (0, 0, 255)  # green if reliable, red if not
    x, y = int(round(x)), int(round(y))

    # crosshair
    cv2.drawMarker(out, (x, y), color, markerType=cv2.MARKER_CROSS,
                    markerSize=30, thickness=2)
    # box showing approx reference-patch footprint at the predicted location
    half = ref_size // 2
    cv2.rectangle(out, (x - half, y - half), (x + half, y + half), color, 2)

    label = f"({x},{y})  conf={confidence:.2f}"
    cv2.putText(out, label, (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)

    banner = "RELIABLE MATCH" if reliable else f"LOW CONFIDENCE: {reason}"
    cv2.putText(out, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    return out


def crop_reference_interactively(search_gray):
    print("Select the region you want to use as the reference patch, "
          "then press ENTER or SPACE. Press 'c' to cancel.")
    r = cv2.selectROI("Select reference (ENTER to confirm)", search_gray, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = r
    if w == 0 or h == 0:
        print("No region selected, exiting.")
        sys.exit(1)
    return search_gray[y:y + h, x:x + w]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default=None,
                         help="path to a reference patch image. If omitted, use --crop_reference.")
    parser.add_argument("--search", type=str, required=True,
                         help="path to the full search image to detect within")
    parser.add_argument("--out", type=str, default="results/visual_detection.png")
    parser.add_argument("--weights", type=str, default=None,
                         help="path to trained CNN weights (.pt). Omit to use untrained CNN.")
    parser.add_argument("--crop_reference", action="store_true",
                         help="interactively crop the reference patch from the search image "
                              "(requires a display; won't work over SSH without X forwarding)")
    args = parser.parse_args()

    search_bgr = cv2.imread(args.search, cv2.IMREAD_COLOR)
    if search_bgr is None:
        print(f"ERROR: could not read search image at {args.search}")
        sys.exit(1)
    search_gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)

    if args.crop_reference:
        reference = crop_reference_interactively(search_gray)
    elif args.reference:
        reference = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
        if reference is None:
            print(f"ERROR: could not read reference image at {args.reference}")
            sys.exit(1)
    else:
        print("ERROR: provide --reference PATH or use --crop_reference")
        sys.exit(1)

    print(f"Reference patch size: {reference.shape}")
    print(f"Search image size: {search_gray.shape}")

    detector = DriftSenseDetector(cnn_weights_path=args.weights)
    result = detector.locate(reference, search_gray)

    print("\n=== DETECTION RESULT ===")
    print(f"Predicted location : ({result.x:.2f}, {result.y:.2f})")
    print(f"Confidence          : {result.confidence:.3f}")
    print(f"Reliable            : {result.reliable}  ({result.reason})")
    print(f"Inference time      : {result.inference_time_s * 1000:.1f} ms")
    print(f"Debug               : {result.debug}")

    annotated = draw_result(search_bgr, result.x, result.y, result.confidence,
                             result.reliable, result.reason, ref_size=reference.shape[0])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, annotated)
    print(f"\nAnnotated image saved to: {args.out}")
    print("Open that file and check: does the crosshair land on the same "
          "content as your reference patch?")


if __name__ == "__main__":
    main()
