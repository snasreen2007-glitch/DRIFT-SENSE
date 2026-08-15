"""
drift_sense_pipeline.py
==========================
Wires together every module into the exact stage sequence from
Document Section 3 / the flow diagram:

  Reference Image -> CNN Feature Extractor -> Reference Feature Embedding
  Search Image -> Image Pyramid -> Deep Feature Maps -> Coarse Matching
  -> Top-K Candidates -> Fine Registration -> Subpixel Localization
  -> Confidence Estimation -> Navigation Error Recovery
  -> Corrected Coordinate (x*, y*)

This is the single entry point ("detector") the rest of the project
(evaluation, comparison, GUI) should call.
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from cnn_feature_extractor import CNNFeatureExtractor
from image_pyramid import build_pyramid
from coarse_matching import topk_candidates
from classical_coarse_matching import classical_topk_candidates
from fine_registration import fine_register
from subpixel import subpixel_refine, pixel_to_physical
from confidence import compute_confidence


@dataclass
class DriftSenseResult:
    x: float
    y: float
    confidence: float
    reliable: bool
    reason: str
    inference_time_s: float
    physical_dx: float = 0.0
    physical_dy: float = 0.0
    debug: dict = field(default_factory=dict)


class DriftSenseDetector:
    def __init__(self, backbone: str = "driftsense", pyramid_scales=(0.85, 0.92, 1.0, 1.08, 1.15),
                 top_k: int = 5, s_x: float = 5.0, s_y: float = 5.0, cnn_weights_path: str = None):
        """s_x / s_y: calibrated nanometers (or your chosen physical unit)
        per pixel -- REQUIRED before any physical/nanometer claim (Section 11).
        cnn_weights_path: path to a .pt file saved by train_cnn.py. If None,
        the CNN branch runs on random (untrained) weights."""
        self.extractor = CNNFeatureExtractor(backbone=backbone, weights_path=cnn_weights_path)
        self.pyramid_scales = pyramid_scales
        self.top_k = top_k
        self.s_x = s_x
        self.s_y = s_y

    def locate(self, reference: np.ndarray, search: np.ndarray) -> DriftSenseResult:
        t0 = time.time()

        # ---- Section 3.1: reference embedding ----
        ref_embedding = self.extractor.extract_embedding(reference)

        # ---- Section 3.2 + 3.3: pyramid + deep feature maps ----
        pyramid = build_pyramid(search, scales=self.pyramid_scales)
        pyramid_fmaps = {s: self.extractor.extract_map(img) for s, img in pyramid.items()}

        # ---- Section 3.4 + 3.5: coarse matching + Top-K candidates ----
        # Hybrid architecture (Section 5): merge CNN-branch candidates with
        # classical multi-scale/rotation NCC candidates into one pool.
        cnn_candidates = topk_candidates(ref_embedding, pyramid_fmaps,
                                          stride=self.extractor.stride, k=self.top_k)
        classical_candidates = classical_topk_candidates(search, reference,
                                                           scales=self.pyramid_scales,
                                                           k=self.top_k)
        candidates = cnn_candidates + classical_candidates
        if not candidates:
            return DriftSenseResult(x=-1, y=-1, confidence=0.0, reliable=False,
                                     reason="no candidates found", inference_time_s=time.time() - t0)

        # ---- Section 3.6: fine registration on every candidate ----
        fine_results = []
        for c in candidates:
            fm = fine_register(search, reference, center_xy=(c.x, c.y), candidate_scale=c.scale)
            fine_results.append((c, fm))
        fine_results.sort(key=lambda t: t[1].ncc_score, reverse=True)

        best_candidate, best_fine = fine_results[0]
        second_best_ncc = fine_results[1][1].ncc_score if len(fine_results) > 1 else -1.0

        # ---- Section 3.7: subpixel localization ----
        sub_x, sub_y, dx, dy = subpixel_refine(best_fine.x, best_fine.y, best_fine.response_patch)
        phys_dx, phys_dy = pixel_to_physical(dx, dy, self.s_x, self.s_y)

        # ---- Section 3.8: confidence-aware output ----
        conf = compute_confidence(best_coarse_score=best_candidate.score,
                                   best_fine_ncc=best_fine.ncc_score,
                                   second_best_fine_ncc=second_best_ncc)

        elapsed = time.time() - t0
        return DriftSenseResult(
            x=sub_x, y=sub_y, confidence=conf.confidence, reliable=conf.reliable,
            reason=conf.reason, inference_time_s=elapsed,
            physical_dx=phys_dx, physical_dy=phys_dy,
            debug={
                "num_candidates": len(candidates),
                "best_coarse_score": best_candidate.score,
                "best_fine_ncc": best_fine.ncc_score,
                "second_best_fine_ncc": second_best_ncc,
                "rotation_deg": best_fine.angle,
                "scale": best_candidate.scale,
            },
        )


if __name__ == "__main__":
    import json
    reference = cv2.imread("data/reference/reference_000.png", cv2.IMREAD_GRAYSCALE)
    search = cv2.imread("data/test/test_000.png", cv2.IMREAD_GRAYSCALE)
    with open("data/test/ground_truth_000.json") as f:
        gt = json.load(f)["ground_truth"]

    detector = DriftSenseDetector()
    result = detector.locate(reference, search)

    err = ((result.x - gt["x"]) ** 2 + (result.y - gt["y"]) ** 2) ** 0.5
    print(f"Predicted: ({result.x:.2f}, {result.y:.2f})   Ground truth: ({gt['x']:.2f}, {gt['y']:.2f})")
    print(f"Euclidean error: {err:.2f} px   Confidence: {result.confidence:.3f} ({result.reason})")
    print(f"Inference time: {result.inference_time_s*1000:.1f} ms")
    print("Debug:", result.debug)
