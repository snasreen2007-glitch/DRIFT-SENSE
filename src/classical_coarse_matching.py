"""
classical_coarse_matching.py
===============================
The "Classical Computer Vision" half of Section 5's hybrid architecture,
feeding the SAME Top-K candidate pool as the CNN branch (coarse_matching.py).

Why this exists alongside the CNN branch
-------------------------------------------
Document Section 5 is explicit that DRIFT-SENSE is NOT meant to depend
entirely on a neural network -- it should combine deep learning with
classical image pyramids / feature matching / candidate selection.
A CNN trained from scratch (or randomly initialized, before you've
collected/trained on real wafer data) will not yet produce reliable
embeddings. Multi-scale, multi-rotation normalized cross-correlation
is a strong classical prior that keeps the system usable *today*, while
the CNN branch is trained/fine-tuned over time. In production you'd
expect the CNN branch to gradually dominate the confidence score as it
is trained on real wafer defect/navigation data.
"""

from dataclasses import dataclass
import cv2
import numpy as np

from coarse_matching import Candidate


def classical_topk_candidates(search: np.ndarray, reference: np.ndarray,
                               scales=(0.85, 0.92, 1.0, 1.08, 1.15),
                               angles=(-4, -2, 0, 2, 4),
                               k: int = 5,
                               min_separation_px: int = 12) -> list:
    all_points = []  # (score, x, y, scale)
    rh, rw = reference.shape[:2]

    for scale in scales:
        new_w, new_h = max(8, int(rw * scale)), max(8, int(rh * scale))
        if new_w >= search.shape[1] or new_h >= search.shape[0]:
            continue
        resized_ref = cv2.resize(reference, (new_w, new_h),
                                  interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        for angle in angles:
            if angle != 0:
                M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
                templ = cv2.warpAffine(resized_ref, M, (new_w, new_h),
                                        borderMode=cv2.BORDER_REPLICATE)
            else:
                templ = resized_ref

            result = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            cx = max_loc[0] + new_w / 2
            cy = max_loc[1] + new_h / 2
            all_points.append((float(max_val), cx, cy, scale))

    all_points.sort(key=lambda t: t[0], reverse=True)

    kept = []
    for score, x, y, scale in all_points:
        if all(((x - c.x) ** 2 + (y - c.y) ** 2) ** 0.5 > min_separation_px for c in kept):
            kept.append(Candidate(x=x, y=y, scale=scale, score=score))
        if len(kept) >= k:
            break
    return kept


if __name__ == "__main__":
    search = (np.random.rand(400, 400) * 255).astype(np.uint8)
    ref = search[150:190, 150:190].copy()
    cands = classical_topk_candidates(search, ref, k=3)
    for c in cands:
        print(c)
