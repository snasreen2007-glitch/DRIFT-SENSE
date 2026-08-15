"""
fine_registration.py
======================
Implements Document Section 3.6 (Fine Registration).

For each coarse candidate region, this module refines the match at
FULL image resolution using classical registration -- this is the
"classical computer vision" half of the hybrid architecture
(Section 5). Deep features got us to the right neighborhood; pixel-
accurate normalized cross-correlation gets us the precise location.

We search over a small set of rotation angles (since Section 3.6 says
the transform T(.) can include rotation) and, for each angle, run
cv2.matchTemplate with TM_CCOEFF_NORMED to find the best translation.
The angle+location with the highest NCC score wins.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FineMatch:
    x: float                 # refined top-left-relative center x, in original image coords
    y: float
    angle: float              # best rotation angle (degrees)
    ncc_score: float          # normalized cross-correlation peak value
    response_patch: np.ndarray  # small NCC response map around the peak (for subpixel step)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)


def fine_register(search_image: np.ndarray, reference_patch: np.ndarray,
                   center_xy, candidate_scale: float = 1.0, window_radius: int = 40,
                   angles=(-4, -3, -2, -1, 0, 1, 2, 3, 4)) -> FineMatch:
    """search_image: full-resolution grayscale search image.
    reference_patch: the reference template at ORIGINAL scale (not resized).
    center_xy: (x, y) coarse candidate location in search_image coords.
    candidate_scale: the pyramid scale the candidate came from -- the
      reference is resized by this factor before matching, since a
      candidate found at pyramid scale s implies the target appears
      ~s times the reference's original size in the search image.
    window_radius: how far around the coarse guess to crop before matching.
    """
    cx, cy = center_xy
    h, w = search_image.shape[:2]

    if candidate_scale != 1.0:
        new_h = max(8, int(reference_patch.shape[0] * candidate_scale))
        new_w = max(8, int(reference_patch.shape[1] * candidate_scale))
        interp = cv2.INTER_AREA if candidate_scale < 1 else cv2.INTER_LINEAR
        reference_patch = cv2.resize(reference_patch, (new_w, new_h), interpolation=interp)

    rh, rw = reference_patch.shape[:2]

    half = window_radius + max(rh, rw) // 2
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    crop = search_image[y0:y1, x0:x1]

    if crop.shape[0] < rh or crop.shape[1] < rw:
        # coarse candidate too close to the border to fit the template
        return FineMatch(x=cx, y=cy, angle=0.0, ncc_score=-1.0,
                          response_patch=np.zeros((3, 3), dtype=np.float32))

    best = None
    for angle in angles:
        rotated_ref = _rotate(reference_patch, angle)
        result = cv2.matchTemplate(crop, rotated_ref, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best[0]:
            best = (max_val, max_loc, angle, result)

    max_val, max_loc, angle, result = best
    mx, my = max_loc  # top-left of best match within `crop`, in crop coords
    center_x_in_crop = mx + rw / 2
    center_y_in_crop = my + rh / 2

    refined_x = x0 + center_x_in_crop
    refined_y = y0 + center_y_in_crop

    # small window of the NCC response around the peak, used for subpixel refinement
    py, px = my, mx
    r0y, r1y = max(0, py - 1), min(result.shape[0], py + 2)
    r0x, r1x = max(0, px - 1), min(result.shape[1], px + 2)
    response_patch = result[r0y:r1y, r0x:r1x].astype(np.float32)

    return FineMatch(x=refined_x, y=refined_y, angle=angle,
                      ncc_score=float(max_val), response_patch=response_patch)


if __name__ == "__main__":
    search = (np.random.rand(400, 400) * 255).astype(np.uint8)
    ref = search[150:190, 150:190].copy()
    match = fine_register(search, ref, center_xy=(172, 172))
    print(match.x, match.y, match.angle, match.ncc_score)
