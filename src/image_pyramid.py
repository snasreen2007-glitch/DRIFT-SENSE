"""
image_pyramid.py
==================
Implements Document Section 3.2 (Search Image Pyramid):

    P_S = { I_S^1, I_S^2, ..., I_S^L }

Each level is a different spatial resolution of the search image, which
makes the coarse-matching stage robust to large scale differences
between the reference patch and the way the target appears in the
search image (thermal drift, stage re-zero, different objective/zoom).
"""

import cv2
import numpy as np


def build_pyramid(image: np.ndarray, scales=(0.5, 0.7, 1.0, 1.4, 2.0)) -> dict:
    """Return {scale_factor: resized_image} for the given scale factors.

    scales < 1.0  -> search image shrunk (target may be relatively larger)
    scales > 1.0  -> search image enlarged (target may be relatively smaller)
    """
    pyramid = {}
    h, w = image.shape[:2]
    for s in scales:
        new_w, new_h = max(8, int(w * s)), max(8, int(h * s))
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        pyramid[s] = resized
    return pyramid


if __name__ == "__main__":
    dummy = (np.random.rand(500, 500) * 255).astype(np.uint8)
    pyr = build_pyramid(dummy)
    for s, im in pyr.items():
        print(f"scale={s} -> shape={im.shape}")
