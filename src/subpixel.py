"""
subpixel.py
=============
Implements Document Section 3.7 (Subpixel Localization):

    P = (x + delta_x, y + delta_y)
    Delta_X = s_x * delta_x,   Delta_Y = s_y * delta_y

We fit a 1D parabola through the three NCC response values around the
integer-pixel peak (one peak-fit along x, one along y). This is the
standard, cheap, and numerically stable way to get sub-pixel accuracy
out of a correlation peak without training an extra network.

`pixel_to_distance` implements the "must be mapped to the physical
wafer coordinate system" note at the end of Section 2, and the
s_x / s_y calibration factors from Section 3.7 -- this is what your
Limitations section (Section 11) says can NOT be skipped if you want
to claim real nanometer precision.
"""

import numpy as np


def _parabolic_peak_1d(y_minus1: float, y0: float, y_plus1: float) -> float:
    """Returns the subpixel offset (in [-0.5, 0.5]) of the true peak
    relative to the center sample, given three samples around it."""
    denom = (y_minus1 - 2 * y0 + y_plus1)
    if abs(denom) < 1e-8:
        return 0.0
    offset = 0.5 * (y_minus1 - y_plus1) / denom
    return float(np.clip(offset, -0.5, 0.5))


def subpixel_refine(refined_x: float, refined_y: float, response_patch: np.ndarray):
    """response_patch: up to 3x3 NCC response values centered on the integer peak
    (as produced by fine_registration.fine_register).

    Returns (x_subpixel, y_subpixel, dx, dy).
    """
    h, w = response_patch.shape
    cy, cx = h // 2, w // 2

    dx = dy = 0.0
    if w >= 3:
        dx = _parabolic_peak_1d(response_patch[cy, cx - 1],
                                  response_patch[cy, cx],
                                  response_patch[cy, cx + 1])
    if h >= 3:
        dy = _parabolic_peak_1d(response_patch[cy - 1, cx],
                                  response_patch[cy, cx],
                                  response_patch[cy + 1, cx])

    return refined_x + dx, refined_y + dy, dx, dy


def pixel_to_physical(dx_px: float, dy_px: float, s_x: float, s_y: float):
    """Convert a pixel-space correction to physical wafer-stage units
    (e.g. nanometers or microns per pixel), using the calibrated
    pixel-to-distance factors s_x, s_y (Section 3.7 / Section 11)."""
    return dx_px * s_x, dy_px * s_y


if __name__ == "__main__":
    resp = np.array([[0.1, 0.2, 0.1],
                      [0.2, 0.9, 0.3],
                      [0.1, 0.25, 0.1]], dtype=np.float32)
    x, y, dx, dy = subpixel_refine(100.0, 100.0, resp)
    print(f"subpixel: x={x:.3f} y={y:.3f} dx={dx:.3f} dy={dy:.3f}")
