"""Day / night routing: luminance rule.

night iff the mean BT.601 luma of a 256-px-long-side downsample (cv2 INTER_AREA) is below tau = 84
(tau selected on the validation set).
"""
import cv2
import numpy as np

TAU = 84


def luma(rgb):
    """Mean BT.601 luma of a uint8/float RGB image, 0..255."""
    rgb = rgb.astype(np.float64)
    return float((0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).mean())


def luma256(rgb_u8):
    h, w = rgb_u8.shape[:2]
    s = 256 / max(h, w)
    return luma(cv2.resize(rgb_u8, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA))


def is_night(rgb_u8, tau=TAU):
    return luma256(rgb_u8) < tau
