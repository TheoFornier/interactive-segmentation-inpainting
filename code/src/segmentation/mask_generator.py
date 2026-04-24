"""Conversion des masques GrabCut en masque binaire exploitable."""

from __future__ import annotations

import cv2
import numpy as np


def grabcut_to_binary(mask: np.ndarray) -> np.ndarray:
    return np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype("uint8")
