"""Prétraitements d'image."""

from __future__ import annotations

import cv2
import numpy as np


def resize_for_display(
    image_bgr: np.ndarray,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, float]:
    """Redimensionne l'image en conservant le ratio."""
    h, w = image_bgr.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return image_bgr.copy(), 1.0

    resized = cv2.resize(
        image_bgr,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale
