"""Utilitaires d'overlay pour visualiser les masques."""

from __future__ import annotations

import cv2
import numpy as np


def overlay_mask(
    image_bgr: np.ndarray,
    mask_binary: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[mask_binary > 0] = color
    return cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)


def draw_mask_contours(
    image_bgr: np.ndarray,
    mask_binary: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    result = image_bgr.copy()
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, thickness)
    return result
