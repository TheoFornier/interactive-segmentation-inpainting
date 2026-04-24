"""Fonctions de préparation d'affichage pour les masques."""

from __future__ import annotations

import cv2
import numpy as np


def to_color_mask(mask_binary: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    canvas = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8)
    canvas[mask_binary > 0] = color
    return canvas


def to_heatmap(mask_gray: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(mask_gray.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
