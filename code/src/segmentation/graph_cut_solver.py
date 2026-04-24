"""Solveur GrabCut / graph cut."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.segmentation.graph_builder import validate_grabcut_mask


def run_grabcut(
    image_bgr: np.ndarray,
    init_mask: np.ndarray,
    rectangle: Optional[tuple[int, int, int, int]],
    iterations: int = 5,
) -> np.ndarray:
    """Exécute GrabCut à partir d'un masque initial et/ou d'un rectangle."""
    mask = init_mask.copy()
    validate_grabcut_mask(mask)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    if rectangle is not None:
        cv2.grabCut(
            image_bgr,
            mask,
            rectangle,
            bg_model,
            fg_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )

    cv2.grabCut(
        image_bgr,
        mask,
        None,
        bg_model,
        fg_model,
        max(1, iterations),
        cv2.GC_INIT_WITH_MASK,
    )
    return mask
