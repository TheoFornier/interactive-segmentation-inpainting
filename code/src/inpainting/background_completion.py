"""Fonctions de composition et reconstruction du fond."""

from __future__ import annotations

from typing import Tuple

import numpy as np

import config
from src.core.postprocessing import feather_mask

Rect = Tuple[int, int, int, int]


def merge_inpainted_region(
    original_bgr: np.ndarray,
    inpainted_roi_bgr: np.ndarray,
    hole_mask_roi: np.ndarray,
    roi_rect: Rect,
) -> np.ndarray:
    x0, y0, x1, y1 = roi_rect
    merged = original_bgr.copy()

    roi_original = merged[y0:y1, x0:x1].astype(np.float32)
    h, w = roi_original.shape[:2]
    # Resize inpainted ROI if needed
    roi_inpainted = inpainted_roi_bgr.astype(np.float32)
    if roi_inpainted.shape[:2] != (h, w):
        import cv2
        roi_inpainted = cv2.resize(roi_inpainted, (w, h), interpolation=cv2.INTER_LINEAR)
        if roi_inpainted.ndim == 2:
            roi_inpainted = roi_inpainted[..., None]
    # Resize alpha if needed
    alpha = feather_mask(hole_mask_roi, blur_size=config.INPAINT_FEATHER_BLUR_SIZE)[..., None]
    if alpha.shape[:2] != (h, w):
        import cv2
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    # Ensure alpha has 3 channels if needed
    if alpha.shape[2] == 1 and roi_original.shape[2] == 3:
        alpha = np.repeat(alpha, 3, axis=2)
    roi_merged = roi_original * (1.0 - alpha) + roi_inpainted * alpha
    merged[y0:y1, x0:x1] = np.clip(roi_merged, 0, 255).astype(np.uint8)
    return merged
