"""Utilitaires pour l'inpainting."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

import config

Rect = Tuple[int, int, int, int]


def build_hole_mask_from_object_mask(object_mask: np.ndarray, dilation: int | None = None) -> np.ndarray:
    binary = np.where(object_mask > 0, 255, 0).astype(np.uint8)
    dilation = config.INPAINT_MASK_DILATION if dilation is None else max(0, int(dilation))
    if dilation > 0:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=dilation)
    return binary


def mask_bbox(mask: np.ndarray) -> Rect:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def expand_rect(rect: Rect, shape: tuple[int, int], margin_ratio: float, min_padding: int) -> Rect:
    h, w = shape[:2]
    x0, y0, x1, y1 = rect
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad_x = max(min_padding, int(round(bw * margin_ratio)))
    pad_y = max(min_padding, int(round(bh * margin_ratio)))
    ex0 = max(0, x0 - pad_x)
    ey0 = max(0, y0 - pad_y)
    ex1 = min(w, x1 + pad_x)
    ey1 = min(h, y1 + pad_y)
    return (ex0, ey0, ex1, ey1)


def crop_to_rect(image: np.ndarray, rect: Rect) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return image[y0:y1, x0:x1].copy()


def compute_local_inpaint_roi(hole_mask: np.ndarray) -> Rect:
    bbox = mask_bbox(hole_mask)
    return expand_rect(
        bbox,
        hole_mask.shape,
        margin_ratio=config.INPAINT_ROI_MARGIN_RATIO,
        min_padding=config.INPAINT_MIN_ROI_PADDING,
    )
