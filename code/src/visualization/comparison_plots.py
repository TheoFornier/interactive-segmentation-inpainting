"""Génération de planches comparatives annotées."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


FONT = cv2.FONT_HERSHEY_SIMPLEX


def _resize_to_height(image_bgr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    new_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(image_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)


def add_title(image_bgr: np.ndarray, title: str, bar_height: int = 32) -> np.ndarray:
    title_bar = np.full((bar_height, image_bgr.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 22), FONT, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    return cv2.vconcat([title_bar, image_bgr])


def labeled_strip(items: Iterable[tuple[str, np.ndarray]]) -> np.ndarray:
    items = list(items)
    if not items:
        raise ValueError("Aucune image à assembler.")
    target_h = min(image.shape[0] for _, image in items)
    panels = []
    for title, image in items:
        resized = _resize_to_height(image, target_h)
        panels.append(add_title(resized, title))
    return cv2.hconcat(panels)
