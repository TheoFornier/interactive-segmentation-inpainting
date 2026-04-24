"""Outils pour l'initialisation par scribbles."""

from __future__ import annotations

import cv2
import numpy as np


def add_brush_stroke(mask: np.ndarray, x: int, y: int, radius: int, value: int) -> None:
    cv2.circle(mask, (int(x), int(y)), int(radius), int(value), thickness=-1)


def add_brush_line(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    radius: int,
    value: int,
) -> None:
    """Trace un trait continu entre deux positions successives du curseur."""
    thickness = max(1, int(radius) * 2)
    cv2.line(
        mask,
        (int(start[0]), int(start[1])),
        (int(end[0]), int(end[1])),
        int(value),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    add_brush_stroke(mask, start[0], start[1], radius, value)
    add_brush_stroke(mask, end[0], end[1], radius, value)
