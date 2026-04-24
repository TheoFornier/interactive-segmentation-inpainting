"""Structures de données pour les annotations utilisateur."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class UserAnnotations:
    rectangle: Optional[tuple[int, int, int, int]] = None
    foreground_mask: np.ndarray | None = None
    background_mask: np.ndarray | None = None

    def ensure_masks(self, shape: tuple[int, int]) -> None:
        h, w = shape
        if self.foreground_mask is None:
            self.foreground_mask = np.zeros((h, w), dtype=np.uint8)
        if self.background_mask is None:
            self.background_mask = np.zeros((h, w), dtype=np.uint8)

    def has_rectangle(self) -> bool:
        return self.rectangle is not None

    def has_foreground(self) -> bool:
        return self.foreground_mask is not None and bool(np.any(self.foreground_mask > 0))

    def has_background(self) -> bool:
        return self.background_mask is not None and bool(np.any(self.background_mask > 0))

    def foreground_points(self) -> np.ndarray:
        if self.foreground_mask is None:
            return np.empty((0, 2), dtype=np.int32)
        ys, xs = np.where(self.foreground_mask > 0)
        return np.column_stack((xs, ys)).astype(np.int32)

    def background_points(self) -> np.ndarray:
        if self.background_mask is None:
            return np.empty((0, 2), dtype=np.int32)
        ys, xs = np.where(self.background_mask > 0)
        return np.column_stack((xs, ys)).astype(np.int32)
