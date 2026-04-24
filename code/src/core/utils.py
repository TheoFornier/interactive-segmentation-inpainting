"""Fonctions utilitaires communes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Retourne un masque uint8 {0,255}."""
    out = np.where(mask > 0, 255, 0).astype(np.uint8)
    return out
