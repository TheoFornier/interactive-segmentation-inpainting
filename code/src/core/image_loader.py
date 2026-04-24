"""Chargement des images depuis le disque."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {path}")
    return image
