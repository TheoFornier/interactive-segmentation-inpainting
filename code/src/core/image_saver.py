"""Sauvegarde des images et masques."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.core.utils import ensure_dir


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Échec de sauvegarde: {path}")
