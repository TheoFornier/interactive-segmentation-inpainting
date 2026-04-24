"""Outils de génération de visuels pour analyse qualitative."""

from __future__ import annotations

import numpy as np

from src.visualization.comparison_plots import labeled_strip


def build_segmentation_board(
    original_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    object_bgr: np.ndarray,
    inpainted_bgr: np.ndarray | None = None,
) -> np.ndarray:
    items = [
        ("Original", original_bgr),
        ("Segmentation", overlay_bgr),
        ("Objet extrait", object_bgr),
    ]
    if inpainted_bgr is not None:
        items.append(("Fond reconstruit", inpainted_bgr))
    return labeled_strip(items)
