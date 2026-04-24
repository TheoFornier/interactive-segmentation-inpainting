"""Assemblage d'images comparatives."""

from __future__ import annotations

import numpy as np

from src.visualization.comparison_plots import labeled_strip


def side_by_side(*images_bgr: np.ndarray) -> np.ndarray:
    titles = [f"Vue {idx + 1}" for idx, _ in enumerate(images_bgr)]
    return labeled_strip(list(zip(titles, images_bgr)))
