"""Affinage des masques de segmentation."""

from __future__ import annotations

import numpy as np

import config
from src.core.postprocessing import refine_binary_mask


def refine_mask(
    mask: np.ndarray,
    foreground_seeds: np.ndarray | None = None,
    background_seeds: np.ndarray | None = None,
) -> np.ndarray:
    return refine_binary_mask(
        mask,
        open_kernel_size=config.MASK_OPEN_KERNEL,
        close_kernel_size=config.MASK_CLOSE_KERNEL,
        dilate_iterations=config.MASK_DILATE_ITERATIONS,
        keep_largest=config.MASK_KEEP_LARGEST_COMPONENT,
        min_component_area=config.MASK_MIN_COMPONENT_AREA,
        foreground_seeds=foreground_seeds,
        background_seeds=background_seeds,
        keep_fg_connected_component=True,
    )
