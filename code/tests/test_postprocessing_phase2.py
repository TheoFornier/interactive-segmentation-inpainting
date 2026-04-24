"""Tests supplémentaires de la phase 2."""

import numpy as np

from src.core.postprocessing import keep_largest_component, refine_binary_mask
from src.evaluation.metrics import dice_score, intersection_over_union
from src.inpainting.fill_utils import build_hole_mask_from_object_mask


def test_keep_largest_component_removes_small_blob() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    mask[60:65, 60:65] = 255
    result = keep_largest_component(mask)
    assert result[20, 20] == 255
    assert result[62, 62] == 0


def test_build_hole_mask_can_expand_region() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 255
    expanded = build_hole_mask_from_object_mask(mask, dilation=2)
    assert expanded.sum() > mask.sum()


def test_metrics_are_consistent() -> None:
    gt = np.zeros((32, 32), dtype=np.uint8)
    pred = np.zeros((32, 32), dtype=np.uint8)
    gt[8:24, 8:24] = 255
    pred[8:24, 8:24] = 255
    assert intersection_over_union(pred, gt) == 1.0
    assert dice_score(pred, gt) == 1.0


def test_refine_binary_mask_stays_binary() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    refined = refine_binary_mask(mask)
    assert set(np.unique(refined)).issubset({0, 255})
