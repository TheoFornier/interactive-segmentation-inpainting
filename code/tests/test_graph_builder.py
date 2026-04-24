"""Tests de validation du masque GrabCut."""

import cv2
import numpy as np
import pytest

from src.segmentation.graph_builder import validate_grabcut_mask


def test_validate_grabcut_mask_accepts_valid_values() -> None:
    mask = np.array([[cv2.GC_BGD, cv2.GC_PR_BGD], [cv2.GC_FGD, cv2.GC_PR_FGD]], dtype=np.uint8)
    validate_grabcut_mask(mask)


def test_validate_grabcut_mask_rejects_invalid_values() -> None:
    mask = np.array([[7]], dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_grabcut_mask(mask)
