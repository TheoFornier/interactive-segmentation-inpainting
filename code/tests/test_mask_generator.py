"""Tests de conversion en masque binaire."""

import cv2
import numpy as np

from src.segmentation.mask_generator import grabcut_to_binary


def test_grabcut_to_binary_maps_foreground_and_probable_foreground() -> None:
    mask = np.array([
        [cv2.GC_BGD, cv2.GC_PR_BGD],
        [cv2.GC_FGD, cv2.GC_PR_FGD],
    ], dtype=np.uint8)
    binary = grabcut_to_binary(mask)
    assert binary[0, 0] == 0
    assert binary[0, 1] == 0
    assert binary[1, 0] == 255
    assert binary[1, 1] == 255
