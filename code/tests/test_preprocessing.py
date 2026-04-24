"""Tests du prétraitement."""

import numpy as np

from src.core.preprocessing import resize_for_display


def test_resize_for_display_reduces_large_image() -> None:
    image = np.zeros((2000, 1000, 3), dtype=np.uint8)
    resized, scale = resize_for_display(image, max_width=500, max_height=500)
    assert resized.shape[0] <= 500
    assert resized.shape[1] <= 500
    assert 0 < scale < 1
