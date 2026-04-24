"""Tests du pipeline de remplissage."""

import numpy as np

from src.inpainting.inpainting_pipeline import InpaintingPipeline


def test_inpainting_pipeline_returns_same_shape() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[30:70, 30:70] = (0, 0, 255)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    pipeline = InpaintingPipeline()
    result = pipeline.run(image, mask)

    assert result.completed_background_bgr.shape == image.shape
    assert result.hole_mask.shape == mask.shape
    assert result.completed_background_bgr[50, 50].sum() > 0


def test_inpainting_pipeline_uses_local_roi() -> None:
    image = np.full((200, 300, 3), 180, dtype=np.uint8)
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[80:120, 130:170] = 255

    pipeline = InpaintingPipeline()
    result = pipeline.run(image, mask)

    x0, y0, x1, y1 = result.roi_rect
    assert 0 <= x0 < x1 <= 300
    assert 0 <= y0 < y1 <= 200
    assert (x1 - x0) < 300
    assert (y1 - y0) < 200
