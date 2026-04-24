"""Tests de segmentation sans rectangle obligatoire."""

import numpy as np
import pytest

from src.segmentation.seeds_manager import UserAnnotations
from src.segmentation.segmentation_pipeline import SegmentationPipeline


def test_segmentation_pipeline_detects_foreground_with_scribbles_only() -> None:
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    image[55:125, 55:125] = (245, 245, 245)

    fg = np.zeros((180, 180), dtype=np.uint8)
    bg = np.zeros((180, 180), dtype=np.uint8)

    fg[75:105, 75:105] = 255
    bg[:20, :] = 255
    bg[-20:, :] = 255
    bg[:, :20] = 255
    bg[:, -20:] = 255

    annotations = UserAnnotations(rectangle=None, foreground_mask=fg, background_mask=bg)
    pipeline = SegmentationPipeline(iterations=2)
    result = pipeline.run(image, annotations)

    foreground_pixels = int((result.refined_mask > 0).sum())
    assert foreground_pixels > 1500
    assert result.extracted_object_bgr.shape == image.shape


def test_build_initial_mask_rejects_missing_object_hint() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    annotations = UserAnnotations(
        rectangle=None,
        foreground_mask=np.zeros((120, 120), dtype=np.uint8),
        background_mask=np.zeros((120, 120), dtype=np.uint8),
    )
    pipeline = SegmentationPipeline(iterations=1)

    with pytest.raises(ValueError):
        pipeline.run(image, annotations)
