"""Tests de bout en bout sur une image synthétique simple."""

import numpy as np

from src.segmentation.seeds_manager import UserAnnotations
from src.segmentation.segmentation_pipeline import SegmentationPipeline


def test_segmentation_pipeline_detects_simple_foreground() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[60:140, 60:140] = (255, 255, 255)

    annotations = UserAnnotations(rectangle=(40, 40, 120, 120))
    pipeline = SegmentationPipeline(iterations=2)
    result = pipeline.run(image, annotations)

    foreground_pixels = int((result.refined_mask > 0).sum())
    assert foreground_pixels > 2000
    assert result.extracted_object_bgr.shape == image.shape
