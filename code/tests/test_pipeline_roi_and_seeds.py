"""Tests sur la ROI locale et le respect strict des seeds."""

import numpy as np

from src.segmentation.seeds_manager import UserAnnotations
from src.segmentation.segmentation_pipeline import SegmentationPipeline


def test_small_object_with_tight_fg_bg_scribbles_stays_local() -> None:
    image = np.zeros((220, 220, 3), dtype=np.uint8)
    image[:] = (80, 120, 160)

    # fond texturé synthétique
    for x in range(0, 220, 4):
        image[:, x : x + 2] = (120, 90, 60)

    # petit objet clair
    image[90:130, 100:130] = (220, 230, 40)

    fg = np.zeros((220, 220), dtype=np.uint8)
    bg = np.zeros((220, 220), dtype=np.uint8)
    fg[100:120, 108:122] = 255
    bg[88:132, 92:98] = 255
    bg[88:92, 98:132] = 255
    bg[128:132, 98:132] = 255

    annotations = UserAnnotations(
        rectangle=(92, 88, 44, 44),
        foreground_mask=fg,
        background_mask=bg,
    )
    result = SegmentationPipeline(iterations=2).run(image, annotations)

    ys, xs = np.where(result.refined_mask > 0)
    assert len(xs) > 0
    # le masque ne doit pas fuiter très loin à gauche
    assert xs.min() >= 96
    assert xs.max() <= 136


def test_final_mask_respects_foreground_and_background_seeds() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[35:85, 35:85] = (240, 240, 240)

    fg = np.zeros((120, 120), dtype=np.uint8)
    bg = np.zeros((120, 120), dtype=np.uint8)
    fg[55:65, 55:65] = 255
    bg[40:50, 20:30] = 255

    annotations = UserAnnotations(rectangle=(28, 28, 64, 64), foreground_mask=fg, background_mask=bg)
    result = SegmentationPipeline(iterations=2).run(image, annotations)

    assert np.all(result.refined_mask[fg > 0] == 255)
    assert np.all(result.refined_mask[bg > 0] == 0)
