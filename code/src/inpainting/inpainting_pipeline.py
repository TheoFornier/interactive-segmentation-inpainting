"""Pipeline de remplissage de l'arrière-plan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from src.inpainting.background_completion import merge_inpainted_region
from src.inpainting.fill_utils import (
    build_hole_mask_from_object_mask,
    compute_local_inpaint_roi,
    crop_to_rect,
)
from src.inpainting.generative_fill import GenerativeInpainter


@dataclass
class InpaintingResult:
    hole_mask: np.ndarray
    inpainted_bgr: np.ndarray
    completed_background_bgr: np.ndarray
    backend_name: str
    roi_rect: tuple[int, int, int, int]
    roi_shape: tuple[int, int]


class InpaintingPipeline:
    def __init__(self) -> None:
        self.inpainter = GenerativeInpainter()

    def run(self, image_bgr: np.ndarray, object_mask: np.ndarray) -> InpaintingResult:
        hole_mask = build_hole_mask_from_object_mask(object_mask)
        roi_rect = compute_local_inpaint_roi(hole_mask)
        roi_image = crop_to_rect(image_bgr, roi_rect)
        roi_hole_mask = crop_to_rect(hole_mask, roi_rect)

        inpainted_roi = self.inpainter.inpaint(roi_image, roi_hole_mask)
        merged = merge_inpainted_region(image_bgr, inpainted_roi, roi_hole_mask, roi_rect)

        if config.INPAINT_DEBUG_LOG:
            x0, y0, x1, y1 = roi_rect
            print(
                f"[Inpainting] backend={self.inpainter.backend_name} "
                f"roi=({x0},{y0})-({x1},{y1}) size={x1-x0}x{y1-y0} "
                f"mask_pixels={(roi_hole_mask > 0).sum()}"
            )

        return InpaintingResult(
            hole_mask=hole_mask,
            inpainted_bgr=inpainted_roi,
            completed_background_bgr=merged,
            backend_name=self.inpainter.backend_name,
            roi_rect=roi_rect,
            roi_shape=roi_image.shape[:2],
        )
