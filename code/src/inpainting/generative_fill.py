"""Remplissage automatique via LaMa (si disponible) avec fallback OpenCV."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

import config


class GenerativeInpainter:
    def __init__(self) -> None:
        self._backend = None
        self.backend_name = "opencv-fallback"
        try:
            from simple_lama_inpainting import SimpleLama  # type: ignore
            self._backend = SimpleLama()
            self.backend_name = "simple-lama-inpainting"
        except Exception:
            self._backend = None

    @property
    def is_lama_available(self) -> bool:
        return self._backend is not None

    def inpaint(self, image_bgr: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
        hole_mask = np.where(hole_mask > 0, 255, 0).astype(np.uint8)
        if self._backend is None:
            method = cv2.INPAINT_TELEA if config.INPAINT_FALLBACK_METHOD.lower() == "telea" else cv2.INPAINT_NS
            return cv2.inpaint(
                image_bgr,
                hole_mask,
                config.INPAINT_FALLBACK_RADIUS,
                method,
            )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        pil_mask = Image.fromarray(hole_mask)
        pil_result = self._backend(pil_img, pil_mask)
        result_rgb = np.array(pil_result)
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
