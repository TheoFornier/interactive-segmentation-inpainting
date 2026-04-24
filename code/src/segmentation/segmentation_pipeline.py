"""Pipeline complet de segmentation interactive."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config
from src.segmentation.graph_cut_solver import run_grabcut
from src.segmentation.mask_generator import grabcut_to_binary
from src.segmentation.mask_refinement import refine_mask
from src.segmentation.rectangle_init import normalize_rectangle
from src.segmentation.seeds_manager import UserAnnotations
from src.visualization.overlays import draw_mask_contours, overlay_mask


@dataclass
class SegmentationResult:
    grabcut_mask: np.ndarray
    binary_mask: np.ndarray
    refined_mask: np.ndarray
    overlay_bgr: np.ndarray
    contour_preview_bgr: np.ndarray
    extracted_object_bgr: np.ndarray
    background_removed_bgr: np.ndarray


class SegmentationPipeline:
    def __init__(self, iterations: int = config.GRABCUT_ITERATIONS) -> None:
        self.iterations = iterations

    @staticmethod
    def _expanded_bbox_from_mask(
        mask: np.ndarray,
        image_shape: tuple[int, int],
        margin_ratio: float = 0.9,
    ) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Impossible de déduire une zone objet sans annotation de premier plan.")

        h, w = image_shape
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        bw = max(1, x_max - x_min + 1)
        bh = max(1, y_max - y_min + 1)
        margin_x = max(8, int(round(bw * margin_ratio)))
        margin_y = max(8, int(round(bh * margin_ratio)))

        x = max(0, x_min - margin_x)
        y = max(0, y_min - margin_y)
        x2 = min(w, x_max + margin_x + 1)
        y2 = min(h, y_max + margin_y + 1)
        return x, y, max(1, x2 - x), max(1, y2 - y)

    @staticmethod
    def _expand_rect(
        rect: tuple[int, int, int, int],
        image_shape: tuple[int, int],
        margin_ratio: float,
    ) -> tuple[int, int, int, int]:
        h, w = image_shape
        x, y, rw, rh = rect
        pad_x = max(8, int(round(rw * margin_ratio)))
        pad_y = max(8, int(round(rh * margin_ratio)))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + rw + pad_x)
        y2 = min(h, y + rh + pad_y)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def _crop(array: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = roi
        return array[y : y + h, x : x + w].copy()

    @staticmethod
    def _paste(mask_roi: np.ndarray, image_shape: tuple[int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = roi
        full = np.zeros(image_shape, dtype=mask_roi.dtype)
        full[y : y + h, x : x + w] = mask_roi
        return full

    @staticmethod
    def _apply_center_prior(mask: np.ndarray, prior_margin_ratio: float) -> np.ndarray:
        """Favorise le centre de la ROI comme premier plan probable.

        Cela aide beaucoup quand le rectangle est plus grand que l'objet réel.
        """
        h, w = mask.shape
        mx = max(1, int(round(w * prior_margin_ratio)))
        my = max(1, int(round(h * prior_margin_ratio)))
        if w - 2 * mx <= 2 or h - 2 * my <= 2:
            return mask

        center = np.full_like(mask, cv2.GC_PR_BGD)
        center[my : h - my, mx : w - mx] = cv2.GC_PR_FGD

        # Ne jamais écraser les contraintes fortes déjà posées.
        strong_fg = mask == cv2.GC_FGD
        strong_bg = mask == cv2.GC_BGD
        merged = center
        merged[strong_fg] = cv2.GC_FGD
        merged[strong_bg] = cv2.GC_BGD
        return merged

    @staticmethod
    def _fill_closed_contours(mask: np.ndarray) -> np.ndarray:
        """Si les traits FG forment un contour fermé, remplir l'intérieur.

        Retourne un masque où les zones *encerclées* par les traits sont
        également marquées, ce qui permet à GrabCut de segmenter l'objet
        entier et pas seulement le trait.
        """
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = binary.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            arc = cv2.arcLength(cnt, closed=True)
            if arc > 0 and area > 50:
                # Remplir l'intérieur du contour
                cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
        return filled

    def build_initial_mask(
        self,
        image_bgr: np.ndarray,
        annotations: UserAnnotations,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None, tuple[int, int, int, int]]:
        h, w = image_bgr.shape[:2]
        annotations.ensure_masks((h, w))

        # Remplir l'intérieur des contours fermés tracés au pinceau FG
        if annotations.foreground_mask is not None and np.any(annotations.foreground_mask > 0):
            annotations.foreground_mask = self._fill_closed_contours(annotations.foreground_mask)

        fg = annotations.foreground_mask > 0
        bg = annotations.background_mask > 0
        has_rect = annotations.has_rectangle()
        has_fg = bool(np.any(fg))
        has_bg = bool(np.any(bg))

        if not has_rect and not has_fg:
            raise ValueError(
                "Ajoute soit un rectangle, soit au moins quelques traits de premier plan. "
                "Le rectangle n'est plus obligatoire si tu utilises le pinceau vert."
            )

        if has_rect:
            base_rect = normalize_rectangle(annotations.rectangle)
        else:
            base_rect = self._expanded_bbox_from_mask(annotations.foreground_mask, (h, w))

        roi = self._expand_rect(base_rect, (h, w), margin_ratio=config.ROI_MARGIN_RATIO)
        x_roi, y_roi, w_roi, h_roi = roi

        fg_roi = self._crop(annotations.foreground_mask, roi)
        bg_roi = self._crop(annotations.background_mask, roi)

        mask = np.full((h_roi, w_roi), cv2.GC_BGD, dtype=np.uint8)
        inner_x = max(0, base_rect[0] - x_roi)
        inner_y = max(0, base_rect[1] - y_roi)
        inner_w = min(base_rect[2], w_roi - inner_x)
        inner_h = min(base_rect[3], h_roi - inner_y)
        mask[inner_y : inner_y + inner_h, inner_x : inner_x + inner_w] = cv2.GC_PR_FGD
        mask = self._apply_center_prior(mask, config.CENTER_PRIOR_MARGIN_RATIO)

        if has_fg:
            mask[fg_roi > 0] = cv2.GC_FGD
        if has_bg:
            mask[bg_roi > 0] = cv2.GC_BGD

        rect_for_grabcut = (inner_x, inner_y, inner_w, inner_h) if has_rect else None
        return mask, rect_for_grabcut, roi

    def extract_object(self, image_bgr: np.ndarray, mask_binary: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(image_bgr, image_bgr, mask=mask_binary)

    def remove_object(self, image_bgr: np.ndarray, mask_binary: np.ndarray) -> np.ndarray:
        result = image_bgr.copy()
        result[mask_binary > 0] = 0
        return result

    def run(
        self,
        image_bgr: np.ndarray,
        annotations: UserAnnotations,
    ) -> SegmentationResult:
        init_mask_roi, rect_roi, roi = self.build_initial_mask(image_bgr, annotations)
        image_roi = self._crop(image_bgr, roi)
        fg_roi = self._crop(annotations.foreground_mask, roi)
        bg_roi = self._crop(annotations.background_mask, roi)

        if config.GRABCUT_GAUSSIAN_BLUR > 1:
            blur = int(config.GRABCUT_GAUSSIAN_BLUR)
            blur = blur if blur % 2 == 1 else blur + 1
            image_for_gc = cv2.GaussianBlur(image_roi, (blur, blur), 0)
        else:
            image_for_gc = image_roi

        grabcut_mask_roi = run_grabcut(
            image_bgr=image_for_gc,
            init_mask=init_mask_roi,
            rectangle=rect_roi,
            iterations=self.iterations,
        )
        binary_roi = grabcut_to_binary(grabcut_mask_roi)
        refined_roi = refine_mask(binary_roi, foreground_seeds=fg_roi, background_seeds=bg_roi)

        # Respect strict des seeds, même après post-traitement.
        refined_roi[fg_roi > 0] = 255
        refined_roi[bg_roi > 0] = 0

        h, w = image_bgr.shape[:2]
        binary_mask = self._paste(binary_roi, (h, w), roi)
        refined = self._paste(refined_roi, (h, w), roi)
        grabcut_mask = self._paste(grabcut_mask_roi, (h, w), roi)

        overlay = overlay_mask(image_bgr, refined, color=config.FOREGROUND_COLOR, alpha=config.MASK_OVERLAY_ALPHA)
        contour_preview = draw_mask_contours(
            image_bgr,
            refined,
            color=(0, 255, 255),
            thickness=config.CONTOUR_THICKNESS,
        )
        extracted = self.extract_object(image_bgr, refined)
        removed = self.remove_object(image_bgr, refined)

        return SegmentationResult(
            grabcut_mask=grabcut_mask,
            binary_mask=binary_mask,
            refined_mask=refined,
            overlay_bgr=overlay,
            contour_preview_bgr=contour_preview,
            extracted_object_bgr=extracted,
            background_removed_bgr=removed,
        )
