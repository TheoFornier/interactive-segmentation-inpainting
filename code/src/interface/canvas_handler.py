"""Canvas interactif basé sur QGraphicsView."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView

import config
from src.interface.events import InteractionMode
from src.interface.image_viewer import numpy_bgr_to_qpixmap
from src.segmentation.rectangle_init import normalize_rectangle
from src.segmentation.scribble_init import add_brush_line, add_brush_stroke
from src.segmentation.seeds_manager import UserAnnotations
from src.visualization.overlays import overlay_mask


class ImageCanvas(QGraphicsView):
    annotations_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)

        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)

        self._rect_item = QGraphicsRectItem()
        pen = QPen(Qt.yellow)
        pen.setWidth(2)
        self._rect_item.setPen(pen)
        self._rect_item.setVisible(False)
        self.scene().addItem(self._rect_item)

        self.mode = InteractionMode.NONE
        self.brush_radius = config.SCRIBBLE_RADIUS
        self.base_image_bgr: Optional[np.ndarray] = None
        self.display_image_bgr: Optional[np.ndarray] = None
        self.annotations = UserAnnotations()
        self._drawing = False
        self._start_point: Optional[tuple[int, int]] = None
        self._last_brush_point: Optional[tuple[int, int]] = None

    def load_image(self, image_bgr: np.ndarray) -> None:
        self.base_image_bgr = image_bgr.copy()
        h, w = image_bgr.shape[:2]
        self.annotations = UserAnnotations(
            rectangle=None,
            foreground_mask=np.zeros((h, w), dtype=np.uint8),
            background_mask=np.zeros((h, w), dtype=np.uint8),
        )
        self._rect_item.setVisible(False)
        self.show_annotations()

    def set_mode(self, mode: InteractionMode) -> None:
        self.mode = mode

    def set_brush_radius(self, radius: int) -> None:
        self.brush_radius = max(1, int(radius))

    def clear_annotations(self) -> None:
        if self.base_image_bgr is None:
            return
        h, w = self.base_image_bgr.shape[:2]
        self.annotations = UserAnnotations(
            rectangle=None,
            foreground_mask=np.zeros((h, w), dtype=np.uint8),
            background_mask=np.zeros((h, w), dtype=np.uint8),
        )
        self._rect_item.setVisible(False)
        self.show_annotations()
        self.annotations_changed.emit()

    def get_annotations(self) -> UserAnnotations:
        return self.annotations

    def show_original(self) -> None:
        if self.base_image_bgr is not None:
            self._set_display_image(self.base_image_bgr)

    def show_annotations(self) -> None:
        if self.base_image_bgr is None:
            return
        canvas = self.base_image_bgr.copy()
        if self.annotations.foreground_mask is not None:
            canvas[self.annotations.foreground_mask > 0] = config.FOREGROUND_COLOR
        if self.annotations.background_mask is not None:
            canvas[self.annotations.background_mask > 0] = config.BACKGROUND_COLOR
        self._set_display_image(canvas)
        if self.annotations.rectangle is not None:
            x, y, w, h = normalize_rectangle(self.annotations.rectangle)
            self._rect_item.setRect(QRectF(x, y, w, h))
            self._rect_item.setVisible(True)
        else:
            self._rect_item.setVisible(False)

    def show_mask_overlay(self, mask_binary: np.ndarray | None) -> None:
        if self.base_image_bgr is None:
            return
        if mask_binary is None:
            self.show_annotations()
            return
        over = overlay_mask(self.base_image_bgr, mask_binary, color=config.FOREGROUND_COLOR, alpha=config.MASK_OVERLAY_ALPHA)
        self._set_display_image(over)

    def show_image(self, image_bgr: np.ndarray | None) -> None:
        if image_bgr is not None:
            self._set_display_image(image_bgr)

    def _set_display_image(self, image_bgr: np.ndarray) -> None:
        self.display_image_bgr = image_bgr.copy()
        pixmap = numpy_bgr_to_qpixmap(image_bgr)
        self._pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.base_image_bgr is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def _to_image_coords(self, event) -> tuple[int, int] | None:
        if self.base_image_bgr is None:
            return None
        point = self.mapToScene(event.position().toPoint())
        x = int(round(point.x()))
        y = int(round(point.y()))
        h, w = self.base_image_bgr.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return x, y
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        coords = self._to_image_coords(event)
        if coords is None or self.base_image_bgr is None:
            return super().mousePressEvent(event)

        self._drawing = True
        self._start_point = coords
        self._last_brush_point = coords

        if self.mode == InteractionMode.RECTANGLE:
            self.annotations.rectangle = (coords[0], coords[1], 1, 1)
            self.show_annotations()
        elif self.mode == InteractionMode.FOREGROUND:
            start = self._last_brush_point or coords
            add_brush_line(self.annotations.foreground_mask, start, coords, self.brush_radius, 255)
            add_brush_line(self.annotations.background_mask, start, coords, self.brush_radius, 0)
            self._last_brush_point = coords
            self.show_annotations()
            self.annotations_changed.emit()
        elif self.mode == InteractionMode.BACKGROUND:
            start = self._last_brush_point or coords
            add_brush_line(self.annotations.background_mask, start, coords, self.brush_radius, 255)
            add_brush_line(self.annotations.foreground_mask, start, coords, self.brush_radius, 0)
            self._last_brush_point = coords
            self.show_annotations()
            self.annotations_changed.emit()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        coords = self._to_image_coords(event)
        if not self._drawing or coords is None or self.base_image_bgr is None:
            return super().mouseMoveEvent(event)

        if self.mode == InteractionMode.RECTANGLE and self._start_point is not None:
            x0, y0 = self._start_point
            self.annotations.rectangle = (x0, y0, coords[0] - x0, coords[1] - y0)
            self.show_annotations()
            self.annotations_changed.emit()
        elif self.mode == InteractionMode.FOREGROUND:
            start = self._last_brush_point or coords
            add_brush_line(self.annotations.foreground_mask, start, coords, self.brush_radius, 255)
            add_brush_line(self.annotations.background_mask, start, coords, self.brush_radius, 0)
            self._last_brush_point = coords
            self.show_annotations()
            self.annotations_changed.emit()
        elif self.mode == InteractionMode.BACKGROUND:
            start = self._last_brush_point or coords
            add_brush_line(self.annotations.background_mask, start, coords, self.brush_radius, 255)
            add_brush_line(self.annotations.foreground_mask, start, coords, self.brush_radius, 0)
            self._last_brush_point = coords
            self.show_annotations()
            self.annotations_changed.emit()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drawing = False
        self._start_point = None
        self._last_brush_point = None
        self.annotations_changed.emit()
        super().mouseReleaseEvent(event)
