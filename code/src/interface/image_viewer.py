"""Helpers de conversion image/Qt pour l'affichage."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def numpy_bgr_to_qpixmap(image_bgr: np.ndarray) -> QPixmap:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = image_rgb.shape
    bytes_per_line = ch * w
    qimage = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())
