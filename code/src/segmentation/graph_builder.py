"""Validation de la représentation de masque pour GrabCut.

GrabCut construit lui-même un graphe interne; ce module vérifie et prépare
les catégories de masque attendues par OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np

VALID_VALUES = {cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD}


def validate_grabcut_mask(mask: np.ndarray) -> None:
    values = set(np.unique(mask).tolist())
    if not values.issubset(VALID_VALUES):
        raise ValueError(f"Valeurs de masque invalides: {sorted(values)}")
