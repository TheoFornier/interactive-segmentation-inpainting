"""Post-traitements généraux sur les masques."""

from __future__ import annotations

import cv2
import numpy as np


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def smooth_binary_mask(
    mask: np.ndarray,
    open_kernel_size: int = 3,
    close_kernel_size: int = 5,
) -> np.ndarray:
    """Nettoie un masque binaire avec ouverture puis fermeture."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    open_kernel = np.ones((_odd(open_kernel_size), _odd(open_kernel_size)), np.uint8)
    close_kernel = np.ones((_odd(close_kernel_size), _odd(close_kernel_size)), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    return closed


def keep_largest_component(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Conserve uniquement la plus grande composante connexe significative."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    best_label = 0
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= max(1, min_area) and area > best_area:
            best_area = area
            best_label = label

    if best_label == 0:
        return binary

    result = np.zeros_like(binary)
    result[labels == best_label] = 255
    return result


def keep_component_touching_foreground(
    mask: np.ndarray,
    foreground_seeds: np.ndarray | None,
    min_area: int = 0,
) -> np.ndarray:
    """Conserve la composante connexe la plus cohérente avec les seeds FG.

    Si aucun seed FG n'est fourni, retombe sur la plus grande composante.
    """
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if foreground_seeds is None or not np.any(foreground_seeds > 0):
        return keep_largest_component(binary, min_area=min_area)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    seed_binary = np.where(foreground_seeds > 0, 1, 0).astype(np.uint8)
    best_label = 0
    best_score = -1.0
    best_area = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(1, min_area):
            continue

        component = labels == label
        overlap = int(seed_binary[component].sum())
        if overlap == 0:
            continue

        score = overlap / float(area)
        if score > best_score or (score == best_score and area > best_area):
            best_label = label
            best_score = score
            best_area = area

    if best_label == 0:
        return keep_largest_component(binary, min_area=min_area)

    result = np.zeros_like(binary)
    result[labels == best_label] = 255
    return result


def enforce_seeds(
    mask: np.ndarray,
    foreground_seeds: np.ndarray | None = None,
    background_seeds: np.ndarray | None = None,
) -> np.ndarray:
    """Force le respect strict des annotations utilisateur dans le masque final."""
    result = np.where(mask > 0, 255, 0).astype(np.uint8)
    if foreground_seeds is not None:
        result[foreground_seeds > 0] = 255
    if background_seeds is not None:
        result[background_seeds > 0] = 0
    return result


def refine_binary_mask(
    mask: np.ndarray,
    open_kernel_size: int = 3,
    close_kernel_size: int = 5,
    dilate_iterations: int = 0,
    keep_largest: bool = True,
    min_component_area: int = 0,
    foreground_seeds: np.ndarray | None = None,
    background_seeds: np.ndarray | None = None,
    keep_fg_connected_component: bool = False,
) -> np.ndarray:
    """Pipeline de raffinage simple et robuste pour masque binaire."""
    refined = smooth_binary_mask(mask, open_kernel_size, close_kernel_size)
    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        refined = cv2.dilate(refined, kernel, iterations=int(dilate_iterations))

    if keep_fg_connected_component:
        refined = keep_component_touching_foreground(refined, foreground_seeds, min_component_area)
    elif keep_largest:
        refined = keep_largest_component(refined, min_component_area)

    refined = enforce_seeds(refined, foreground_seeds, background_seeds)
    return np.where(refined > 0, 255, 0).astype(np.uint8)


def feather_mask(mask: np.ndarray, blur_size: int = 9) -> np.ndarray:
    """Produit un alpha flou [0,1] à partir d'un masque binaire."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    blur_size = _odd(blur_size)
    blurred = cv2.GaussianBlur(binary, (blur_size, blur_size), 0)
    return blurred.astype(np.float32) / 255.0
