"""Métriques simples pour évaluer la segmentation."""

from __future__ import annotations

import numpy as np


def _binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def intersection_over_union(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = _binary(pred_mask)
    gt = _binary(gt_mask)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return 1.0 if union == 0 else inter / union


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = _binary(pred_mask)
    gt = _binary(gt_mask)
    inter = float(np.logical_and(pred, gt).sum())
    total = float(pred.sum() + gt.sum())
    return 1.0 if total == 0 else (2.0 * inter) / total


def precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    pred = _binary(pred_mask)
    gt = _binary(gt_mask)
    tp = float(np.logical_and(pred == 1, gt == 1).sum())
    fp = float(np.logical_and(pred == 1, gt == 0).sum())
    fn = float(np.logical_and(pred == 0, gt == 1).sum())
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    return precision, recall
