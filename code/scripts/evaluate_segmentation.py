#!/usr/bin/env python3
"""
Evaluation script for segmentation against ground-truth masks.

Usage:
  .venv312/bin/python scripts/evaluate_segmentation.py --data data/eval --out results/eval_seg

Behavior:
 - For each folder in data/eval (e.g. image1/), expects image.png and gt_mask.png
 - If annotation masks exist under data/annotations/<image>/ they will be used when available:
     fg_mask.png, bg_mask.png, rect.json (optional)
 - Otherwise, synthetic reproducible annotations are generated from GT:
     - rectangle: bounding box of GT
     - FG seeds: eroded GT
     - BG seeds: ring around GT
 - Runs the segmentation pipeline in headless mode and computes IoU, Dice, Precision, Recall and pixel ratio.
 - Saves CSV summary and per-case overlays in the output folder.

Notes:
 - This script uses the project's SegmentationPipeline and UserAnnotations APIs.
 - Synthetic annotations are derived from GT (so they are optimistic / privileged). See docs for limitations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import csv

import cv2
import numpy as np

from src.segmentation.segmentation_pipeline import SegmentationPipeline
from src.segmentation.seeds_manager import UserAnnotations


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return np.where(m > 127, 255, 0).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def synthetic_fg_from_gt(gt: np.ndarray) -> np.ndarray:
    # Erode GT to get inner seeds (deterministic)
    h, w = gt.shape[:2]
    k = max(3, int(min(h, w) * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.erode(gt, kernel, iterations=1)
    return np.where(fg > 0, 255, 0).astype(np.uint8)


def synthetic_bg_from_gt(gt: np.ndarray) -> np.ndarray:
    # Create a ring around GT as probable background seeds
    h, w = gt.shape[:2]
    k = max(5, int(min(h, w) * 0.03))
    dil = cv2.dilate(gt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    ring = dil.copy()
    ring[gt > 0] = 0
    return np.where(ring > 0, 255, 0).astype(np.uint8)


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    # pred and gt are binary 0/255
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    tp = int((p & g).sum())
    fp = int((p & (1 - g)).sum())
    fn = int(((1 - p) & g).sum())
    tn = int(((1 - p) & (1 - g)).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ratio = (p.sum() / 255) / (g.sum() / 255) if g.sum() > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "ratio": ratio,
    }


def overlay_pred_gt(image: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # color blend: FP=red, FN=green, TP=yellow, TN=unchanged
    base = image.copy().astype(np.float32)
    result = base.copy()
    mask_pred = (pred > 0)
    mask_gt = (gt > 0)
    tp = mask_pred & mask_gt
    fp = mask_pred & ~mask_gt
    fn = ~mask_pred & mask_gt
    # red for FP
    result[fp] = base[fp] * 0.4 + np.array([0, 0, 200], dtype=np.float32) * 0.6
    # green for FN
    result[fn] = base[fn] * 0.4 + np.array([0, 200, 0], dtype=np.float32) * 0.6
    # yellow for TP
    result[tp] = base[tp] * 0.4 + np.array([0, 220, 220], dtype=np.float32) * 0.6
    return np.clip(result, 0, 255).astype(np.uint8)


def load_annotations_if_exists(ann_folder: Path, shape: tuple[int, int]) -> dict:
    ann = {}
    fg = ann_folder / "fg_mask.png"
    bg = ann_folder / "bg_mask.png"
    rect = ann_folder / "rect.json"
    if fg.exists():
        ann["fg"] = read_mask(fg)
    if bg.exists():
        ann["bg"] = read_mask(bg)
    if rect.exists():
        try:
            j = json.loads(rect.read_text())
            ann["rect"] = (int(j["x"]), int(j["y"]), int(j["w"]), int(j["h"]))
        except Exception:
            pass
    return ann


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/eval", help="Root folder with per-image subfolders containing image.png and gt_mask.png")
    p.add_argument("--out", default="results/eval_seg", help="Output folder for CSV and visuals")
    args = p.parse_args(argv)

    data_root = Path(args.data)
    out_root = Path(args.out)
    ensure_dir(out_root)

    pipeline = SegmentationPipeline()

    rows = []
    for case in sorted([d for d in data_root.iterdir() if d.is_dir()]):
        img_path = case / "image.png"
        gt_path = case / "gt_mask.png"
        if not img_path.exists() or not gt_path.exists():
            print(f"Skipping {case.name}: missing image.png or gt_mask.png")
            continue

        image = read_image(img_path)
        gt = read_mask(gt_path)
        h, w = gt.shape[:2]

        # try to load user annotations if present
        ann_folder = Path("data/annotations") / case.name
        ann_exists = ann_folder.exists()
        ann_data = load_annotations_if_exists(ann_folder, (h, w)) if ann_exists else {}

        # Build modes to evaluate
        modes = ["rectangle", "fg", "bg", "combined"]
        for mode in modes:
            anns = UserAnnotations()
            # fill default masks
            anns.foreground_mask = np.zeros((h, w), dtype=np.uint8)
            anns.background_mask = np.zeros((h, w), dtype=np.uint8)

            if mode == "rectangle":
                if "rect" in ann_data:
                    anns.rectangle = ann_data["rect"]
                else:
                    anns.rectangle = bbox_from_mask(gt)
            elif mode == "fg":
                if "fg" in ann_data:
                    anns.foreground_mask = ann_data["fg"]
                else:
                    anns.foreground_mask = synthetic_fg_from_gt(gt)
            elif mode == "bg":
                if "bg" in ann_data:
                    anns.background_mask = ann_data["bg"]
                else:
                    anns.background_mask = synthetic_bg_from_gt(gt)
            elif mode == "combined":
                # combined uses rectangle + fg + bg
                anns.rectangle = ann_data.get("rect") if "rect" in ann_data else bbox_from_mask(gt)
                anns.foreground_mask = ann_data.get("fg") if "fg" in ann_data else synthetic_fg_from_gt(gt)
                anns.background_mask = ann_data.get("bg") if "bg" in ann_data else synthetic_bg_from_gt(gt)

            # run segmentation
            try:
                result = pipeline.run(image, anns)
            except Exception as e:
                print(f"Segmentation failed for {case.name} mode {mode}: {e}")
                continue

            pred = np.where(result.refined_mask > 0, 255, 0).astype(np.uint8)
            metrics = binary_metrics(pred, gt)

            # save overlay
            vis_dir = out_root / case.name
            ensure_dir(vis_dir)
            overlay = overlay_pred_gt(image, pred, gt)
            vis_path = vis_dir / f"overlay_{mode}.png"
            cv2.imwrite(str(vis_path), overlay)

            row = {
                "case": case.name,
                "mode": mode,
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "ratio": metrics["ratio"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
            rows.append(row)

            # save predicted mask
            cv2.imwrite(str(vis_dir / f"pred_{mode}.png"), pred)

    # write CSV
    csv_path = out_root / "segmentation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "mode", "iou", "dice", "precision", "recall", "ratio", "tp", "fp", "fn"]) 
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Done. Results saved to {out_root}")


if __name__ == "__main__":
    main()
