#!/usr/bin/env python3
"""
Generate a small synthetic evaluation dataset under data/eval/

Creates three folders: image1, image2, image3 each containing:
 - image.png : RGB image with a simple colored shape on a background
 - gt_mask.png: binary mask for the shape (255 foreground, 0 background)

Run with the project's venv to ensure OpenCV is available:
  .venv312/bin/python scripts/generate_synthetic_eval_dataset.py
"""
from pathlib import Path
import numpy as np
import cv2


def make_case(path: Path, shape_type: str = "circle"):
    path.mkdir(parents=True, exist_ok=True)
    h, w = 320, 480
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    if shape_type == "circle":
        center = (int(w * 0.5), int(h * 0.5))
        radius = int(min(h, w) * 0.25)
        cv2.circle(img, center, radius, (10, 120, 200), -1)
        cv2.circle(mask, center, radius, 255, -1)
    elif shape_type == "rectangle":
        x0, y0 = int(w * 0.2), int(h * 0.25)
        x1, y1 = int(w * 0.8), int(h * 0.75)
        cv2.rectangle(img, (x0, y0), (x1, y1), (200, 80, 80), -1)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    else:
        # ellipse
        center = (int(w * 0.45), int(h * 0.55))
        axes = (int(w * 0.25), int(h * 0.15))
        cv2.ellipse(img, center, axes, 30, 0, 360, (80, 200, 120), -1)
        cv2.ellipse(mask, center, axes, 30, 0, 360, 255, -1)

    cv2.imwrite(str(path / "image.png"), img)
    cv2.imwrite(str(path / "gt_mask.png"), mask)


def main():
    root = Path("data/eval")
    make_case(root / "image1", "circle")
    make_case(root / "image2", "rectangle")
    make_case(root / "image3", "ellipse")
    print(f"Synthetic dataset created at {root.resolve()}")


if __name__ == "__main__":
    main()
