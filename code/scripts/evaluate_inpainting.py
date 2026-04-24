#!/usr/bin/env python3
"""
Évaluation quantitative de l'inpainting : LaMa vs OpenCV Telea vs Navier-Stokes.

Protocole :
  - Pour chaque image de data/eval/ (image.png + gt_mask.png) :
    1. Dilate le masque GT pour obtenir un masque de "trou" synthétique
    2. Applique ce masque sur l'image originale pour créer l'image dégradée
    3. Infère les 3 méthodes d'inpainting sur l'image dégradée
    4. Compare le résultat à l'image originale avec PSNR et SSIM
    5. Sauvegarde les visuels et un CSV de synthèse

Usage :
  PYTHONPATH=. .venv312/bin/python scripts/evaluate_inpainting.py --data data/eval --out results/eval_inpaint
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ── PSNR / SSIM fallbacks using only NumPy/OpenCV ──────────────────────────
def psnr(ref: np.ndarray, pred: np.ndarray) -> float:
    if HAS_SKIMAGE:
        return float(psnr_fn(ref, pred, data_range=255))
    mse = np.mean((ref.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def ssim(ref: np.ndarray, pred: np.ndarray) -> float:
    if HAS_SKIMAGE:
        return float(ssim_fn(ref, pred, channel_axis=2, data_range=255))
    # Simple luminance SSIM on grayscale (fallback)
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g_ref, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g_pred, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = cv2.GaussianBlur(g_ref ** 2, (11, 11), 1.5) - mu1_sq
    s2 = cv2.GaussianBlur(g_pred ** 2, (11, 11), 1.5) - mu2_sq
    s12 = cv2.GaussianBlur(g_ref * g_pred, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return float(ssim_map.mean())


# ── Inpainting methods ───────────────────────────────────────────────────────
def inpaint_telea(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def inpaint_ns(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)


def inpaint_lama(img: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    try:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        from PIL import Image as PilImage
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PilImage.fromarray(img_rgb)
        pil_mask = PilImage.fromarray(mask)
        result_pil = lama(pil_img, pil_mask)
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        return result.astype(np.uint8)
    except Exception as e:
        print(f"  [LaMa unavailable: {e}]")
        return None


# ── Helpers ──────────────────────────────────────────────────────────────────
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return np.where(m > 127, 255, 0).astype(np.uint8)


def dilate_mask(mask: np.ndarray, pixels: int = 15) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    return cv2.dilate(mask, k, iterations=1)


def apply_hole(img: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    degraded = img.copy()
    degraded[hole_mask > 0] = 0
    return degraded


def comparison_strip(original, degraded, results: dict[str, np.ndarray | None]) -> np.ndarray:
    """Build a horizontal comparison strip: original | degraded | method1 | ..."""
    h, w = original.shape[:2]
    label_h = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    panels = [("original", original), ("dégradée", degraded)]
    for name, img in results.items():
        if img is not None:
            panels.append((name, img))
    strip_w = w * len(panels)
    strip = np.zeros((h + label_h, strip_w, 3), dtype=np.uint8)
    for i, (label, panel) in enumerate(panels):
        strip[label_h:, i * w:(i + 1) * w] = panel
        cv2.putText(strip, label, (i * w + 4, 16), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return strip


# ── Main ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/eval")
    parser.add_argument("--out", default="results/eval_inpaint")
    parser.add_argument("--dilation", type=int, default=15,
                        help="Pixels to dilate GT mask to create the synthetic hole")
    args = parser.parse_args(argv)

    data_root = Path(args.data)
    out_root = Path(args.out)
    ensure_dir(out_root)

    rows = []
    for case in sorted([d for d in data_root.iterdir() if d.is_dir()]):
        img_path = case / "image.png"
        gt_path = case / "gt_mask.png"
        if not img_path.exists() or not gt_path.exists():
            print(f"Skipping {case.name}")
            continue

        print(f"\n=== {case.name} ===")
        original = read_image(img_path)
        gt_mask = read_mask(gt_path)

        hole_mask = dilate_mask(gt_mask, pixels=args.dilation)
        degraded = apply_hole(original, hole_mask)

        # Save hole mask and degraded image for reference
        vis_dir = out_root / case.name
        ensure_dir(vis_dir)
        cv2.imwrite(str(vis_dir / "hole_mask.png"), hole_mask)
        cv2.imwrite(str(vis_dir / "degraded.png"), degraded)

        methods = {
            "Telea": inpaint_telea(degraded, hole_mask),
            "Navier-Stokes": inpaint_ns(degraded, hole_mask),
        }
        lama_result = inpaint_lama(degraded, hole_mask)
        if lama_result is not None:
            methods["LaMa"] = lama_result

        for method_name, result in methods.items():
            p = psnr(original, result)
            s = ssim(original, result)
            print(f"  {method_name:15s}  PSNR={p:6.2f} dB   SSIM={s:.4f}")
            cv2.imwrite(str(vis_dir / f"result_{method_name.lower().replace(' ', '_').replace('-', '_')}.png"), result)
            rows.append({
                "case": case.name,
                "method": method_name,
                "psnr_db": round(p, 3),
                "ssim": round(s, 4),
            })

        # Comparison strip
        strip = comparison_strip(original, degraded, methods)
        cv2.imwrite(str(vis_dir / "comparison.png"), strip)

    # Write CSV
    csv_path = out_root / "inpainting_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "method", "psnr_db", "ssim"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nDone. Results saved to {out_root.resolve()}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
