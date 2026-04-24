"""Exécution de lots d'expériences simples sur plusieurs images."""

from __future__ import annotations

from pathlib import Path

import cv2

import config
from src.core.image_loader import load_image_bgr
from src.core.image_saver import save_image
from src.evaluation.qualitative_analysis import build_segmentation_board
from src.inpainting.inpainting_pipeline import InpaintingPipeline
from src.segmentation.seeds_manager import UserAnnotations
from src.segmentation.segmentation_pipeline import SegmentationPipeline


def run_batch_with_rectangles(job_list: list[tuple[str | Path, tuple[int, int, int, int]]]) -> list[Path]:
    segmentation = SegmentationPipeline()
    inpainting = InpaintingPipeline()
    outputs: list[Path] = []

    for image_path, rectangle in job_list:
        image_bgr = load_image_bgr(image_path)
        annotations = UserAnnotations(rectangle=rectangle)
        seg = segmentation.run(image_bgr, annotations)
        inp = inpainting.run(image_bgr, seg.refined_mask)
        board = build_segmentation_board(image_bgr, seg.overlay_bgr, seg.extracted_object_bgr, inp.completed_background_bgr)
        out_path = config.VISUALIZATIONS_DIR / f"{Path(image_path).stem}_board.png"
        save_image(out_path, board)
        outputs.append(out_path)
    return outputs
