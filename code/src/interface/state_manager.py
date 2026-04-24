"""État global de l'application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.interface.events import DisplayMode
from src.segmentation.seeds_manager import UserAnnotations


@dataclass
class AppState:
    image_path: Optional[Path] = None
    image_bgr: np.ndarray | None = None
    annotations: UserAnnotations = field(default_factory=UserAnnotations)
    segmentation_result: object | None = None
    inpainting_result: object | None = None
    display_mode: DisplayMode = DisplayMode.ANNOTATIONS

    def reset_results(self) -> None:
        self.segmentation_result = None
        self.inpainting_result = None
        self.display_mode = DisplayMode.ANNOTATIONS

    def has_image(self) -> bool:
        return self.image_bgr is not None
