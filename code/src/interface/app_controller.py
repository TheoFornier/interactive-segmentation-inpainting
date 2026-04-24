"""Fenêtre principale de l'application."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import config
from src.core.image_loader import load_image_bgr
from src.core.image_saver import save_image
from src.core.preprocessing import resize_for_display
from src.evaluation.qualitative_analysis import build_segmentation_board
from src.interface.canvas_handler import ImageCanvas
from src.interface.events import DisplayMode, InteractionMode
from src.interface.state_manager import AppState
from src.interface.widgets import make_button
from src.inpainting.inpainting_pipeline import InpaintingPipeline
from src.segmentation.segmentation_pipeline import SegmentationPipeline
from src.visualization.display_masks import to_color_mask
from src.visualization.display_results import side_by_side


class SegmentationMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Segmentation interactive + inpainting — Version finale")
        self.resize(1500, 920)

        self.state = AppState()
        self.segmentation_pipeline = SegmentationPipeline()
        self.inpainting_pipeline = InpaintingPipeline()

        self.canvas = ImageCanvas()
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumWidth(320)

        self.status_label = QLabel("Charge une image puis dessine un rectangle ou utilise directement les pinceaux.")
        self._build_ui()
        self._connect_signals()
        self._log("Application prête. Rectangle optionnel : les pinceaux peuvent suffire.")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        controls = QVBoxLayout()
        controls.setAlignment(Qt.AlignTop)

        self.btn_load = make_button("1. Charger une image")
        self.btn_mode_rect = make_button("2. Mode rectangle")
        self.btn_mode_fg = make_button("3. Pinceau premier plan")
        self.btn_mode_bg = make_button("4. Pinceau arrière-plan")
        self.btn_segment = make_button("5. Lancer segmentation")
        self.btn_inpaint = make_button("6. Remplir l'arrière-plan")
        self.btn_clear = make_button("Réinitialiser annotations")
        self.btn_save = make_button("Sauvegarder résultats")

        self.btn_show_annotations = make_button("Afficher annotations")
        self.btn_show_original = make_button("Afficher original")
        self.btn_show_overlay = make_button("Afficher overlay")
        self.btn_show_mask = make_button("Afficher masque")
        self.btn_show_object = make_button("Afficher objet")
        self.btn_show_inpainted = make_button("Afficher fond reconstruit")

        self.brush_size_label = QLabel("Taille du pinceau")
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 64)
        self.brush_size.setValue(config.SCRIBBLE_RADIUS)

        for widget in [
            self.btn_load,
            self.btn_mode_rect,
            self.btn_mode_fg,
            self.btn_mode_bg,
            self.brush_size_label,
            self.brush_size,
            self.btn_segment,
            self.btn_inpaint,
            self.btn_show_annotations,
            self.btn_show_original,
            self.btn_show_overlay,
            self.btn_show_mask,
            self.btn_show_object,
            self.btn_show_inpainted,
            self.btn_clear,
            self.btn_save,
            self.status_label,
            self.info_box,
        ]:
            controls.addWidget(widget)

        root.addLayout(controls, stretch=0)
        root.addWidget(self.canvas, stretch=1)

    def _connect_signals(self) -> None:
        self.btn_load.clicked.connect(self.load_image)
        self.btn_mode_rect.clicked.connect(lambda: self._set_mode(InteractionMode.RECTANGLE))
        self.btn_mode_fg.clicked.connect(lambda: self._set_mode(InteractionMode.FOREGROUND))
        self.btn_mode_bg.clicked.connect(lambda: self._set_mode(InteractionMode.BACKGROUND))
        self.brush_size.valueChanged.connect(self.canvas.set_brush_radius)
        self.btn_segment.clicked.connect(self.run_segmentation)
        self.btn_inpaint.clicked.connect(self.run_inpainting)
        self.btn_clear.clicked.connect(self.clear_annotations)
        self.btn_save.clicked.connect(self.save_outputs)
        self.btn_show_annotations.clicked.connect(self.show_annotations)
        self.btn_show_original.clicked.connect(self.show_original)
        self.btn_show_overlay.clicked.connect(self.show_overlay)
        self.btn_show_mask.clicked.connect(self.show_mask)
        self.btn_show_object.clicked.connect(self.show_object)
        self.btn_show_inpainted.clicked.connect(self.show_inpainted)
        self.canvas.annotations_changed.connect(self._on_annotations_changed)

    def _set_mode(self, mode: InteractionMode) -> None:
        self.canvas.set_mode(mode)
        self.status_label.setText(f"Mode actif: {mode.value}")
        self._log(f"Mode -> {mode.value}")

    def _log(self, message: str) -> None:
        self.info_box.append(message)

    def _on_annotations_changed(self) -> None:
        ann = self.canvas.get_annotations()
        self.state.annotations = ann
        self.state.reset_results()

    def load_image(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir une image",
            str(config.INPUT_DIR),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not path_str:
            return

        image_bgr = load_image_bgr(path_str)
        resized, _ = resize_for_display(
            image_bgr,
            config.MAX_IMAGE_WIDTH,
            config.MAX_IMAGE_HEIGHT,
        )
        self.state.image_path = Path(path_str)
        self.state.image_bgr = resized
        self.state.annotations = self.canvas.get_annotations()
        self.state.reset_results()

        self.canvas.load_image(resized)
        self.status_label.setText("Image chargée. Dessine un rectangle autour de l'objet ou utilise les pinceaux FG/BG.")
        self._log(f"Image chargée: {path_str} - taille affichée: {resized.shape[1]}x{resized.shape[0]}")

    def clear_annotations(self) -> None:
        self.canvas.clear_annotations()
        self.state.annotations = self.canvas.get_annotations()
        self.state.reset_results()
        self.status_label.setText("Annotations réinitialisées.")
        self._log("Annotations supprimées.")

    def _check_image_loaded(self) -> bool:
        if self.state.image_bgr is None:
            QMessageBox.warning(self, "Erreur", "Charge d'abord une image.")
            return False
        return True

    def show_original(self) -> None:
        if self._check_image_loaded():
            self.state.display_mode = DisplayMode.ORIGINAL
            self.canvas.show_original()

    def show_annotations(self) -> None:
        if self._check_image_loaded():
            self.state.display_mode = DisplayMode.ANNOTATIONS
            self.canvas.show_annotations()

    def show_overlay(self) -> None:
        if self.state.segmentation_result is None:
            QMessageBox.warning(self, "Affichage", "Lance d'abord la segmentation.")
            return
        self.state.display_mode = DisplayMode.OVERLAY
        self.canvas.show_image(self.state.segmentation_result.overlay_bgr)

    def show_mask(self) -> None:
        if self.state.segmentation_result is None:
            QMessageBox.warning(self, "Affichage", "Lance d'abord la segmentation.")
            return
        self.state.display_mode = DisplayMode.MASK
        self.canvas.show_image(to_color_mask(self.state.segmentation_result.refined_mask))

    def show_object(self) -> None:
        if self.state.segmentation_result is None:
            QMessageBox.warning(self, "Affichage", "Lance d'abord la segmentation.")
            return
        self.state.display_mode = DisplayMode.OBJECT
        self.canvas.show_image(self.state.segmentation_result.extracted_object_bgr)

    def show_inpainted(self) -> None:
        if self.state.inpainting_result is None:
            QMessageBox.warning(self, "Affichage", "Lance d'abord le remplissage.")
            return
        self.state.display_mode = DisplayMode.INPAINTED
        self.canvas.show_image(self.state.inpainting_result.completed_background_bgr)

    def run_segmentation(self) -> None:
        if not self._check_image_loaded():
            return
        # Vérification : au moins un des trois (rectangle, FG, BG) doit être présent
        ann = self.canvas.get_annotations()
        fg_pixels = (ann.foreground_mask > 0).sum()
        bg_pixels = (ann.background_mask > 0).sum()
        has_rect = ann.has_rectangle() if hasattr(ann, 'has_rectangle') else False
        if not (has_rect or fg_pixels > 0 or bg_pixels > 0):
            QMessageBox.warning(self, "Annotations insuffisantes", "Merci d'annoter au moins le rectangle, le premier plan (FG) ou l'arrière-plan (BG) avant de lancer la segmentation.")
            self._log("Erreur : aucune annotation rectangle/FG/BG.")
            return
        # Avertissement si configuration risquée
        if not has_rect:
            if fg_pixels == 0:
                QMessageBox.warning(self, "Attention", "Aucune annotation de premier plan (FG) détectée. Le résultat risque d'être mauvais.")
            if bg_pixels == 0:
                QMessageBox.warning(self, "Attention", "Aucune annotation d'arrière-plan (BG) détectée. Le résultat risque d'être mauvais.")
        elif fg_pixels == 0 and bg_pixels == 0:
            QMessageBox.information(self, "Info", "Segmentation initialisée uniquement par rectangle. Pour de meilleurs résultats, ajoute des annotations FG/BG.")
        try:
            self.state.annotations = ann
            result = self.segmentation_pipeline.run(self.state.image_bgr, self.state.annotations)
            self.state.segmentation_result = result
            self.canvas.show_image(result.overlay_bgr)
            self.state.display_mode = DisplayMode.OVERLAY
            self.status_label.setText("Segmentation terminée. Tu peux maintenant lancer le remplissage.")
            self._log(f"Segmentation réussie. Aire masque: {(result.refined_mask > 0).sum()} pixels")
        except Exception as exc:
            QMessageBox.critical(self, "Segmentation", str(exc))
            self._log(f"Erreur segmentation: {exc}")

    def run_inpainting(self) -> None:
        if self.state.segmentation_result is None:
            QMessageBox.warning(self, "Erreur", "Lance d'abord la segmentation.")
            return

        from time import perf_counter
        from PySide6.QtWidgets import QApplication

        try:
            self.status_label.setText("Remplissage en cours... merci de patienter.")
            self._log("Inpainting lancé...")
            QApplication.processEvents()

            t0 = perf_counter()

            result = self.inpainting_pipeline.run(
                self.state.image_bgr,
                self.state.segmentation_result.refined_mask,
            )

            dt = perf_counter() - t0

            self.state.inpainting_result = result
            self.state.display_mode = DisplayMode.INPAINTED
            self.canvas.show_image(result.completed_background_bgr)
            self.status_label.setText(f"Remplissage terminé via {result.backend_name}.")
            self._log(
                f"Inpainting backend: {result.backend_name} | ROI: {result.roi_shape[1]}x{result.roi_shape[0]} | "
                f"Rect: {result.roi_rect} | Temps: {dt:.2f}s"
            )

            if result.backend_name == "opencv-fallback":
                self._log("Attention: fallback OpenCV actif. Pour un meilleur résultat visuel, installe LaMa.")

        except Exception as exc:
            QMessageBox.critical(self, "Inpainting", f"Erreur pendant le remplissage : {exc}")
            self._log(f"Erreur inpainting: {exc}")

    def save_outputs(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        import re
        if self.state.image_bgr is None or self.state.image_path is None:
            QMessageBox.warning(self, "Erreur", "Aucune image à sauvegarder.")
            return

        # Demander le nom du dossier à l'utilisateur
        default_name = self.state.image_path.stem
        name, ok = QInputDialog.getText(self, "Nom du dossier de sauvegarde", "Choisis un nom pour le dossier de sauvegarde dans output/ :", text=default_name)
        if not ok or not name.strip():
            self._log("Sauvegarde annulée (aucun nom fourni).")
            return
        name = re.sub(r'[^\w\-]', '_', name.strip())  # Nettoyage basique du nom
        base_dir = (config.PROJECT_ROOT / "data" / "output" / name)
        if base_dir.exists():
            # Proposer un nom alternatif
            i = 1
            alt_dir = base_dir
            while alt_dir.exists():
                alt_dir = base_dir.parent / f"{name}_{i}"
                i += 1
            QMessageBox.warning(self, "Conflit de nom", f"Le dossier '{name}' existe déjà. Un nom alternatif sera utilisé : '{alt_dir.name}'")
            self._log(f"Conflit de nom, utilisation de : {alt_dir.name}")
            base_dir = alt_dir

        # Créer la structure interne
        subdirs = ["background_removed", "inpainted", "segmented_objects", "visualizations"]
        for sub in subdirs:
            (base_dir / sub).mkdir(parents=True, exist_ok=True)

        stem = self.state.image_path.stem

        # Sauvegarde segmentation
        if self.state.segmentation_result is not None:
            seg = self.state.segmentation_result
            save_image(base_dir / "segmented_objects" / f"{stem}_object.png", seg.extracted_object_bgr)
            save_image(base_dir / "background_removed" / f"{stem}_removed.png", seg.background_removed_bgr)
            save_image(base_dir / "visualizations" / f"{stem}_overlay.png", seg.overlay_bgr)
            save_image(base_dir / "visualizations" / f"{stem}_contours.png", seg.contour_preview_bgr)
            visual = side_by_side(
                self.state.image_bgr,
                to_color_mask(seg.refined_mask),
                seg.extracted_object_bgr,
            )
            save_image(base_dir / "visualizations" / f"{stem}_segmentation_triptych.png", visual)

        # Sauvegarde inpainting
        if self.state.inpainting_result is not None and self.state.segmentation_result is not None:
            inp = self.state.inpainting_result
            save_image(base_dir / "inpainted" / f"{stem}_inpainted.png", inp.completed_background_bgr)
            board = build_segmentation_board(
                self.state.image_bgr,
                self.state.segmentation_result.overlay_bgr,
                self.state.segmentation_result.extracted_object_bgr,
                inp.completed_background_bgr,
            )
            save_image(base_dir / "visualizations" / f"{stem}_final_board.png", board)

        self._log(f"Résultats sauvegardés dans {base_dir.relative_to(config.PROJECT_ROOT)}.")
        QMessageBox.information(self, "Sauvegarde", f"Résultats sauvegardés dans : {base_dir.relative_to(config.PROJECT_ROOT)}")
