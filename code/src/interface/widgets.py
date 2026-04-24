"""Création des widgets principaux."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton

import config


def make_button(label: str) -> QPushButton:
    button = QPushButton(label)
    button.setMinimumHeight(config.UI_BUTTON_HEIGHT)
    return button
