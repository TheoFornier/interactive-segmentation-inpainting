"""Constantes liées aux modes d'interaction et d'affichage."""

from __future__ import annotations

from enum import Enum


class InteractionMode(str, Enum):
    NONE = "none"
    RECTANGLE = "rectangle"
    FOREGROUND = "foreground"
    BACKGROUND = "background"


class DisplayMode(str, Enum):
    ANNOTATIONS = "annotations"
    ORIGINAL = "original"
    MASK = "mask"
    OVERLAY = "overlay"
    OBJECT = "object"
    INPAINTED = "inpainted"
