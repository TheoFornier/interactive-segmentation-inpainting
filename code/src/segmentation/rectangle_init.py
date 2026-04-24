"""Outils pour l'initialisation par rectangle."""

from __future__ import annotations


def normalize_rectangle(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    if w < 0:
        x += w
        w = abs(w)
    if h < 0:
        y += h
        h = abs(h)
    return x, y, w, h
