"""Lancement de l'interface desktop."""

import sys

from PySide6.QtWidgets import QApplication

from src.interface.app_controller import SegmentationMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = SegmentationMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
