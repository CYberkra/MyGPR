from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app_qt import create_main_window


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_processing_bscan_canvases_are_top_aligned_in_windows_fit_viewport() -> None:
    app = _app()
    window = create_main_window()
    window.resize(1920, 1020)
    window.show()
    app.processEvents()
    window.switch_workspace("processing_lab")
    app.processEvents()

    canvas = window.processing_bscan_canvas
    assert canvas is not None
    parent = canvas.parentWidget()
    assert parent is not None
    assert canvas.geometry().y() <= 32
    assert canvas.geometry().height() >= 190
    assert parent.geometry().height() <= canvas.geometry().height() + 45

    window.close()


def test_spatial_preview_canvases_are_top_aligned_and_legible() -> None:
    app = _app()
    window = create_main_window()
    window.resize(1920, 1020)
    window.show()
    app.processEvents()
    window.switch_workspace("spatial")
    app.processEvents()

    assert window.spatial_map_canvas.geometry().y() <= 40
    assert window.spatial_map_canvas.geometry().height() >= 230
    assert window.spatial_elevation_canvas.geometry().y() <= 32
    assert window.spatial_elevation_canvas.geometry().height() >= 100

    window.close()
