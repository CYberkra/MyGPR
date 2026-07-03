from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app_qt import create_main_window
from ui.field_panels.field_ui_styles import COMPACT_1080P_FIT_SIZE


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_v0910_processing_bscan_work_area_prioritizes_large_canvases() -> None:
    app = _app()
    window = create_main_window()
    window.resize(*COMPACT_1080P_FIT_SIZE)
    window.show()
    app.processEvents()
    window.switch_workspace("processing_lab")
    app.processEvents()

    raw = window.processing_bscan_canvas
    assert raw is not None
    assert raw.geometry().height() >= 340
    assert raw.mapTo(window, raw.rect().topLeft()).y() < 260

    window.close()


def test_v0910_spatial_main_map_is_larger_than_auxiliary_charts() -> None:
    app = _app()
    window = create_main_window()
    window.resize(*COMPACT_1080P_FIT_SIZE)
    window.show()
    app.processEvents()
    window.switch_workspace("spatial")
    app.processEvents()

    main_map = window.spatial_map_canvas
    elevation = window.spatial_elevation_canvas
    assert main_map.geometry().height() >= 360
    assert main_map.geometry().width() > elevation.geometry().width() * 2
    assert main_map.geometry().height() > elevation.geometry().height() * 2

    window.close()
