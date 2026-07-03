from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QScrollArea

from app_qt import create_main_window
from ui.field_panels.field_ui_styles import COMPACT_1080P_FIT_SIZE, DEFAULT_1080P_SIZE, MIN_WORKBENCH_SIZE


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_default_window_size_is_not_larger_than_common_laptop_logical_area() -> None:
    assert DEFAULT_1080P_SIZE == (1480, 790)
    assert COMPACT_1080P_FIT_SIZE == (1536, 816)
    assert MIN_WORKBENCH_SIZE[0] <= 1120
    assert MIN_WORKBENCH_SIZE[1] <= 650


def test_home_page_has_no_scrollbars_in_compact_1080p_viewport() -> None:
    app = _app()
    window = create_main_window()
    window.resize(*COMPACT_1080P_FIT_SIZE)
    window.show()
    app.processEvents()
    window.switch_workspace("home")
    app.processEvents()

    visible_scrolls = [area for area in window.findChildren(QScrollArea) if area.isVisible()]
    assert visible_scrolls, "home page should keep its managed scroll area"
    for area in visible_scrolls:
        assert area.verticalScrollBar().maximum() == 0
        assert area.horizontalScrollBar().maximum() == 0

    window.close()


def test_all_primary_pages_render_in_compact_1080p_viewport() -> None:
    app = _app()
    window = create_main_window()
    window.resize(*COMPACT_1080P_FIT_SIZE)
    window.show()
    app.processEvents()
    assert window.compact_mode is True

    for key in ["home", "data_management", "processing_lab", "interpretation", "spatial", "delivery"]:
        window.switch_workspace(key)
        app.processEvents()
        page = window.workspace_pages[key]
        assert page.size().width() > 0
        assert page.size().height() > 0
        # The key regression target is not allowing the central workspace area to exceed the compact viewport.
        assert page.size().height() <= COMPACT_1080P_FIT_SIZE[1]
        for area in window.findChildren(QScrollArea):
            if area.isVisible():
                assert area.verticalScrollBar().maximum() == 0
                assert area.horizontalScrollBar().maximum() == 0

    window.close()


def test_primary_pages_render_in_real_windows_1080p_client_height() -> None:
    app = _app()
    window = create_main_window()
    # User-provided screenshots are 1920x1020 including the Qt frame/client capture.
    # This guards against content being clipped when Windows title bar / taskbar reduce usable height.
    window.resize(1920, 1020)
    window.show()
    app.processEvents()

    for key in ["home", "data_management", "processing_lab", "interpretation", "spatial", "delivery"]:
        window.switch_workspace(key)
        app.processEvents()
        page = window.workspace_pages[key]
        assert page.size().height() <= 1020

    window.close()
