#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V0.8.42 page-contract tests for spatial sidecar placement."""

from __future__ import annotations

import os
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from app_qt import GPRGuiQt


def _get_app() -> QApplication:
    return cast(QApplication, QApplication.instance() or QApplication([]))


def test_spatial_page_owns_visible_sidecar_controls() -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        assert win.page_terrain3d.sidecar_box.isHidden() is False
        assert win.page_terrain3d.rtk_sidecar_button.text() == "选择"
        assert win.page_terrain3d.imu_sidecar_button.text() == "选择"
        assert win.page_terrain3d.altimeter_sidecar_button.text() == "选择"
        assert "未选择" in win.page_terrain3d.rtk_sidecar_label.text()
    finally:
        win.close()
        app.processEvents()


def test_display_page_keeps_sidecar_compatibility_hidden() -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        assert hasattr(win.page_advanced, "sidecar_box")
        assert win.page_advanced.sidecar_box.isHidden() is True
    finally:
        win.close()
        app.processEvents()


def test_sidecar_state_syncs_to_space_and_compat_label() -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        win._set_sidecar_file("rtk", "/tmp/demo_rtk.csv")
        assert "demo_rtk.csv" in win.page_terrain3d.rtk_sidecar_label.text()
        assert "demo_rtk.csv" in win.page_advanced.rtk_sidecar_label.text()
    finally:
        win.close()
        app.processEvents()
