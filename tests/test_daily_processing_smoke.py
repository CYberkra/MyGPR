#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日常处理与主标签页冒烟测试。"""

from __future__ import annotations

import os
import time

import numpy as np
from PyQt6.QtWidgets import QApplication

from app_qt import GPRGuiQt

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_worker(app: QApplication, win: GPRGuiQt, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if win._worker is None and win._worker_thread is None:
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for processing worker to finish")


def test_main_window_uses_workflow_studio_without_legacy_control_tabs():
    app = _get_app()
    win = GPRGuiQt()
    try:
        assert win.control_tabs is None
        assert win.page_basic.isHidden()
        assert win.page_auto_tune is None
        assert win.page_advanced is None
        assert win.page_quality is None
        assert hasattr(win, "page_workflow")
        assert win.page_workflow.objectName() == "workflowStudioPage"
        assert win._main_splitter is None
        assert win.page_workflow.project_panel.title() == "项目 / 数据"
        assert win.page_workflow.palette_panel.title() == "节点库"
        assert win.page_workflow.inspector_box.title() == "属性 / 检查"
        old_page_attr = "page_" + "work" + "bench"
        assert not hasattr(win, old_page_attr)
    finally:
        win.close()
        app.processEvents()


def test_daily_processing_flow_supports_apply_undo_and_reset():
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.tile(np.linspace(0, 10, 80, dtype=np.float32)[:, None], (1, 16))
        raw += np.linspace(0, 1, 16, dtype=np.float32)[None, :]
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        app.processEvents()

        dewow_index = win.page_basic.method_keys.index("dewow")
        win.page_basic.method_combo.setCurrentIndex(dewow_index)
        app.processEvents()

        win.apply_method_manual()
        _wait_for_worker(app, win)

        assert not np.array_equal(win.data, raw)
        assert win.shared_data.can_undo() is True
        assert (win._last_run_summary or {}).get("run_type") == "single"

        win.undo_last()
        assert np.array_equal(win.data, raw)

        win.apply_method_manual()
        _wait_for_worker(app, win)
        assert not np.array_equal(win.data, raw)

        win.reset_original()
        assert np.array_equal(win.data, raw)
    finally:
        if win._worker_thread is not None:
            _wait_for_worker(app, win)
        win.close()
        app.processEvents()


def test_tuning_and_preview_controls_open_as_studio_dialogs():
    app = _get_app()
    win = GPRGuiQt()
    try:
        win.open_tuning_lab()
        app.processEvents()
        assert win._tuning_lab_dialog.isVisible()
        assert win.page_auto_tune.parent() is win._tuning_lab_dialog

        win.open_preview_settings()
        app.processEvents()
        assert win._preview_settings_dialog.isVisible()
        assert win.page_advanced.parent() is win._preview_settings_dialog
    finally:
        if hasattr(win, "_tuning_lab_dialog"):
            win._tuning_lab_dialog.close()
        if hasattr(win, "_preview_settings_dialog"):
            win._preview_settings_dialog.close()
        win.close()
        app.processEvents()
