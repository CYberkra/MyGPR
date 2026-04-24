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


def test_main_tabs_remove_batch_entry_but_keep_daily_processing_first():
    app = _get_app()
    win = GPRGuiQt()
    try:
        labels = [win.control_tabs.tabText(i) for i in range(win.control_tabs.count())]

        assert labels == ["日常处理", "调参与实验", "显示与对比", "质量与导出"]
        assert "批处理与报告" not in labels
        assert win.control_tabs.currentWidget() is win.page_basic
        assert win.page_workbench is not None
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
