#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processing-lineage controller GUI smoke tests."""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtWidgets import QApplication

from app_qt import GPRGuiQt

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


def _close_window(app: QApplication, win: GPRGuiQt) -> None:
    win.close()
    app.processEvents()


def test_processing_lineage_controller_updates_stepper_and_preview_override():
    app = _get_app()
    win = GPRGuiQt("MyGPR V-test")
    try:
        raw = np.arange(20, dtype=np.float32).reshape(4, 5)
        step = raw + 10
        win.shared_data.load_data(raw, path="demo.csv")
        win.data = win.shared_data.current_data
        win.data_path = win.shared_data.data_path
        win.header_info = {"a_scan_length": 4, "num_traces": 5}
        win.original_data = win.shared_data.original_data
        win.shared_data.apply_current_data(
            step,
            push_history=True,
            label="Trace Median",
            header_info={"a_scan_length": 4, "num_traces": 5, "method_key": "trace_median_filter"},
        )
        win.data = win.shared_data.current_data
        win.header_info = win.shared_data.header_info

        win._update_processing_lineage_display()
        assert win.processing_lineage_controller.build_steps() == ["Raw", "Trace Median"]
        assert len(win._lineage_step_buttons) == 2

        win._on_processing_step_clicked(0)
        assert win._lineage_view_index == 0
        payload, header, _meta = win._get_active_plot_payload()
        assert payload is not None
        assert np.array_equal(payload, raw)
        assert "历史步骤" in str((header or {}).get("display_title"))

        win._on_processing_step_clicked(1)
        assert win._lineage_view_index is None
        payload, _header, _meta = win._get_active_plot_payload()
        assert payload is not None
        assert np.array_equal(payload, step)
    finally:
        _close_window(app, win)
