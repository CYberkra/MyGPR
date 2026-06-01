#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoTune synchronization controller GUI smoke tests."""

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


def test_autotune_sync_controller_dataset_roi_and_no_prior_smoke():
    app = _get_app()
    win = GPRGuiQt("MyGPR V-test")
    try:
        data = np.arange(40, dtype=np.float32).reshape(8, 5)
        win.shared_data.load_data(
            data,
            path="demo.csv",
            source="csv",
            header_info={"source": "csv", "component": "Ey", "a_scan_length": 8, "num_traces": 5},
        )
        win.data = win.shared_data.current_data
        win.data_path = win.shared_data.data_path
        win.header_info = win.shared_data.header_info
        win.original_data = win.shared_data.original_data

        assert win.autotune_sync_controller.host is win
        win._sync_auto_tune_page_dataset_state({"reason": "loaded", "source": "csv"})
        assert win.page_auto_tune.state.data_source == "已载入"
        assert win.page_auto_tune.state.data_shape == (8, 5)
        assert win.page_auto_tune.state.component == "Ey"
        assert win.page_auto_tune._current_data is data or win.page_auto_tune._current_data is win.data

        win._set_manual_roi_pick_enabled(True)
        assert win._is_manual_roi_pick_enabled() is True
        assert win.page_auto_tune.btn_pick_roi.isChecked() is True
        assert "开启" in win.page_auto_tune.roi_picker_status_label.text()

        win._manual_roi_values = {
            "dist_start": 0.0,
            "dist_end": 4.0,
            "time_start": 0.0,
            "time_end": 7.0,
        }
        bounds = win._get_manual_roi_bounds()
        assert bounds is not None
        assert bounds["time_start_idx"] <= bounds["time_end_idx"]
        assert bounds["dist_start_idx"] <= bounds["dist_end_idx"]

        roi_spec = win._build_auto_tune_roi_spec("prefer_crop")
        assert roi_spec["source"] == "manual"
        assert win._roi_available_for_no_prior() is True
        policy = win._build_no_prior_qc_policy(metrics={}, airborne_qc={})
        assert isinstance(policy, dict)
        assert "no_prior_level" in policy

        win._clear_manual_roi()
        assert win._manual_roi_values is None
    finally:
        _close_window(app, win)
