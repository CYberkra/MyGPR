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
        assert win._lineage_step_buttons[0].text().startswith("Raw")
        assert win._lineage_step_buttons[1].text().startswith("当前")
        assert "Trace Median" in win.processing_lineage_controller.build_copy_text()
        assert not hasattr(win, "_lineage_copy_button")
        assert not hasattr(win, "_lineage_compare_button")
        assert getattr(win, "_lineage_slider_compare_button").text() == "滑动对比"
        assert getattr(win, "_lineage_grid_compare_button").text() == "网格对比"
        assert getattr(win, "_lineage_diff_compare_button").text() == "差值图"
        assert not hasattr(win, "_lineage_compare_exit_button")
        export_steps = win.processing_lineage_controller.build_export_steps()
        assert export_steps[-1]["role"] == "current"
        assert export_steps[-1]["ui_status"] == "当前正式结果"

        win._on_processing_step_clicked(0)
        assert win._lineage_view_index == 0
        payload, header, _meta = win._get_active_plot_payload()
        assert payload is not None
        assert np.array_equal(payload, raw)
        assert "历史步骤" in str((header or {}).get("display_title"))
        assert "不会修改当前正式结果" in win.processing_lineage_controller.step_detail_text(0)

        # 对比篮选择与模式按钮是分离的：先选 Raw 和当前，再进入滑动对比。
        selector_raw = win._lineage_step_select_buttons[0]
        selector_current = win._lineage_step_select_buttons[1]
        selector_raw.click()
        selector_current.click()
        app.processEvents()
        assert getattr(win, "_lineage_slider_compare_button").isEnabled()
        assert getattr(win, "_lineage_diff_compare_button").isEnabled()
        assert getattr(win, "_lineage_grid_compare_button").isEnabled()
        assert win.processing_lineage_controller._selected_compare_indices_sorted() == [0, 1]

        slider_btn = getattr(win, "_lineage_slider_compare_button")
        slider_btn.click()
        app.processEvents()
        assert win.page_advanced.compare_var.isChecked()
        assert win.page_advanced.slider_compare_var.isChecked()
        assert win.page_advanced.mode_slider.isChecked()
        assert slider_btn.isChecked()
        assert win.processing_lineage_controller.get_active_compare_mode() == "slider"
        assert getattr(win, "_lineage_compare_source_indices") == [0, 1]
        assert any(s.get("source") == "processing_lineage" for s in win.compare_snapshots)

        slider_btn.click()
        app.processEvents()
        assert not win.page_advanced.compare_var.isChecked()
        assert not win.page_advanced.slider_compare_var.isChecked()
        assert win.page_advanced.mode_single.isChecked()
        assert not slider_btn.isChecked()
        assert win.processing_lineage_controller.get_active_compare_mode() is None
        assert getattr(win, "_lineage_compare_source_index", None) is None
        assert not any(s.get("source") == "processing_lineage" for s in win.compare_snapshots)
        payload, _header, _meta = win._get_active_plot_payload()
        assert payload is not None
        assert np.array_equal(payload, step)

        # 三到四步由网格对比承载；两步时也允许网格作为轻量比较视图。
        grid_btn = getattr(win, "_lineage_grid_compare_button")
        grid_btn.click()
        app.processEvents()
        assert grid_btn.isChecked()
        assert win.processing_lineage_controller.get_active_compare_mode() == "grid"
        pairs = win._build_compare_data_pairs(win.data)
        assert len(pairs) == 2

        grid_btn.click()
        app.processEvents()
        diff_btn = getattr(win, "_lineage_diff_compare_button")
        diff_btn.click()
        app.processEvents()
        assert diff_btn.isChecked()
        assert win.processing_lineage_controller.get_active_compare_mode() == "diff"
        diff_pairs = win._build_compare_data_pairs(win.data)
        assert len(diff_pairs) == 1
        assert diff_pairs[0][0].startswith("|")

        win.processing_lineage_controller.clear_compare_selection()
        app.processEvents()
        assert win.processing_lineage_controller._selected_compare_indices_sorted() == []

        win._on_processing_step_clicked(1)
        assert win._lineage_view_index is None
        payload, _header, _meta = win._get_active_plot_payload()
        assert payload is not None
        assert np.array_equal(payload, step)
    finally:
        _close_window(app, win)
