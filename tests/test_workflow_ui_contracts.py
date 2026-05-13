#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow UI contract tests."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QSlider

from app_qt import GPRGuiQt


def _get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _row_for_method(win: GPRGuiQt, method_id: str) -> int:
    for row, method in enumerate(win.page_workflow.config.methods):
        if method.method_id == method_id:
            return row
    raise AssertionError(f"workflow method not found: {method_id}")


def test_workflow_tab_exposes_default_uavgpr_chain_and_agc_warning():
    app = _get_app()
    win = GPRGuiQt()
    try:
        labels = [win.control_tabs.tabText(i) for i in range(win.control_tabs.count())]
        assert labels[1] == "工作流"

        default_methods = [method.method_id for method in win.page_workflow.config.methods]
        assert default_methods[:4] == [
            "set_zero_time",
            "dc_shift",
            "dewow",
            "frequency_filter_1d",
        ]
        assert "manual_velocity_model" in default_methods
        assert "geometry_depth_context" in default_methods

        gain_row = _row_for_method(win, "sec_gain")
        win.page_workflow.step_list.setCurrentRow(gain_row)
        app.processEvents()

        gain_candidates = [
            win.page_workflow.method_combo.itemData(index)
            for index in range(win.page_workflow.method_combo.count())
        ]
        assert "agcGain" in gain_candidates
        assert "非严格保幅" in win.page_workflow.stage_warning.text()
    finally:
        win.close()
        app.processEvents()


def test_workflow_hidden_steps_are_excluded_from_runtime_methods():
    app = _get_app()
    win = GPRGuiQt()
    try:
        zero_row = _row_for_method(win, "set_zero_time")
        win.page_workflow.step_list.setCurrentRow(zero_row)
        win.page_workflow.hidden_check.setChecked(True)
        app.processEvents()

        enabled_methods = [
            method.method_id for method in win.page_workflow.get_enabled_methods()
        ]
        assert "set_zero_time" not in enabled_methods
        assert "dc_shift" in enabled_methods
    finally:
        win.close()
        app.processEvents()


def test_workflow_numeric_params_render_slider_and_spinbox_pair():
    app = _get_app()
    win = GPRGuiQt()
    try:
        dewow_row = _row_for_method(win, "dewow")
        win.page_workflow.step_list.setCurrentRow(dewow_row)
        app.processEvents()

        assert "window" in win.page_workflow._param_controls
        sliders = win.page_workflow.param_host.findChildren(QSlider)
        assert sliders, "integer window parameter should render a slider"
    finally:
        win.close()
        app.processEvents()
