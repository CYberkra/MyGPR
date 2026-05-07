#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 预设契约测试。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import app_qt
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QGroupBox, QScrollArea, QStackedWidget

from app_qt import GPRGuiQt
from core.methods_registry import (
    PROCESSING_METHODS,
    get_method_category_label,
    get_public_method_keys,
)
from core.preset_profiles import (
    DEFAULT_STARTUP_PRESET_KEY,
    GUI_PRESETS_V1,
    RECOMMENDED_RUN_PROFILES,
)
from core.workflow_template_manager import WorkflowTemplateManager
from core.workflow_data import QUICK_PRESETS
from ui.gui_method_browser import MethodBrowserTree


class _DummyCanvasEvent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _DummyAxes:
    def add_patch(self, _patch):
        return None

    def get_xlim(self):
        return (0.0, 10.0)

    def get_ylim(self):
        return (0.0, 10.0)


def _get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _top_level_group_titles(page) -> list[str]:
    if isinstance(page, QGroupBox):
        return [page.title()] if not page.isHidden() else []

    if hasattr(page, "page_mode") and hasattr(page, "stack") and isinstance(page.stack, QStackedWidget):
        titles = []
        for page_index in range(page.stack.count()):
            titles.extend(_top_level_group_titles(page.stack.widget(page_index)))
        return titles

    if isinstance(page, QScrollArea):
        content = page.widget()
        content_layout = content.layout()
    else:
        content_layout = page.layout()

    titles = []
    for index in range(content_layout.count()):
        widget = content_layout.itemAt(index).widget()
        if widget is None:
            continue
        if isinstance(widget, QGroupBox):
            if not widget.isHidden():
                titles.append(widget.title())
        elif isinstance(widget, QScrollArea):
            titles.extend(_top_level_group_titles(widget))
    return titles


def _find_home_toolbar_action(toolbar):
    for action in toolbar.actions():
        haystack = " ".join(
            part
            for part in [
                action.text() or "",
                action.toolTip() or "",
                action.statusTip() or "",
                action.iconText() or "",
            ]
            if part
        ).lower()
        if "home" in haystack or "reset original view" in haystack:
            return action
    if toolbar.actions():
        return toolbar.actions()[0]
    raise AssertionError("Main plot toolbar has no actions")


def test_startup_preset_matches_registry_contract():
    app = _get_app()
    win = GPRGuiQt()
    try:
        preset = GUI_PRESETS_V1[DEFAULT_STARTUP_PRESET_KEY]
        assert win._selected_preset_key == DEFAULT_STARTUP_PRESET_KEY
        assert not hasattr(win.page_advanced, "preset_combo")
        assert not hasattr(win.page_advanced, "fast_preview_var")
        assert not hasattr(win.page_advanced, "max_samples_edit")
        assert not hasattr(win.page_advanced, "max_traces_edit")
        assert not hasattr(win.page_advanced, "display_downsample_var")
        assert not hasattr(win.page_advanced, "display_max_samples_edit")
        assert not hasattr(win.page_advanced, "display_max_traces_edit")
        assert win._method_param_overrides == preset["method_params"]
        assert win.page_advanced.sidecar_box.isHidden() is False
    finally:
        win.close()
        app.processEvents()


def test_apply_preset_updates_ui_and_method_overrides():
    app = _get_app()
    win = GPRGuiQt()
    try:
        preset_key = "denoise_first"
        preset = GUI_PRESETS_V1[preset_key]
        win._apply_preset_by_key(preset_key)

        assert win._selected_preset_key == preset_key
        assert win.page_advanced.normalize_var.isChecked() is bool(
            preset["ui"]["normalize"]
        )
        assert win.page_advanced.demean_var.isChecked() is bool(preset["ui"]["demean"])
        assert win.page_advanced.p_low_edit.text() == str(preset["ui"]["p_low"])
        assert win._method_param_overrides["dewow"] == preset["method_params"]["dewow"]
    finally:
        win.close()
        app.processEvents()


def test_apply_selected_preset_reuses_current_selected_preset_key():
    app = _get_app()
    win = GPRGuiQt()
    try:
        win._apply_preset_by_key("denoise_first")
        preset_key = win._selected_preset_key or DEFAULT_STARTUP_PRESET_KEY
        win._apply_preset_by_key(preset_key)
        assert win._selected_preset_key == "denoise_first"
        assert (
            win._method_param_overrides["dewow"]
            == GUI_PRESETS_V1["denoise_first"]["method_params"]["dewow"]
        )
    finally:
        win.close()
        app.processEvents()


def test_motion_compensation_v1_preset_order_matches_plan_order():
    expected_order = [
        "trajectory_smoothing",
        "motion_compensation_speed",
        "motion_compensation_attitude",
        "motion_compensation_height",
        "motion_compensation_vibration",
    ]

    assert [
        item["method_id"] for item in QUICK_PRESETS["motion_compensation_v1"]["methods"]
    ] == expected_order
    assert RECOMMENDED_RUN_PROFILES["motion_compensation_v1"]["order"] == expected_order


def test_auto_tune_defaults_live_in_auto_tune_page():
    app = _get_app()
    win = GPRGuiQt()
    try:
        assert win.page_auto_tune.get_auto_tune_roi_mode() == "prefer_crop"
        assert win.page_auto_tune.get_auto_tune_search_mode() == "standard"
        assert win.page_auto_tune.btn_auto_tune.isEnabled() is True
        assert win.page_advanced.roi_status_label.text() == "手动 ROI: 未设置"
        assert win.page_advanced.btn_clear_manual_roi.isEnabled() is False
        assert win.page_basic.action_apply_manual.text() == "使用当前参数（默认）"
        assert win.page_basic.action_apply_auto_tuned.text() == "使用自动调参参数"
        assert win.page_basic.action_apply_auto_tuned.isEnabled() is True
    finally:
        win.close()
        app.processEvents()


def test_auto_tune_page_handles_state_transitions():
    app = _get_app()
    win = GPRGuiQt()
    try:
        win.page_auto_tune.reset_for_method("dewow")
        assert "支持自动选参" in win.page_auto_tune.auto_tune_summary.toPlainText()
        assert win.page_auto_tune.result_state_label.text() == "等待分析"

        win.page_auto_tune.show_running("当前裁剪区", "fast")
        running_text = win.page_auto_tune.auto_tune_summary.toPlainText()
        assert "ROI 来源: 当前裁剪区" in running_text
        assert "搜索模式: fast" in running_text
        assert win.page_auto_tune.btn_view_auto_tune.isEnabled() is False
        assert win.page_auto_tune.result_state_label.text() == "分析中"

        result = {
            "method_key": "dewow",
            "method_name": "低频漂移矫正（Dewow）",
            "coarse_trials": [{"score": 1.0}],
            "fine_trials": [{"score": 1.1}],
            "all_trials": [{"score": 1.0}, {"score": 1.1}],
            "failed_trials": [{"params": {"window": "bad"}, "error": "bad param"}],
            "selection_confidence": 0.42,
            "selection_margin": 0.08,
            "execution_stats": {
                "total_trial_count": 2,
                "valid_trial_count": 1,
                "failed_trial_count": 1,
                "cache_hit_count": 0,
            },
            "recommended_profile": "balanced",
            "best_score": 1.1,
            "best_params": {"window": 23},
            "profiles": {
                "balanced": {
                    "label": "平衡档",
                    "score": 1.1,
                    "params": {"window": 23},
                    "metrics": {"baseline_bias": 0.1},
                    "reason": "测试摘要",
                }
            },
            "roi_info": {"label": "自动 ROI"},
        }
        win.page_auto_tune.show_result(result)
        result_text = win.page_auto_tune.auto_tune_summary.toPlainText()
        assert "方法: 低频漂移矫正（Dewow）" in result_text
        assert "推荐调试档: 平衡档" in result_text
        assert "稳定性:" in result_text
        assert win.page_auto_tune.result_state_label.text() == "结果可用"
        assert win.page_auto_tune.recommended_profile_label.text() == "平衡档"
        assert "总候选 2" in win.page_auto_tune.execution_stats_label.text()
        assert "失败候选" in win.page_auto_tune.risk_hint_label.text()
        assert win.page_auto_tune.btn_view_auto_tune.isEnabled() is True

        win.page_auto_tune.show_error("demo error")
        assert "自动选参失败:" in win.page_auto_tune.auto_tune_summary.toPlainText()
        assert win.page_auto_tune.result_state_label.text() == "失败"
        assert win.page_auto_tune.btn_view_auto_tune.isEnabled() is False
    finally:
        win.close()
        app.processEvents()


def test_phase2_tabs_expose_prioritized_group_hierarchy_and_bridge():
    app = _get_app()
    win = GPRGuiQt()
    try:
        assert _top_level_group_titles(win.page_advanced) == [
            "显示模式",
            "单图查看",
            "核心显示",
            "聚焦裁剪",
            "ROI 状态",
            "增强与对比辅助",
            "可选 RTK/IMU 辅助文件",
        ]
        assert win.page_advanced.sidecar_box.isHidden() is False

        assert _top_level_group_titles(win.page_auto_tune) == [
            "实验流程",
        ]
        assert win.page_auto_tune.btn_open_workbench.text() == "进入工作台深度实验"

        assert _top_level_group_titles(win.page_quality) == [
            "查看顺序",
            "导出与诊断",
            "质量摘要",
            "图表查看",
            "运行记录",
        ]
    finally:
        win.close()
        app.processEvents()


def test_auto_tune_workbench_bridge_switches_to_workbench_mode():
    app = _get_app()
    win = GPRGuiQt()
    try:
        assert win._content_stack.currentWidget() is win._main_content_widget
        win.page_auto_tune.btn_open_workbench.click()
        app.processEvents()
        assert win._content_stack.currentWidget() is win.page_workbench
    finally:
        win.close()
        app.processEvents()


def test_runtime_drawer_prefers_global_log_and_demotes_perf_metrics():
    app = _get_app()
    win = GPRGuiQt()
    try:
        assert win.btn_toggle_global_log.text() == "全局日志"
        assert win.btn_toggle_quality.text() == "质量摘要"

        raw = np.arange(120, dtype=np.float32).reshape(10, 12) / 10.0
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win._log("runtime drawer smoke")

        win.btn_toggle_global_log.click()
        app.processEvents()

        assert win._runtime_panel_container.isHidden() is False
        assert win._runtime_panel_stack.currentWidget() is win.global_log_box
        assert "runtime drawer smoke" in win.runtime_log_view.toPlainText()
        assert win.performance_diag_box.title() == "性能诊断（低频）"
        assert win.performance_diag_box.isCheckable() is True
        assert win.performance_diag_box.isChecked() is False
    finally:
        win.close()
        app.processEvents()


def test_plot_toolbar_home_restores_original_full_view():
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(800, dtype=np.float32).reshape(40, 20) / 10.0
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win.plot_data(win.data)
        app.processEvents()

        ax = win._main_plot_axes[0]
        original_xlim = tuple(float(v) for v in ax.get_xlim())
        original_ylim = tuple(float(v) for v in ax.get_ylim())

        ax.set_xlim(
            original_xlim[0] + (original_xlim[1] - original_xlim[0]) * 0.2,
            original_xlim[0] + (original_xlim[1] - original_xlim[0]) * 0.8,
        )
        ax.set_ylim(
            original_ylim[0] + (original_ylim[1] - original_ylim[0]) * 0.2,
            original_ylim[0] + (original_ylim[1] - original_ylim[0]) * 0.8,
        )
        win._capture_main_view_limits_from_axes()
        win.canvas.draw_idle()
        app.processEvents()

        current_xlim = tuple(float(v) for v in ax.get_xlim())
        current_ylim = tuple(float(v) for v in ax.get_ylim())
        assert current_xlim != original_xlim
        assert current_ylim != original_ylim

        home_action = _find_home_toolbar_action(win._main_toolbar)
        home_action.trigger()
        app.processEvents()
        app.processEvents()

        reset_ax = win._main_plot_axes[0]
        reset_xlim = tuple(float(v) for v in reset_ax.get_xlim())
        reset_ylim = tuple(float(v) for v in reset_ax.get_ylim())

        assert np.allclose(reset_xlim, original_xlim)
        assert np.allclose(reset_ylim, original_ylim)
    finally:
        win.close()
        app.processEvents()


def test_manual_roi_is_prioritized_for_auto_tune_when_present():
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(120, dtype=np.float32).reshape(10, 12) / 10.0
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win._manual_roi_values = {
            "dist_start": 2.0,
            "dist_end": 6.0,
            "time_start": 2.0,
            "time_end": 8.0,
        }
        roi_spec = win._build_auto_tune_roi_spec("prefer_crop")
        assert roi_spec["source"] == "manual"
        assert roi_spec["label"] == "手动框选 ROI"
    finally:
        win.close()
        app.processEvents()


def test_main_canvas_plain_drag_pans_like_grabbing_image(monkeypatch):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(800, dtype=np.float32).reshape(40, 20) / 10.0
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win.plot_data(win.data)
        app.processEvents()

        monkeypatch.setattr(win.canvas, "draw_idle", lambda: None)

        ax = win._main_plot_axes[0]
        original_xlim = tuple(float(v) for v in ax.get_xlim())
        original_ylim = tuple(float(v) for v in ax.get_ylim())

        press = _DummyCanvasEvent(
            button=1,
            inaxes=ax,
            x=100.0,
            y=100.0,
            xdata=original_xlim[0] + (original_xlim[1] - original_xlim[0]) * 0.3,
            ydata=original_ylim[0] + (original_ylim[1] - original_ylim[0]) * 0.3,
            dblclick=False,
            key=None,
        )
        move = _DummyCanvasEvent(
            inaxes=ax,
            x=150.0,
            y=150.0,
            xdata=original_xlim[0] + (original_xlim[1] - original_xlim[0]) * 0.5,
            ydata=original_ylim[0] + (original_ylim[1] - original_ylim[0]) * 0.5,
        )
        release = _DummyCanvasEvent(
            inaxes=ax,
            x=150.0,
            y=150.0,
            xdata=original_xlim[0] + (original_xlim[1] - original_xlim[0]) * 0.5,
            ydata=original_ylim[0] + (original_ylim[1] - original_ylim[0]) * 0.5,
            key=None,
        )

        win._on_main_canvas_press(press)
        win._on_main_canvas_motion(move)
        win._on_main_canvas_release(release)

        new_xlim = tuple(float(v) for v in ax.get_xlim())
        new_ylim = tuple(float(v) for v in ax.get_ylim())

        # Pan direction: grabbing the image means viewport moves opposite to pointer.
        # Pointer moved right -> xlim should decrease (content appears to follow finger).
        assert new_xlim[0] < original_xlim[0]
        assert new_xlim[1] < original_xlim[1]
        # For y with origin='upper', pointer moved down -> ylim values increase.
        assert new_ylim[0] > original_ylim[0]
        assert new_ylim[1] > original_ylim[1]
        assert win._manual_roi_values is None
    finally:
        win.close()
        app.processEvents()


def test_manual_roi_can_be_set_by_shift_drag(monkeypatch):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(120, dtype=np.float32).reshape(10, 12) / 10.0
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        monkeypatch.setattr(win, "plot_data", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(win.canvas, "draw_idle", lambda: None)

        ax = _DummyAxes()
        win._main_plot_axes = [ax]
        press = _DummyCanvasEvent(
            button=1,
            inaxes=ax,
            x=10.0,
            y=10.0,
            xdata=1.0,
            ydata=2.0,
            dblclick=False,
            key="shift",
        )
        move = _DummyCanvasEvent(
            inaxes=ax,
            x=40.0,
            y=60.0,
            xdata=5.0,
            ydata=8.0,
        )
        release = _DummyCanvasEvent(
            inaxes=ax,
            x=40.0,
            y=60.0,
            xdata=5.0,
            ydata=8.0,
            key="shift",
        )

        win._on_main_canvas_press(press)
        win._on_main_canvas_motion(move)
        win._on_main_canvas_release(release)

        assert win._manual_roi_values is not None
        assert win._manual_roi_values["dist_start"] == 1.0
        assert win._manual_roi_values["dist_end"] == 5.0
        assert win._manual_roi_values["time_start"] == 2.0
        assert win._manual_roi_values["time_end"] == 8.0
    finally:
        win.close()
        app.processEvents()


def test_manual_roi_can_be_cleared_by_right_click(monkeypatch):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(120, dtype=np.float32).reshape(10, 12) / 10.0
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        monkeypatch.setattr(win, "plot_data", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(win, "_capture_main_view_limits_from_axes", lambda: None)

        ax = _DummyAxes()
        win._main_plot_axes = [ax]
        win._manual_roi_values = {
            "dist_start": 1.0,
            "dist_end": 5.0,
            "time_start": 2.0,
            "time_end": 8.0,
        }

        right_click = _DummyCanvasEvent(
            button=3,
            inaxes=ax,
            x=20.0,
            y=20.0,
            xdata=2.0,
            ydata=3.0,
            dblclick=False,
            key=None,
        )

        win._on_main_canvas_press(right_click)

        assert win._manual_roi_values is None
        assert win.page_advanced.roi_status_label.text() == "手动 ROI: 未设置"
    finally:
        win.close()
        app.processEvents()


def test_apply_menu_switches_default_source_mode():
    app = _get_app()
    win = GPRGuiQt()
    try:
        win.page_basic.set_apply_source_mode("auto_tune")
        assert win.page_basic.get_apply_source_mode() == "auto_tune"
        assert win.page_basic.action_apply_manual.text() == "使用当前参数"
        assert (
            win.page_basic.action_apply_auto_tuned.text() == "使用自动调参参数（默认）"
        )

        win.page_basic.set_apply_source_mode("manual")
        assert win.page_basic.get_apply_source_mode() == "manual"
        assert win.page_basic.action_apply_manual.text() == "使用当前参数（默认）"
        assert win.page_basic.action_apply_auto_tuned.text() == "使用自动调参参数"
    finally:
        win.close()
        app.processEvents()


def test_apply_auto_tuned_default_starts_auto_tune_when_result_missing(monkeypatch):
    app = _get_app()
    win = GPRGuiQt()
    try:
        called = {}

        def fake_start_auto_tune_current_method(auto_apply_after_finish=False):
            called["auto_apply_after_finish"] = auto_apply_after_finish
            return True

        monkeypatch.setattr(
            win, "start_auto_tune_current_method", fake_start_auto_tune_current_method
        )

        win._last_auto_tune_result = None
        win.apply_method_auto_tuned_default()

        assert called["auto_apply_after_finish"] is True
        assert "正在分析当前参数" in win.page_basic._apply_source_hint_text
    finally:
        win.close()
        app.processEvents()


def test_start_auto_select_current_stage_uses_same_stage_public_methods(monkeypatch):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(120, dtype=np.float32).reshape(10, 12) / 10.0
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win.page_basic.set_method_by_key("subtracting_average_2D")

        called = {}

        class DummyThread:
            def __init__(self, *_args, **_kwargs):
                class _Signal:
                    def connect(self, *_a, **_k):
                        return None

                self.started = _Signal()
                self.finished = _Signal()

            def start(self):
                called["thread_started"] = True

            def quit(self):
                return None

            def wait(self, *_args, **_kwargs):
                return None

        class DummyWorker:
            def __init__(self, _data, method_keys, base_params_map, **kwargs):
                called["method_keys"] = method_keys
                called["base_params_map"] = base_params_map

                class _Signal:
                    def connect(self, *_a, **_k):
                        return None

                self.progress = _Signal()
                self.finished = _Signal()
                self.error = _Signal()

            def moveToThread(self, _thread):
                return None

            def run(self):
                return None

        monkeypatch.setattr(app_qt, "QThread", DummyThread)
        monkeypatch.setattr(app_qt, "AutoTuneStageWorker", DummyWorker)

        started = win.start_auto_select_current_stage()

        assert started is True
        assert called["thread_started"] is True
        assert set(called["method_keys"]) == {
            "subtracting_average_2D",
            "median_background_2D",
            "svd_bg",
            "fk_filter",
        }
    finally:
        win.close()
        app.processEvents()


def test_apply_stage_compare_choice_writes_back_best_method(monkeypatch):
    app = _get_app()
    win = GPRGuiQt()
    try:
        applied = {}
        monkeypatch.setattr(win, "_log", lambda *args, **kwargs: None)

        def fake_apply_method_params(method_key, params):
            applied["method_key"] = method_key
            applied["params"] = params

        monkeypatch.setattr(
            win.page_basic, "apply_method_params", fake_apply_method_params
        )

        win._last_auto_tune_group_result = {
            "best_method_key": "fk_filter",
            "best_method_name": "F-K cone filter",
            "best_auto_tune_result": {
                "recommended_params": {
                    "angle_low": 10,
                    "angle_high": 55,
                    "taper_width": 4,
                }
            },
        }

        win.apply_stage_compare_choice()

        assert applied["method_key"] == "fk_filter"
        assert applied["params"]["angle_low"] == 10
        assert win.page_basic.get_apply_source_mode() == "auto_tune"
    finally:
        win.close()
        app.processEvents()


def test_stage_compare_result_summary_is_human_readable():
    app = _get_app()
    win = GPRGuiQt()
    try:
        win.page_auto_tune.set_stage_compare_result(
            {
                "stage": "background",
                "best_method_key": "fk_filter",
                "best_method_name": "F-K cone filter",
                "outer_score": 1.2345,
                "outer_reason": "优先比较背景压制改善、结构保真和目标频带保真。",
                "candidates": [
                    {
                        "method_key": "fk_filter",
                        "method_name": "F-K cone filter",
                        "outer_score": 1.2345,
                        "outer_reason": "说明A",
                        "champion_profile": "平衡档",
                    },
                    {
                        "method_key": "subtracting_average_2D",
                        "method_name": "平均背景抑制",
                        "outer_score": 1.1000,
                        "outer_reason": "说明B",
                        "champion_profile": "平衡档",
                    },
                ],
            }
        )

        assert "Stage：background" in win.page_auto_tune.stage_compare_label.text()
        summary = win.page_auto_tune.stage_compare_summary.toPlainText()
        assert "推荐方法: F-K cone filter" in summary
        assert "比较结果:" in summary
        assert "平均背景抑制" in summary
        assert win.page_auto_tune.btn_apply_stage_choice.isEnabled() is True
    finally:
        win.close()
        app.processEvents()


def test_auto_tune_finished_auto_applies_recommended_profile_when_pending(monkeypatch):
    app = _get_app()
    win = GPRGuiQt()
    try:
        applied = {}
        monkeypatch.setattr(win, "_set_busy", lambda *args, **kwargs: None)
        monkeypatch.setattr(win, "_log", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            win,
            "apply_method_from_profile",
            lambda profile_key: applied.setdefault("profile", profile_key),
        )

        win._pending_apply_after_auto_tune = True
        result = {
            "method_key": "dewow",
            "method_name": "低频漂移矫正（Dewow）",
            "recommended_profile": "balanced",
            "recommended_params": {"window": 23},
            "profiles": {
                "balanced": {
                    "label": "平衡档",
                    "params": {"window": 23},
                    "score": 1.0,
                    "metrics": {},
                    "reason": "test",
                }
            },
            "coarse_trials": [],
            "fine_trials": [],
            "all_trials": [],
            "best_score": 1.0,
            "best_params": {"window": 23},
            "roi_info": {"label": "全图"},
        }

        win._on_auto_tune_finished(result)

        assert applied["profile"] == "balanced"
        assert win._pending_apply_after_auto_tune is False
    finally:
        win.close()
        app.processEvents()


def test_basic_flow_parses_bool_params_from_registry_contract():
    app = _get_app()
    win = GPRGuiQt()
    try:
        win.page_basic.set_method_by_key("ccbs")
        edit, _meta = win.page_basic.param_vars["use_custom_ref"]

        edit.setText("true")
        params = win.page_basic.get_current_params()
        assert params["use_custom_ref"] is True

        edit.setText("false")
        params = win.page_basic.get_current_params()
        assert params["use_custom_ref"] is False
    finally:
        win.close()
        app.processEvents()


def test_apply_single_method_separates_preview_and_commit_payload(monkeypatch):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(16, dtype=np.float32).reshape(4, 4)
        actual = raw + 10.0
        preview = np.arange(6, dtype=np.float32).reshape(2, 3)

        def fake_run_processing_method(_data, _method_id, _params):
            return actual, {
                "display_data": preview,
                "display_header_info_updates": {"total_time_ns": 12.0},
                "header_info_updates": {"total_time_ns": 24.0},
            }

        monkeypatch.setattr(app_qt, "run_processing_method", fake_run_processing_method)

        execution = win._apply_single_method(
            raw,
            "dewow",
            {"window": 23},
            header_info={"total_time_ns": 48.0},
            trace_metadata={},
        )

        assert np.array_equal(execution["result_data"], actual)
        assert np.array_equal(execution["preview_data"], preview)
        assert execution["result_header_info"]["total_time_ns"] == 24.0
        assert execution["preview_header_info"]["total_time_ns"] == 12.0
        assert execution["result_header_info"]["a_scan_length"] == 4
        assert execution["preview_header_info"]["a_scan_length"] == 2
    finally:
        win.close()
        app.processEvents()


def test_workbench_commit_prefers_preview_commit_payload(monkeypatch):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(16, dtype=np.float32).reshape(4, 4)
        preview = np.arange(6, dtype=np.float32).reshape(2, 3)
        committed = raw + 100.0

        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win.page_workbench.set_preview_result(
            preview,
            title="预览: demo",
            header_info={"total_time_ns": 12.0},
            commit_data=committed,
            commit_header_info={"total_time_ns": 24.0},
        )

        monkeypatch.setattr(win, "_mark_data_changed", lambda: None)
        monkeypatch.setattr(win, "_update_current_compare_snapshot", lambda: None)
        monkeypatch.setattr(win, "_update_empty_state_and_brief", lambda: None)
        monkeypatch.setattr(win, "plot_data", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(win, "_set_last_run_summary", lambda *args, **kwargs: None)
        monkeypatch.setattr(win, "_log", lambda *_args, **_kwargs: None)

        win._on_workbench_save_result()

        assert np.array_equal(win.shared_data.current_data, committed)
        assert win.shared_data.header_info["total_time_ns"] == 24.0
    finally:
        win.close()
        app.processEvents()


def test_placeholder_methods_are_hidden_from_public_lists():
    public_keys = set(get_public_method_keys())
    assert "rpca_placeholder" not in public_keys
    assert "wnnm_placeholder" not in public_keys


def test_method_browser_processing_section_matches_public_registry():
    app = _get_app()
    browser = MethodBrowserTree()
    try:
        processing_ids = []
        for i in range(browser.tree.topLevelItemCount()):
            top = browser.tree.topLevelItem(i)
            if top.data(0, Qt.ItemDataRole.UserRole) == "cat:processing":
                for j in range(top.childCount()):
                    group = top.child(j)
                    for k in range(group.childCount()):
                        processing_ids.append(
                            group.child(k).data(0, Qt.ItemDataRole.UserRole)
                        )
        assert set(processing_ids) == set(get_public_method_keys())
        assert len(processing_ids) == len(get_public_method_keys())
    finally:
        browser.close()
        app.processEvents()


def test_basic_flow_method_combo_shows_category_prefix():
    app = _get_app()
    win = GPRGuiQt()
    try:
        first_key = win.page_basic.method_keys[0]
        first_text = win.page_basic.method_combo.itemText(0)
        assert first_text.startswith(f"[{get_method_category_label(first_key)}]")
    finally:
        win.close()
        app.processEvents()


def test_public_method_order_follows_processing_chain():
    keys = get_public_method_keys()
    expected_sequence = [
        "set_zero_time",
        "dewow",
        "subtracting_average_2D",
        "median_background_2D",
        "svd_bg",
        "ccbs",
        "sliding_avg",
        "fk_filter",
        "sec_gain",
        "compensatingGain",
        "agcGain",
        "hankel_svd",
        "svd_subspace",
        "wavelet_svd",
        "running_average_2D",
        "stolt_migration",
        "kirchhoff_migration",
        "time_to_depth",
    ]

    positions = [keys.index(key) for key in expected_sequence if key in keys]
    assert positions == sorted(positions)


def test_hankel_svd_defaults_use_auto_or_preview_safe_overrides(tmp_path):
    params = PROCESSING_METHODS["hankel_svd"]["params"]
    defaults = {item["name"]: item.get("default") for item in params}
    labels = {item["name"]: item.get("label", "") for item in params}
    manager = WorkflowTemplateManager(config_dir=str(tmp_path / "workflow_templates"))
    high_focus_template = next(
        item for item in manager.get_preset_templates() if item["name"] == "高聚焦"
    )
    high_focus_hankel = next(
        item for item in high_focus_template["methods"] if item["method_id"] == "hankel_svd"
    )

    assert defaults["window_length"] == 0
    assert defaults["rank"] == 0
    assert "bounded/literature" in labels["window_length"]
    assert "bounded/literature" in labels["rank"]
    assert "quick_preview" not in GUI_PRESETS_V1
    assert "hankel_svd" not in GUI_PRESETS_V1["denoise_first"]["method_params"]
    assert GUI_PRESETS_V1["detail_first"]["method_params"]["hankel_svd"] == {
        "window_length": 0,
        "rank": 0,
    }
    assert RECOMMENDED_RUN_PROFILES["hankel_denoise"]["method_params"]["hankel_svd"] == {
        "window_length": 0,
        "rank": 0,
    }
    assert high_focus_hankel["params"] == {"window_length": 0, "rank": 0}


def test_svd_denoising_defaults_match_current_gpr_recommendation():
    svd_params = PROCESSING_METHODS["svd_subspace"]["params"]
    svd_defaults = {item["name"]: item.get("default") for item in svd_params}
    assert svd_defaults["rank_start"] == 1
    assert svd_defaults["rank_end"] == 20

    wavelet_params = PROCESSING_METHODS["wavelet_svd"]["params"]
    wavelet_defaults = {item["name"]: item.get("default") for item in wavelet_params}
    assert wavelet_defaults["wavelet"] == "db4"
    assert wavelet_defaults["levels"] == 2
    assert wavelet_defaults["threshold"] == 0.05
    assert wavelet_defaults["rank_start"] == 1
    assert wavelet_defaults["rank_end"] == 20


def test_recommended_profiles_prefer_svd_and_hankel_denoising():
    robust_order = RECOMMENDED_RUN_PROFILES["robust_imaging"]["order"]
    high_focus_order = RECOMMENDED_RUN_PROFILES["high_focus"]["order"]

    assert "svd_subspace" in robust_order
    assert "hankel_svd" not in robust_order
    assert "hankel_svd" in high_focus_order
    assert "wavelet_svd" not in high_focus_order

    assert GUI_PRESETS_V1["denoise_first"]["method_params"]["svd_subspace"] == {
        "rank_start": 1,
        "rank_end": 20,
    }
    assert GUI_PRESETS_V1["stolt_focus_first"]["method_params"]["wavelet_svd"] == {
        "wavelet": "db4",
        "levels": 2,
        "threshold": 0.05,
        "rank_start": 1,
        "rank_end": 20,
    }


def test_run_default_pipeline_uses_new_five_step_order_and_current_source_mode(
    monkeypatch,
):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(120, dtype=np.float32).reshape(10, 12) / 10.0
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        app.processEvents()

        captured = {}
        monkeypatch.setattr(win, "_push_history", lambda: None)
        monkeypatch.setattr(win, "_log", lambda *_args, **_kwargs: None)

        def fake_start_processing_worker(tasks, run_type="single", **kwargs):
            captured["tasks"] = tasks
            captured["run_type"] = run_type
            captured["kwargs"] = kwargs

        monkeypatch.setattr(
            win, "_start_processing_worker", fake_start_processing_worker
        )

        win.page_basic.set_apply_source_mode("auto_tune")
        win.run_default_pipeline()

        assert captured["run_type"] == "pipeline"
        assert [task["method_key"] for task in captured["tasks"]] == [
            "set_zero_time",
            "dewow",
            "subtracting_average_2D",
            "agcGain",
            "running_average_2D",
        ]
        assert {task["param_source_mode"] for task in captured["tasks"]} == {
            "auto_tune"
        }
    finally:
        win.close()
        app.processEvents()


def test_run_default_pipeline_manual_mode_uses_visible_and_saved_params(monkeypatch):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(120, dtype=np.float32).reshape(10, 12) / 10.0
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        app.processEvents()

        captured = {}
        monkeypatch.setattr(win, "_push_history", lambda: None)
        monkeypatch.setattr(win, "_log", lambda *_args, **_kwargs: None)

        win.page_basic.set_method_by_key("dewow")
        win.page_basic.set_apply_source_mode("manual")
        monkeypatch.setattr(
            win.page_basic,
            "get_current_params",
            lambda: {"window": 77},
        )
        win._method_param_overrides["agcGain"] = {"window": 13}

        def fake_start_processing_worker(tasks, run_type="single", **kwargs):
            captured["tasks"] = tasks
            captured["run_type"] = run_type
            captured["kwargs"] = kwargs

        monkeypatch.setattr(
            win, "_start_processing_worker", fake_start_processing_worker
        )

        win.run_default_pipeline()

        assert captured["run_type"] == "pipeline"
        assert {task["param_source_mode"] for task in captured["tasks"]} == {"manual"}

        by_key = {task["method_key"]: task for task in captured["tasks"]}
        assert by_key["dewow"]["params"]["window"] == 77
        assert by_key["agcGain"]["params"]["window"] == 13
    finally:
        win.close()
        app.processEvents()


def test_processing_worker_uses_auto_tuned_params_per_step(monkeypatch):
    _get_app()
    raw = np.arange(24, dtype=np.float32).reshape(6, 4)
    tasks = [
        {
            "method_key": "set_zero_time",
            "method": PROCESSING_METHODS["set_zero_time"],
            "params": {"new_zero_time": 99.0},
            "out_dir": ".",
            "param_source_mode": "auto_tune",
        },
        {
            "method_key": "dewow",
            "method": PROCESSING_METHODS["dewow"],
            "params": {"window": 999},
            "out_dir": ".",
            "param_source_mode": "auto_tune",
        },
    ]

    tuned_params = {
        "set_zero_time": {"new_zero_time": 5.0},
        "dewow": {"window": 23},
    }
    seen_params = []

    def fake_auto_tune_method(
        data,
        method_key,
        candidate_params=None,
        header_info=None,
        trace_metadata=None,
        base_params=None,
        roi_spec=None,
        search_mode="standard",
        progress_callback=None,
        cancel_checker=None,
    ):
        params = tuned_params[method_key]
        return {
            "method_key": method_key,
            "recommended_profile": "balanced",
            "recommended_params": dict(params),
            "profiles": {
                "balanced": {
                    "label": "平衡档",
                    "params": dict(params),
                    "score": 1.0,
                    "metrics": {},
                    "reason": "test",
                }
            },
        }

    def fake_run_processing_method(data, method_key, params, cancel_checker=None):
        seen_params.append((method_key, dict(params)))
        return np.array(data, copy=True), {}

    monkeypatch.setattr(app_qt, "auto_tune_method", fake_auto_tune_method)
    monkeypatch.setattr(app_qt, "run_processing_method", fake_run_processing_method)

    worker = app_qt.ProcessingWorker(raw, tasks, execution_mode="sequential")
    captured = {}
    worker.finished.connect(lambda result: captured.update(result))

    worker.run()

    assert [item[0] for item in seen_params] == ["set_zero_time", "dewow"]
    assert seen_params[0][1]["new_zero_time"] == 5.0
    assert seen_params[0][1]["new_zero_time"] != 99.0
    assert seen_params[1][1]["window"] == 23
    assert seen_params[1][1]["window"] != 999
    assert len(captured["outputs"]) == 2


def test_workflow_quick_presets_align_with_current_denoising_preference():
    robust_methods = [
        item["method_id"] for item in QUICK_PRESETS["robust_imaging"]["methods"]
    ]
    high_focus_methods = [
        item["method_id"] for item in QUICK_PRESETS["high_focus"]["methods"]
    ]

    assert "svd_subspace" in robust_methods
    assert "hankel_svd" not in robust_methods
    assert "hankel_svd" in high_focus_methods
    assert "wavelet_svd" not in high_focus_methods


def test_quality_page_exposes_report_and_snapshot_actions():
    app = _get_app()
    win = GPRGuiQt()
    try:
        assert win.page_quality.btn_generate_report.text() == "生成报告"
        assert win.page_quality.btn_export_quality_snapshot.text() == "导出质量快照"
        assert win.page_quality.btn_generate_report.toolTip()
        assert win.page_quality.btn_export_quality_snapshot.toolTip()
    finally:
        win.close()
        app.processEvents()


def test_processing_worker_compacts_large_meta_arrays():
    _get_app()
    raw = np.ones((4, 3), dtype=np.float32)
    task = {
        "method_key": "sec_gain",
        "method": PROCESSING_METHODS["sec_gain"],
        "params": {"gain_min": 1.0, "gain_max": 4.0, "power": 1.0},
        "out_dir": ".",
    }
    worker = app_qt.ProcessingWorker(raw, [task], execution_mode="sequential")
    captured = {}
    worker.finished.connect(lambda result: captured.update(result))

    worker.run()

    assert captured["outputs"]
    output = captured["outputs"][0]
    assert np.array_equal(output["data"], captured["final_data"])
    assert "gain_curve" not in (output.get("meta") or {})
