#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow page editor regression tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEvent, QPointF
from PyQt6.QtWidgets import QApplication, QAbstractSpinBox, QComboBox, QLineEdit

from core.app_paths import get_workflow_templates_dir
from core.workflow_data import WorkflowConfigManager, build_default_workflow_config
from ui.workflow_canvas_cards import WorkflowNodeCard, WorkflowNodeProxy
from ui.workflow_canvas_cards import WorkflowCanvasView
from ui.workflow_canvas_preview import BscanPreviewCard, _downsample_for_preview
from ui.gui_workflow_page import WorkflowPage


def _get_app() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


def _select_method(page: WorkflowPage, method_id: str) -> int:
    for row, method in enumerate(page.config.methods):
        if method.method_id == method_id:
            page.step_list.setCurrentRow(row)
            return row
    raise AssertionError(f"method not found in workflow: {method_id}")


def test_choice_params_render_as_combo_box():
    app = _get_app()
    page = WorkflowPage()
    try:
        _select_method(page, "frequency_filter_1d")
        app.processEvents()

        control = page._param_controls.get("filter_type")

        assert isinstance(control, QComboBox)
        assert control.currentData() == "bandpass"
        assert control.count() >= 4
    finally:
        page.close()
        app.processEvents()


def test_step_editor_add_duplicate_remove_and_run_scopes():
    app = _get_app()
    page = WorkflowPage()
    emitted: list[tuple[list, bool]] = []
    page.workflow_run_requested.connect(lambda methods, realtime: emitted.append((methods, realtime)))
    try:
        page.realtime_check.setChecked(False)
        page.step_list.setCurrentRow(1)
        app.processEvents()
        initial_count = page.step_list.count()

        page.add_step_after_current()
        app.processEvents()

        assert page.step_list.count() == initial_count + 1
        assert len(page.config.methods) == initial_count + 1
        added_row = page.step_list.currentRow()
        assert added_row == 2
        assert page.config.methods[added_row].order == added_row

        page.duplicate_current_step()
        app.processEvents()

        assert page.step_list.count() == initial_count + 2
        assert len(page.config.methods) == initial_count + 2
        duplicated_row = page.step_list.currentRow()
        assert duplicated_row == added_row + 1
        assert page.config.methods[duplicated_row].method_id == page.config.methods[added_row].method_id

        page.remove_current_step()
        app.processEvents()

        assert page.step_list.count() == initial_count + 1
        assert len(page.config.methods) == initial_count + 1
        assert [method.order for method in page.config.methods] == list(range(initial_count + 1))

        page.step_list.setCurrentRow(1)
        page.request_selected_run()
        app.processEvents()

        assert emitted
        selected_methods, selected_realtime = emitted[-1]
        assert selected_realtime is False
        assert len(selected_methods) == 1
        assert selected_methods[0].method_id == page.config.methods[1].method_id

        page.step_list.setCurrentRow(2)
        expected_tail = [
            method.method_id
            for method in page.config.methods[2:]
            if method.enabled and not method.hidden
        ]
        page.request_run_from_current()
        app.processEvents()

        tail_methods, tail_realtime = emitted[-1]
        assert tail_realtime is False
        assert [method.method_id for method in tail_methods] == expected_tail
    finally:
        page.close()
        app.processEvents()


def test_compact_vertical_layout_uses_short_actions_and_step_labels():
    app = _get_app()
    page = WorkflowPage()
    try:
        assert page.btn_run_all.text() == "全链"
        assert page.btn_run_from_current.text() == "后续"
        assert page.btn_run_selected.text() == "当前"
        assert page.btn_save_live.text() == "保存"
        assert page.btn_new_template.text() == "新"
        assert page.btn_duplicate_template.text() == "副本"
        assert page.btn_save_template.text() == "存模"
        assert page.btn_restore_default.text() == "默认"
        assert page.btn_add_step.text() == "添加"
        assert page.detail_box.title() == "选中步骤参数"
        assert page.detail_box.isHidden()
        assert page.step_list.isHidden()
        assert isinstance(page.workflow_canvas, WorkflowCanvasView)
        assert page.workflow_canvas._scene.proxies

        _select_method(page, "frequency_filter_1d")
        app.processEvents()
        label = page.step_list.currentItem().text()

        assert "基础迹线域校正" in label
        assert "一维频域滤波" in label
        assert "filter_type=" not in label
    finally:
        page.close()
        app.processEvents()


def test_workflow_canvas_node_selection_and_actions_sync_hidden_list():
    app = _get_app()
    page = WorkflowPage()
    emitted: list[tuple[list, bool]] = []
    page.workflow_run_requested.connect(lambda methods, realtime: emitted.append((methods, realtime)))
    try:
        target_row = _select_method(page, "sec_gain")
        app.processEvents()

        assert page.workflow_canvas._current_row == target_row
        assert any(proxy.row == target_row for proxy in page.workflow_canvas._scene.proxies)

        page.workflow_canvas.node_selected.emit(0)
        app.processEvents()
        assert page.step_list.currentRow() == 0

        page.workflow_canvas.run_node_requested.emit(0)
        app.processEvents()
        assert emitted
        assert emitted[-1][1] is False
        assert len(emitted[-1][0]) == 1
        assert emitted[-1][0][0].method_id == page.config.methods[0].method_id
    finally:
        page.close()
        app.processEvents()


def test_workflow_canvas_zoom_lod_switches_card_compact_mode():
    app = _get_app()
    canvas = WorkflowCanvasView()
    try:
        config = build_default_workflow_config("high_quality_uav_gpr")
        canvas.set_methods(config.methods)
        app.processEvents()

        cards = [
            proxy.widget()
            for proxy in canvas._scene.proxies
            if isinstance(proxy.widget(), WorkflowNodeCard)
        ]
        assert cards
        assert not any(card.compact for card in cards)

        canvas.resetTransform()
        canvas.scale(0.5, 0.5)
        canvas._apply_zoom_lod(force=True)
        app.processEvents()
        assert all(card.compact for card in cards)

        canvas.resetTransform()
        canvas.scale(1.0, 1.0)
        canvas._apply_zoom_lod(force=True)
        app.processEvents()
        assert not any(card.compact for card in cards)
    finally:
        canvas.close()
        app.processEvents()


def test_workflow_canvas_preview_node_updates_from_output_data():
    app = _get_app()
    canvas = WorkflowCanvasView()
    try:
        config = build_default_workflow_config("high_quality_uav_gpr")
        canvas.set_methods(config.methods)
        canvas.set_preview_data(np.arange(64, dtype=float).reshape(8, 8), "Test B-scan")
        app.processEvents()

        preview_cards = [
            proxy.widget()
            for proxy in canvas._scene.proxies
            if isinstance(proxy.widget(), BscanPreviewCard)
        ]
        assert len(preview_cards) == 1
        assert preview_cards[0].data_shape == (8, 8)
        assert preview_cards[0].source_label is not None
        assert "Test B-scan" in preview_cards[0].source_label.text()

        canvas.resetTransform()
        canvas.scale(0.5, 0.5)
        canvas._apply_zoom_lod(force=True)
        app.processEvents()
        assert preview_cards[0].compact is True
    finally:
        canvas.close()
        app.processEvents()


def test_workflow_canvas_update_node_defers_scene_rebuild():
    app = _get_app()
    canvas = WorkflowCanvasView()
    calls: list[str] = []
    try:
        config = build_default_workflow_config("high_quality_uav_gpr")
        canvas.set_methods(config.methods)

        canvas._rebuild = lambda: calls.append("rebuild")
        canvas.update_node(0)

        assert calls == []
        assert canvas._rebuild_pending is True
        app.processEvents()
        assert calls == ["rebuild"]
        assert canvas._rebuild_pending is False
    finally:
        canvas.close()
        app.processEvents()


def test_workflow_bscan_preview_downsamples_large_arrays():
    raw = np.arange(2_000 * 3_000, dtype=np.float32).reshape(2_000, 3_000)

    preview = _downsample_for_preview(raw, max_rows=900, max_cols=1400)

    assert preview.shape[0] <= 900
    assert preview.shape[1] <= 1400
    assert preview.dtype == raw.dtype


def test_workflow_node_card_swallows_wheel_on_embedded_editors():
    app = _get_app()
    config = build_default_workflow_config("high_quality_uav_gpr")
    method = None
    for candidate in config.methods:
        probe = WorkflowNodeCard(0, candidate)
        try:
            if probe.findChild(QAbstractSpinBox) is not None:
                method = candidate
                break
        finally:
            probe.close()
    assert method is not None

    card = WorkflowNodeCard(0, method)
    try:
        spin = card.findChild(QAbstractSpinBox)
        assert spin is not None

        wheel_event = QEvent(QEvent.Type.Wheel)
        assert card.eventFilter(spin, wheel_event) is True
        assert wheel_event.isAccepted()

        editor = spin.findChild(QLineEdit)
        assert editor is not None
        editor_wheel_event = QEvent(QEvent.Type.Wheel)
        assert card.eventFilter(editor, editor_wheel_event) is True
        assert editor_wheel_event.isAccepted()

        combo = card.findChild(QComboBox)
        assert combo is not None
        combo_wheel_event = QEvent(QEvent.Type.Wheel)
        assert card.eventFilter(combo, combo_wheel_event) is True
        assert combo_wheel_event.isAccepted()
    finally:
        card.close()
        app.processEvents()


def test_workflow_canvas_view_drag_hit_testing_keeps_controls_interactive():
    app = _get_app()
    canvas = WorkflowCanvasView()
    try:
        config = build_default_workflow_config("high_quality_uav_gpr")
        canvas.set_methods(config.methods)
        app.processEvents()

        proxy = canvas._scene.proxies[0]
        assert isinstance(proxy, WorkflowNodeProxy)

        card = proxy.widget()
        assert isinstance(card, WorkflowNodeCard)

        noninteractive_pos = proxy.mapToScene(QPointF(8, 8))
        assert canvas._is_interactive_card_target(proxy, noninteractive_pos) is False

        combo = card.findChild(QComboBox)
        assert combo is not None
        combo_center = combo.geometry().center()
        interactive_pos = proxy.mapToScene(QPointF(combo_center))
        assert canvas._is_interactive_card_target(proxy, interactive_pos) is True
    finally:
        canvas.close()
        app.processEvents()


def test_workflow_config_manager_uses_user_writable_template_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    manager = WorkflowConfigManager()
    expected_root = Path(get_workflow_templates_dir())
    config_dir = Path(manager.config_dir)

    assert config_dir.is_dir()
    assert config_dir.parent == expected_root
    assert config_dir.name == "workflow_configs"
    assert str(config_dir).startswith(str(tmp_path))
