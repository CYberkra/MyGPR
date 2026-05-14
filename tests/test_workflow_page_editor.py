#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow page editor regression tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QComboBox

from core.app_paths import get_workflow_templates_dir
from core.workflow_data import WorkflowConfigManager
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
        assert page.btn_save_template.text() == "存模板"
        assert page.btn_restore_default.text() == "默认"
        assert page.btn_add_step.text() == "添加"

        _select_method(page, "frequency_filter_1d")
        app.processEvents()
        label = page.step_list.currentItem().text()

        assert "基础迹线域校正" in label
        assert "一维频域滤波" in label
        assert "filter_type=" not in label
    finally:
        page.close()
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
