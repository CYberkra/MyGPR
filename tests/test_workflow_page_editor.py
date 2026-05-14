#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow page editor regression tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEvent, QPointF
from PyQt6.QtWidgets import QApplication, QAbstractSpinBox, QCheckBox, QComboBox, QLineEdit, QSlider, QToolButton

from core.app_paths import get_workflow_templates_dir
from core.workflow_data import WorkflowConfigManager, WorkflowMethod, build_default_workflow_config
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
        assert page.btn_run_all.text() == "Run All"
        assert page.btn_run_from_current.text() == "Run From"
        assert page.btn_run_selected.text() == "Run Selected"
        assert page.btn_save_live.text() == "Save"
        assert page.btn_validate.text() == "Validate"
        assert page.template_menu_button.text() == "模板 ▾"
        assert page.project_panel.title() == "Project / Data"
        assert page.palette_panel.title() == "Node Library"
        assert page.inspector_box.title() == "Inspector"
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


def test_workflow_canvas_zoom_lod_switches_to_mini_nodes_without_rebuilding_widgets():
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
        proxies = [proxy for proxy in canvas._scene.proxies if proxy.row >= 0]
        assert not any(proxy.mini_item.isVisible() for proxy in proxies)
        assert all(proxy.widget().isVisible() for proxy in proxies)

        canvas.resetTransform()
        canvas.scale(0.5, 0.5)
        canvas._apply_zoom_lod(force=True)
        app.processEvents()
        assert all(proxy.mini_item.isVisible() for proxy in proxies)
        assert not any(proxy.widget().isVisible() for proxy in proxies)

        canvas.resetTransform()
        canvas.scale(1.0, 1.0)
        canvas._apply_zoom_lod(force=True)
        app.processEvents()
        assert not any(proxy.mini_item.isVisible() for proxy in proxies)
        assert all(proxy.widget().isVisible() for proxy in proxies)
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
        assert preview_cards[0].isVisible() is True
    finally:
        canvas.close()
        app.processEvents()


def test_workflow_canvas_node_ports_and_edges_follow_proxy_moves():
    app = _get_app()
    canvas = WorkflowCanvasView()
    try:
        config = build_default_workflow_config("high_quality_uav_gpr")
        canvas.set_methods(config.methods)
        app.processEvents()

        algorithm_proxies = [proxy for proxy in canvas._scene.proxies if proxy.row >= 0]
        assert algorithm_proxies
        assert all(proxy.input_port is not None for proxy in algorithm_proxies)
        assert all(proxy.output_port is not None for proxy in algorithm_proxies)
        assert len(canvas._scene.edges) == max(0, len(canvas._scene.proxies) - 1)

        first_edge = canvas._scene.edges[0]
        first_path = first_edge.path()
        first_start = first_path.elementAt(0)
        expected_start = canvas._scene.proxies[0].output_port.scene_anchor()
        assert first_start.x == expected_start.x()
        assert first_start.y == expected_start.y()

        edge_count = len(canvas._scene.edges)
        edge_ids = [id(edge) for edge in canvas._scene.edges]
        old_path = first_edge.path()
        canvas._scene.proxies[0].setPos(canvas._scene.proxies[0].pos() + QPointF(24, 16))
        app.processEvents()

        assert len(canvas._scene.edges) == edge_count
        assert [id(edge) for edge in canvas._scene.edges] == edge_ids
        assert first_edge.path() != old_path
    finally:
        canvas.close()
        app.processEvents()


def test_workflow_canvas_output_drag_to_input_creates_replaceable_link():
    app = _get_app()
    canvas = WorkflowCanvasView()
    try:
        config = build_default_workflow_config("high_quality_uav_gpr")
        canvas.set_methods(config.methods[:3])
        app.processEvents()

        first = canvas._scene.proxies[0]
        second = canvas._scene.proxies[1]
        third = canvas._scene.proxies[2]
        original_link_count = len(canvas.current_links())

        canvas._drag_source_port = first.output_port
        canvas._finish_temp_edge(third.input_port)
        app.processEvents()

        links = canvas.current_links()
        assert len(links) == original_link_count
        assert any(link.from_node == first.node_id and link.to_node == third.node_id for link in links)
        assert not any(link.to_node == third.node_id and link.from_node == second.node_id for link in links)
    finally:
        canvas.close()
        app.processEvents()


def test_workflow_canvas_edge_delete_removes_link_and_edge_item():
    app = _get_app()
    canvas = WorkflowCanvasView()
    try:
        config = build_default_workflow_config("high_quality_uav_gpr")
        canvas.set_methods(config.methods[:3])
        app.processEvents()

        edge = canvas._scene.edges[0]
        link_key = canvas._scene.link_key(edge.link)
        canvas._scene.remove_edge(edge)
        app.processEvents()

        assert all(canvas._scene.link_key(link) != link_key for link in canvas.current_links())
        assert all(canvas._scene.link_key(item.link) != link_key for item in canvas._scene.edges)
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

        slider = card.findChild(QSlider)
        assert slider is not None
        slider_wheel_event = QEvent(QEvent.Type.Wheel)
        assert card.eventFilter(slider, slider_wheel_event) is True
        assert slider_wheel_event.isAccepted()
    finally:
        card.close()
        app.processEvents()


def test_workflow_node_card_numeric_params_expose_slider():
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
        slider = card.findChild(QSlider)
        assert spin is not None
        assert slider is not None
        assert spin.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
        assert bool(spin.property("workflowWheelGuard"))
        assert bool(slider.property("workflowWheelGuard"))
    finally:
        card.close()
        app.processEvents()


def test_workflow_node_card_bool_param_does_not_duplicate_label_text():
    app = _get_app()
    method = WorkflowMethod(
        category="denoise",
        stage_id="denoise",
        method_id="hilbert_envelope",
        params={"normalize": True, "log_compress": False},
    )
    card = WorkflowNodeCard(0, method)
    try:
        checkbox = card.findChild(QCheckBox)
        assert checkbox is not None
        assert checkbox.text() == ""
    finally:
        card.close()
        app.processEvents()


def test_node_context_menu_actions_and_eye_toggle_skip_hidden_step():
    app = _get_app()
    page = WorkflowPage()
    emitted: list[tuple[list, bool]] = []
    page.workflow_run_requested.connect(lambda methods, realtime: emitted.append((methods, realtime)))
    try:
        page.realtime_check.setChecked(False)
        page.step_list.setCurrentRow(0)
        app.processEvents()

        proxy = page.workflow_canvas._scene.proxies[0]
        card = proxy.widget()
        assert isinstance(card, WorkflowNodeCard)
        menu = page.workflow_canvas._build_node_context_menu(proxy)
        action_texts = [action.text() for action in menu.actions() if action.text()]
        assert "运行此节点" in action_texts
        assert "从此节点运行" in action_texts
        assert "Open Tuning Lab" in action_texts
        assert "Apply Best Params" in action_texts
        assert "Benchmark This Node" in action_texts
        assert "复制节点" in action_texts
        assert "删除节点" in action_texts
        assert "添加预览节点" in action_texts

        eye = card.findChild(QToolButton, "eyeButton")
        assert eye is not None
        eye.click()
        app.processEvents()

        assert page.config.methods[0].hidden is True
        assert "隐藏" in page.step_list.item(0).text()
        assert page.workflow_canvas._scene.proxies
        assert page.workflow_canvas._scene.proxies[0].opacity() < 0.6

        page.step_list.setCurrentRow(0)
        page.request_run_from_current()
        app.processEvents()

        assert emitted
        assert all(method.method_id != page.config.methods[0].method_id for method in emitted[-1][0])
    finally:
        page.close()
        app.processEvents()


def test_workflow_project_panel_updates_data_state_and_import_signals():
    app = _get_app()
    page = WorkflowPage()
    imports: list[str] = []
    sidecars: list[str] = []
    page.import_raw_requested.connect(lambda: imports.append("raw"))
    page.import_sidecar_requested.connect(sidecars.append)
    try:
        assert page.project_panel.title() == "Project / Data"
        assert page.palette_panel.title() == "Node Library"
        assert page.inspector_box.title() == "Inspector"

        page.set_project_data_state(
            file_path=r"C:\data\line.csv",
            shape=(501, 2378),
            metadata_status="已同步",
        )
        assert "line.csv" in page.project_file_label.text()
        assert "501 samples" in page.project_shape_label.text()
        assert "已同步" in page.project_metadata_label.text()
        assert "501 samples" in page.qc_label.text()

        page.btn_import_raw.click()
        page.btn_import_rtk.click()
        page.btn_import_imu.click()
        page.btn_import_agl.click()
        app.processEvents()
        assert imports == ["raw"]
        assert sidecars == ["rtk", "imu", "altimeter"]
    finally:
        page.close()
        app.processEvents()


def test_tuning_lab_toolbar_emits_selected_workflow_method():
    app = _get_app()
    page = WorkflowPage()
    emitted = []
    page.tuning_lab_requested.connect(emitted.append)
    try:
        page.step_list.setCurrentRow(0)
        page.btn_open_tuning_lab.click()
        app.processEvents()
        assert emitted
        assert emitted[-1].method_id == page.config.methods[0].method_id
    finally:
        page.close()
        app.processEvents()


def test_canvas_context_menu_exposes_add_node_groups_and_palette_adds_node():
    app = _get_app()
    page = WorkflowPage()
    try:
        menu = page.workflow_canvas._build_canvas_context_menu(QPointF(20, 20))
        action_texts = [action.text() for action in menu.actions() if action.text()]
        assert "添加节点" in action_texts
        assert "适配全部节点" in action_texts
        assert page.palette_list.count() > 0

        initial = len(page.config.methods)
        for row in range(page.palette_list.count()):
            item = page.palette_list.item(row)
            if item.data(0x0100) is not None:  # Qt.UserRole numeric value
                page._on_palette_item_double_clicked(item)
                break
        app.processEvents()
        assert len(page.config.methods) == initial + 1
    finally:
        page.close()
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

        slider = card.findChild(QSlider)
        assert slider is not None
        slider_center = slider.mapTo(card, slider.rect().center())
        slider_pos = proxy.mapToScene(QPointF(slider_center))
        assert canvas._is_interactive_card_target(proxy, slider_pos) is True
    finally:
        canvas.close()
        app.processEvents()


def test_workflow_canvas_resize_handle_changes_node_size_and_ports():
    app = _get_app()
    canvas = WorkflowCanvasView()
    try:
        config = build_default_workflow_config("high_quality_uav_gpr")
        canvas.set_methods(config.methods)
        app.processEvents()

        proxy = canvas._scene.proxies[0]
        old_width = proxy.widget().width()
        old_output = proxy.output_port.scene_anchor()

        proxy.apply_size(old_width + 70, proxy.widget().height() + 20)
        proxy.update_port_positions()
        canvas._scene.update_edges()

        assert proxy.widget().width() > old_width
        assert proxy.output_port.scene_anchor() != old_output
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
