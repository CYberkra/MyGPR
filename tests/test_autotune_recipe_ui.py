#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoTune workflow recipe and simplified UI contracts."""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.autotune_recipe import build_workflow_recipe
from core.autotune_scoring_weights import resolve_scoring_weights
from ui.autotune_tuning_page import AutoTuneTuningPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_landslide_goal_has_own_recipe_and_weights():
    recipe = build_workflow_recipe(
        target_goal="landslide_interface",
        roi_mode="none",
        best_candidate_name="SVD 背景抑制 rank=3",
        best_candidate_params="rank=3",
        best_score=0.86,
        target_response_available=False,
        backend_mode="UI 预览",
    )
    assert recipe.target_goal == "滑坡基覆界面 / 潜在滑移面"
    assert "深部保留增益" in recipe.flow_text
    assert "SVD 背景抑制 rank=3" in recipe.parameter_text

    goal, metrics, weights = resolve_scoring_weights(
        target_goal="滑坡基覆界面 / 潜在滑移面",
        scoring_metrics=["roi_retention", "residual", "cnr", "shape"],
        target_response_available=False,
    )
    assert goal == "滑坡基覆界面 / 潜在滑移面"
    assert set(metrics) == {"roi_retention", "residual", "cnr", "shape"}
    assert weights["shape"] > weights["residual"]


def test_autotune_page_uses_compact_recipe_layout_without_front_risk_copy():
    app = _get_app()
    page = AutoTuneTuningPage()
    try:
        raw = np.tile(np.linspace(0.0, 1.0, 32)[:, None], (1, 16))
        raw[14:20, 6:9] += 0.5
        page.set_loaded_dataset(
            file_path="raw_Ey.npy",
            data_shape=raw.shape,
            data_type="NumPy",
            component="Ey",
            processing_stage="原始数据",
            source_label="raw_Ey.npy",
            data_array=raw,
        )
        page.target_goal_combo.setCurrentText("滑坡基覆界面 / 潜在滑移面")
        page._on_run_recommendation_preview()

        assert page.state.recommendation_status == "已生成"
        assert page.detail_tabs.tabText(0) == "流程"
        assert page.detail_tabs.tabText(1) == "候选参数"
        assert page.workflow_table.rowCount() >= 4
        front_text = "\n".join([
            page.next_step_hint.text(),
        ])
        assert "风险" not in front_text
        assert "复核" not in front_text
    finally:
        page.close()
        app.processEvents()


def test_autotune_page_hides_research_console_in_field_mode():
    app = _get_app()
    page = AutoTuneTuningPage()
    try:
        page.show()
        app.processEvents()
        tab_labels = [page.advanced_panel.tabText(index) for index in range(page.advanced_panel.count())]
        assert "开发工具" not in tab_labels
        assert page.research_console_panel is None
        assert not page.btn_open_research_console.isVisible()
    finally:
        page.close()
        app.processEvents()


def test_autotune_page_generates_workflow_recipes_not_only_background_rows():
    app = _get_app()
    page = AutoTuneTuningPage()
    try:
        y = np.linspace(0.0, 1.0, 64)[:, None]
        x = np.linspace(0.0, 1.0, 32)[None, :]
        raw = 0.08 * np.sin(8 * y) + 0.25 * np.exp(-((y - 0.55) ** 2) / 0.004)
        raw = np.tile(raw, (1, 32)) + 0.03 * np.cos(6 * x)
        page.set_loaded_dataset(
            file_path="raw_Ey.npy",
            data_shape=raw.shape,
            data_type="NumPy",
            component="Ey",
            processing_stage="原始数据",
            source_label="raw_Ey.npy",
            data_array=raw,
        )
        page.target_goal_combo.setCurrentText("连续界面保留")
        page._on_run_recommendation_preview()

        assert page.state.recommendation_status == "已生成"
        assert page.state.backend_mode == "流程推荐"
        assert len(page.state.backend_results) >= 2
        assert page.state.backend_results[0]["method"] == "workflow_recipe"
        assert page.state.backend_results[0]["recipe_steps"]
        assert page.workflow_table.rowCount() >= 4
        assert page.workflow_table.rowCount() >= 4
        front_text = "\n".join([page.next_step_hint.text()])
        assert "风险" not in front_text
        assert "复核" not in front_text
    finally:
        page.close()
        app.processEvents()


def test_autotune_page_surfaces_scoring_v2_in_details_and_report():
    app = _get_app()
    page = AutoTuneTuningPage()
    try:
        y = np.linspace(0.0, 1.0, 64)[:, None]
        x = np.linspace(0.0, 1.0, 32)[None, :]
        raw = 0.06 * np.sin(9 * y) + 0.20 * np.exp(-((y - (0.57 + 0.02 * np.sin(2 * np.pi * x))) ** 2) / 0.004)
        page.set_loaded_dataset(
            file_path="raw_Ey.npy",
            data_shape=raw.shape,
            data_type="NumPy",
            component="Ey",
            processing_stage="原始数据",
            source_label="raw_Ey.npy",
            data_array=raw,
        )
        page.target_goal_combo.setCurrentText("滑坡基覆界面 / 潜在滑移面")
        page._on_run_recommendation_preview()

        assert page.state.recommendation_status == "已生成"
        top = page.state.backend_results[0]
        assert "autotune_scoring_record" in top
        assert "workflow_score" in top["autotune_scoring_record"]
        assert "background_score" in top["autotune_scoring_record"]
        assert "scoring v2" in page.score_text.toPlainText()
        assert "目标权重" in page.score_text.toPlainText()
        assert "scoring v2" in page.apply_report_text.toPlainText()
        assert page.trial_table.columnCount() == 8
    finally:
        page.close()
        app.processEvents()


def test_autotune_workflow_table_is_editable_and_report_button_hidden():
    app = _get_app()
    page = AutoTuneTuningPage()
    try:
        raw = np.tile(np.linspace(0.0, 1.0, 48)[:, None], (1, 24))
        page.set_loaded_dataset(
            file_path="raw_Ey.npy",
            data_shape=raw.shape,
            data_type="NumPy",
            component="Ey",
            processing_stage="原始数据",
            source_label="raw_Ey.npy",
            data_array=raw,
        )
        page._on_run_recommendation_preview()

        assert page.btn_export_step_report.isHidden()
        assert [page.workflow_table.horizontalHeaderItem(i).text() for i in range(3)] == ["步骤", "参数", "处理方式"]

        background_row = None
        for row in range(page.workflow_table.rowCount()):
            if page.workflow_table.item(row, 0).data(0x0100) == "background":
                background_row = row
                break
        assert background_row is not None
        param_item = page.workflow_table.item(background_row, 1)
        param_item.setText("rank=3")
        app.processEvents()

        assert page.state.workflow_customized is True
        assert page.state.workflow_param_overrides["background"] == "rank=3"
        payload = page._selected_recipe_payload()
        assert payload["manual_override"] is True
        assert payload["parameter_override"] is True
        assert payload["manual_review_required"] is True
        edited_step = [step for step in payload["recipe_steps"] if step["key"] == "background"][0]
        assert edited_step["params"] == "rank=3"
        assert edited_step["manual_override"] is True
    finally:
        page.close()
        app.processEvents()


def test_autotune_workflow_order_override_is_recorded_and_validated():
    app = _get_app()
    page = AutoTuneTuningPage()
    try:
        raw = np.tile(np.linspace(0.0, 1.0, 48)[:, None], (1, 24))
        page.set_loaded_dataset(
            file_path="raw_Ey.npy",
            data_shape=raw.shape,
            data_type="NumPy",
            component="Ey",
            processing_stage="原始数据",
            source_label="raw_Ey.npy",
            data_array=raw,
        )
        page._on_run_recommendation_preview()
        assert page._workflow_order_is_valid(["zero_time", "dewow", "bandpass", "background", "gain"])[0]
        assert not page._workflow_order_is_valid(["gain", "zero_time", "dewow", "bandpass", "background"])[0]

        page.state.workflow_order = ["zero_time", "dewow", "bandpass", "background", "gain"]
        page.state.workflow_order_override = True
        page.state.workflow_customized = True
        payload = page._selected_recipe_payload()
        assert payload["workflow_override"] is True
        assert payload["workflow_order"] == ["zero_time", "dewow", "bandpass", "background", "gain"]
        assert [step["key"] for step in payload["recipe_steps"]][:5] == payload["workflow_order"]
    finally:
        page.close()
        app.processEvents()
