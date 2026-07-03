#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V0.8.44 final tab responsibility audit tests."""

from __future__ import annotations

import os
from typing import Iterable

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QCheckBox, QRadioButton, QGroupBox, QWidget

from ui.gui_advanced_settings import AdvancedSettingsPage
from ui.gui_basic_flow import BasicFlowPage
from ui.autotune_tuning_page import AutoTuneTuningPage
from ui.gui_quality_log import QualityLogPage
from ui.georef3d_results_page import Terrain3DResultsPage


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _visible_texts(root: QWidget) -> list[str]:
    texts: list[str] = []
    classes = (QLabel, QPushButton, QCheckBox, QRadioButton, QGroupBox)
    for widget in root.findChildren(classes):
        try:
            if not widget.isVisibleTo(root):
                continue
            text = getattr(widget, "text", lambda: "")()
        except Exception:
            continue
        if text:
            texts.append(str(text))
    return texts


def test_display_page_visible_controls_remain_display_only() -> None:
    app = _app()
    page = AdvancedSettingsPage()
    try:
        page.show()
        app.processEvents()
        page.segmented.setCurrentItem("enhance")
        app.processEvents()
        visible = "\n".join(_visible_texts(page))
        forbidden_visible_terms = [
            "选择 RTK",
            "选择 IMU",
            "选择高度计",
            "Dewow",
            "背景抑制",
            "SVD",
            "AGC",
            "迁移",
            "运动补偿",
        ]
        assert all(term not in visible for term in forbidden_visible_terms)
        assert "不改变处理数组" in visible
        assert page.mutates_data is False
    finally:
        page.close()
        app.processEvents()


def test_processing_page_stage_filter_is_operational_not_static_instruction() -> None:
    app = _app()
    page = BasicFlowPage()
    try:
        page.show()
        app.processEvents()
        assert hasattr(page, "_stage_filter_buttons")
        assert not hasattr(page, "_flow_step_chips")
        page.set_method_stage_filter("suppress")
        visible = "\n".join(_visible_texts(page))
        assert "抑制" in visible
        assert any(key in page.method_keys for key in ["median_background_2D", "svd_bg", "subtracting_average_2D"])
    finally:
        page.close()
        app.processEvents()


def test_autotune_page_primary_tabs_are_concise() -> None:
    app = _app()
    page = AutoTuneTuningPage()
    try:
        page.show()
        app.processEvents()
        labels = [page.detail_tabs.tabText(i) for i in range(page.detail_tabs.count())]
        assert labels == ["流程", "候选", "说明"]
        visible = "\n".join(_visible_texts(page))
        assert "生成推荐" in visible
        assert "应用并运行推荐流程" not in visible
        assert "运行" in visible
    finally:
        page.close()
        app.processEvents()


def test_quality_and_space_pages_keep_distinct_responsibilities() -> None:
    app = _app()
    quality = QualityLogPage()
    space = Terrain3DResultsPage()
    try:
        quality.show()
        space.show()
        app.processEvents()
        for route_key, expected_index in {"qc": 0, "record": 1, "report": 2, "advanced": 3}.items():
            quality.quality_section_segmented.setCurrentItem(route_key)
            app.processEvents()
            assert quality.quality_section_stack.currentIndex() == expected_index
        assert quality.btn_generate_report.text() == "生成项目报告"
        assert space.rtk_sidecar_button.isVisibleTo(space)
        assert space.imu_sidecar_button.isVisibleTo(space)
        assert space.altimeter_sidecar_button.isVisibleTo(space)
        assert "空间数据未接入" in space.terrain_bottom_status.text()
    finally:
        quality.release_plot_resources()
        space.release_plot_resources()
        quality.close()
        space.close()
        app.processEvents()
