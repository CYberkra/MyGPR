#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V0.8.43 layout contract tests for quality and space pages."""

from __future__ import annotations

import os
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication

from ui.gui_quality_log import QualityLogPage
from ui.georef3d_results_page import Terrain3DResultsPage
from core.uav_georeference_3d import build_airborne_georeference_3d_payload


def _get_app() -> QApplication:
    return cast(QApplication, QApplication.instance() or QApplication([]))


def test_quality_page_uses_compact_section_switcher() -> None:
    app = _get_app()
    page = QualityLogPage()
    try:
        assert page.quality_section_stack.count() == 4
        page.quality_section_segmented.setCurrentItem("record")
        app.processEvents()
        assert page.quality_section_stack.currentIndex() == 1
        page.quality_section_segmented.setCurrentItem("report")
        app.processEvents()
        assert page.quality_section_stack.currentIndex() == 2
        assert page.btn_generate_report.text() == "生成项目报告"
        assert page.btn_export_replay_evidence.text() == "导出处理记录包"
    finally:
        page.release_plot_resources()
        page.close()
        app.processEvents()


def test_space_page_shows_clear_empty_state_without_spatial_payload() -> None:
    app = _get_app()
    page = Terrain3DResultsPage()
    try:
        assert page.space_empty_state_card.isHidden() is False
        assert "空间数据未接入" in page.terrain_bottom_status.text()
        assert "等待空间资料" in page.interpretation_object_label.text()
    finally:
        page.release_plot_resources()
        page.close()
        app.processEvents()


def test_space_page_hides_empty_state_when_spatial_payload_available() -> None:
    app = _get_app()
    page = Terrain3DResultsPage()
    try:
        data = np.arange(80, dtype=np.float32).reshape(10, 8)
        payload = build_airborne_georeference_3d_payload(
            data,
            {"total_time_ns": 80.0},
            {
                "trace_distance_m": np.linspace(0.0, 3.5, 8),
                "flight_height_m": np.linspace(1.0, 1.2, 8),
                "ground_elevation_m": np.linspace(100.0, 101.0, 8),
                "longitude": np.linspace(104.0, 104.001, 8),
                "latitude": np.linspace(30.0, 30.001, 8),
            },
        )
        assert payload is not None
        page.set_airborne_georeference_3d_visualization({"raw": payload, "current": payload})
        app.processEvents()
        assert page.space_empty_state_card.isHidden() is True
        assert "坐标" in page.terrain_bottom_status.text()
        assert page.btn_export_georeference_3d.isEnabled()
    finally:
        page.release_plot_resources()
        page.close()
        app.processEvents()
