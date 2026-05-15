#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导入、报告与质量快照测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox

from app_qt import GPRGuiQt

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _advanced(win: GPRGuiQt):
    return win._ensure_advanced_page()


def test_csv_import_records_sanitize_warnings(tmp_path: Path):
    app = _get_app()
    win = GPRGuiQt()
    try:
        csv_path = tmp_path / "demo.csv"
        arr = np.linspace(0, 1, 200 * 40, dtype=np.float32).reshape(200, 40)
        arr[5, 9] = np.nan
        arr[10, 12] = np.inf
        np.savetxt(csv_path, arr, delimiter=",", fmt="%s")

        win._load_single_csv(str(csv_path))

        codes = {warning.get("code") for warning in win._runtime_warnings}
        assert "data_sanitized" in codes
    finally:
        win.close()
        app.processEvents()


def test_report_and_quality_snapshot_include_runtime_warnings(tmp_path: Path):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(120, dtype=np.float32).reshape(10, 12)
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win._append_runtime_warnings(
            [
                {
                    "code": "data_sanitized",
                    "level": "warning",
                    "message": "导入 CSV 数据包含 NaN/Inf，已使用均值填充。",
                    "details": {"source": "csv_import"},
                }
            ],
            log=False,
        )
        win._last_quality_metrics = {
            "focus_ratio": 0.3,
            "hot_pixels": 0,
            "spikiness": 1.1,
            "time_ms": 3.0,
        }
        win._set_last_run_summary(
            "single",
            "低频漂移矫正（Dewow）",
            [
                {
                    "method_key": "dewow",
                    "method_name": "低频漂移矫正（Dewow）",
                    "params": {"window": 23},
                    "elapsed_ms": 5.0,
                }
            ],
            warnings=list(win._runtime_warnings),
        )
        win._default_output_dir = lambda: str(tmp_path)  # type: ignore[method-assign]

        win.generate_report()
        report_path = next(tmp_path.glob("report_*.md"))
        report_text = report_path.read_text(encoding="utf-8")
        assert "Runtime warnings" in report_text
        assert "data_sanitized" in report_text

        original_information = QMessageBox.information
        QMessageBox.information = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
        try:
            win.export_quality_snapshot()
        finally:
            QMessageBox.information = original_information
        json_path = next(tmp_path.glob("quality_snapshot_*.json"))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["runtime_warnings"]
        assert payload["runtime_warnings"][0]["code"] == "data_sanitized"
    finally:
        win.close()
        app.processEvents()


def test_report_and_quality_snapshot_include_profile_workflow_summary(
    monkeypatch, tmp_path: Path
):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(120, dtype=np.float32).reshape(10, 12)
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win._last_quality_metrics = {
            "focus_ratio": 0.3,
            "hot_pixels": 0,
            "spikiness": 1.1,
            "time_ms": 3.0,
        }
        win._set_last_run_summary(
            "recommended",
            "高质量 UAV-GPR",
            [
                {
                    "method_key": "motion_compensation_v2",
                    "method_name": "UAV运动补偿V2",
                    "params": {"height_source": "auto"},
                    "elapsed_ms": 5.0,
                }
            ],
            profile_key="high_quality_uav_gpr",
        )
        win._default_output_dir = lambda: str(tmp_path)  # type: ignore[method-assign]

        win.generate_report()

        report_path = next(tmp_path.glob("report_*.md"))
        report_text = report_path.read_text(encoding="utf-8")
        assert "Workflow stages" in report_text
        assert "motion_compensation_v2" in report_text
        assert "Sensor dependencies" in report_text
        assert "height_agl_m" in report_text

        monkeypatch.setattr(
            QMessageBox,
            "information",
            lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
        )
        win.export_quality_snapshot()

        json_path = next(tmp_path.glob("quality_snapshot_*.json"))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        workflow = payload["last_run_summary"]["workflow_summary"]
        assert workflow["profile_key"] == "high_quality_uav_gpr"
        assert workflow["unassigned_methods"] == []
        assert any(
            warning["method_key"] == "motion_compensation_v2"
            for warning in workflow["sensor_dependency_warnings"]
        )
    finally:
        win.close()
        app.processEvents()


def test_report_includes_crop_bounds_when_crop_enabled(tmp_path: Path):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(400, dtype=np.float32).reshape(20, 20)
        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win._default_output_dir = lambda: str(tmp_path)  # type: ignore[method-assign]
        _advanced(win).crop_enable_var.setChecked(True)
        _advanced(win).time_start_edit.setText("2")
        _advanced(win).time_end_edit.setText("10")
        _advanced(win).dist_start_edit.setText("3")
        _advanced(win).dist_end_edit.setText("12")

        win.generate_report()

        report_path = next(tmp_path.glob("report_*.md"))
        report_text = report_path.read_text(encoding="utf-8")
        assert "- Crop: time 2.0~10.0 ; distance 3.0~12.0" in report_text
    finally:
        win.close()
        app.processEvents()
