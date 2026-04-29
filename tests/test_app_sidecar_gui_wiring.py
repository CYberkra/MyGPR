#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI-level optional RTK/IMU sidecar wiring tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app_qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from app_qt import GPRGuiQt


def _get_app() -> QApplication:
    """Return the singleton QApplication used by raw PyQt tests."""
    return cast(QApplication, QApplication.instance() or QApplication([]))


def _close_window(app: QApplication, win: GPRGuiQt) -> None:
    win.close()
    app.processEvents()


def test_no_sidecar_selection_builds_empty_loader_kwargs() -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        win._sidecar_files = {"rtk": None, "imu": None, "altimeter": None}

        kwargs = win._build_sidecar_loader_kwargs("demo.csv")

        assert kwargs == {}
    finally:
        _close_window(app, win)


def test_sidecar_controls_exist_and_start_empty() -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        page = win.page_advanced

        assert page.rtk_sidecar_button is not None
        assert page.rtk_sidecar_clear_button is not None
        assert "未选择" in page.rtk_sidecar_label.text()
        assert page.imu_sidecar_button is not None
        assert page.imu_sidecar_clear_button is not None
        assert "未选择" in page.imu_sidecar_label.text()
    finally:
        _close_window(app, win)


def test_altimeter_is_not_exposed_in_gui_slice() -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        page = win.page_advanced

        assert not hasattr(page, "altimeter_sidecar_button")
        assert not hasattr(page, "altimeter_sidecar_clear_button")
        assert not hasattr(page, "altimeter_sidecar_label")
    finally:
        _close_window(app, win)


def test_pick_sidecar_updates_state_and_label(monkeypatch, tmp_path: Path) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        rtk_path = tmp_path / "rtk.csv"
        monkeypatch.setattr(
            app_qt.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(rtk_path), "CSV Files (*.csv)"),
        )

        win._pick_sidecar_file("rtk")

        assert win._sidecar_files["rtk"] == str(rtk_path)
        assert "rtk.csv" in win.page_advanced.rtk_sidecar_label.text()
    finally:
        _close_window(app, win)


def test_pick_sidecar_cancel_preserves_existing_path(monkeypatch, tmp_path: Path) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        existing_path = str(tmp_path / "previous_rtk.csv")
        win._set_sidecar_file("rtk", existing_path)
        monkeypatch.setattr(
            app_qt.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: ("", ""),
        )

        win._pick_sidecar_file("rtk")

        assert win._sidecar_files["rtk"] == existing_path
        assert "previous_rtk.csv" in win.page_advanced.rtk_sidecar_label.text()
    finally:
        _close_window(app, win)


def test_clear_sidecar_resets_only_target_kind(tmp_path: Path) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        rtk_path = str(tmp_path / "rtk.csv")
        imu_path = str(tmp_path / "imu.csv")
        win._set_sidecar_file("rtk", rtk_path)
        win._set_sidecar_file("imu", imu_path)

        win._clear_sidecar_file("rtk")

        assert win._sidecar_files["rtk"] is None
        assert win._sidecar_files["imu"] == imu_path
        assert "未选择" in win.page_advanced.rtk_sidecar_label.text()
        assert "imu.csv" in win.page_advanced.imu_sidecar_label.text()
    finally:
        _close_window(app, win)


def test_warn_sidecar_ignored_uses_warning_wrapper(monkeypatch) -> None:
    app = _get_app()
    win = GPRGuiQt()
    calls: list[tuple[Any, ...]] = []
    original_warning = QMessageBox.warning
    QMessageBox.warning = lambda *args, **kwargs: calls.append(args) or QMessageBox.StandardButton.Ok
    try:
        win._warn_sidecar_ignored("RTK", "测试原因")

        assert len(calls) == 1
        assert "RTK" in str(calls[0])
        assert "测试原因" in str(calls[0])
    finally:
        QMessageBox.warning = original_warning
        _close_window(app, win)


def test_import_csv_without_sidecars_uses_legacy_kwargs(monkeypatch, tmp_path: Path) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("1,2\n3,4\n", encoding="utf-8")
        captured: dict[str, Any] = {}

        monkeypatch.setattr(
            app_qt.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(csv_path), "CSV Files (*.csv)"),
        )

        def fake_load_with_progress(title: str, loader, path: str, **loader_kwargs):
            captured["title"] = title
            captured["loader"] = loader
            captured["path"] = path
            captured["loader_kwargs"] = dict(loader_kwargs)
            return {"data": np.zeros((2, 2), dtype=np.float32), "metadata": {}}

        monkeypatch.setattr(win, "_load_with_progress", fake_load_with_progress)
        monkeypatch.setattr(win, "_warn_sidecar_ignored", lambda *args, **kwargs: captured.setdefault("warned", True))

        win.import_csv_file()

        assert captured["path"] == str(csv_path)
        assert captured["loader_kwargs"] == {}
        assert "warned" not in captured
    finally:
        _close_window(app, win)


def test_import_csv_with_sidecar_but_no_timestamps_warns_and_uses_legacy_kwargs(monkeypatch, tmp_path: Path) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        csv_path = tmp_path / "data.csv"
        rtk_path = tmp_path / "rtk.csv"
        csv_path.write_text("1,2\n3,4\n", encoding="utf-8")
        rtk_path.write_text("timestamp_s,lat,lon\n0,1,2\n", encoding="utf-8")
        captured: dict[str, Any] = {"warnings": []}
        win._set_sidecar_file("rtk", str(rtk_path))

        monkeypatch.setattr(
            app_qt.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(csv_path), "CSV Files (*.csv)"),
        )
        monkeypatch.setattr(
            win,
            "_warn_sidecar_ignored",
            lambda kind, reason: captured["warnings"].append((kind, reason)),
        )

        def fake_load_with_progress(title: str, loader, path: str, **loader_kwargs):
            captured["loader_kwargs"] = dict(loader_kwargs)
            return {"data": np.zeros((2, 2), dtype=np.float32), "metadata": {}}

        monkeypatch.setattr(win, "_load_with_progress", fake_load_with_progress)

        win.import_csv_file()

        assert captured["loader_kwargs"] == {}
        assert len(captured["warnings"]) == 1
        assert captured["warnings"][0][0] == "RTK/IMU"
    finally:
        _close_window(app, win)


def test_import_csv_with_explicit_timestamp_column_forwards_sidecars(monkeypatch, tmp_path: Path) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        csv_path = tmp_path / "timestamped_airborne.csv"
        csv_path.write_text(
            "Number of Samples = 2\n"
            "Time windows (ns) = 20.0\n"
            "Number of Traces = 3\n"
            "Trace interval (m) = 0.5\n"
            "100.0,30.0,10.0,1.0,5.0,0.0\n"
            "100.0,30.0,10.0,2.0,5.0,0.0\n"
            "100.1,30.0,10.1,3.0,5.1,0.2\n"
            "100.1,30.0,10.1,4.0,5.1,0.2\n"
            "100.2,30.0,10.2,5.0,5.2,0.4\n"
            "100.2,30.0,10.2,6.0,5.2,0.4\n",
            encoding="utf-8",
        )
        rtk_path = tmp_path / "rtk.csv"
        imu_path = tmp_path / "imu.csv"
        rtk_path.write_text("timestamp_s,lat,lon\n0,30,100\n", encoding="utf-8")
        imu_path.write_text("timestamp_s,roll_deg,pitch_deg,yaw_deg\n0,1,2,3\n", encoding="utf-8")
        captured: dict[str, Any] = {}

        win._set_sidecar_file("rtk", str(rtk_path))
        win._set_sidecar_file("imu", str(imu_path))

        monkeypatch.setattr(
            app_qt.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(csv_path), "CSV Files (*.csv)"),
        )
        monkeypatch.setattr(win, "_warn_sidecar_ignored", lambda *args, **kwargs: captured.setdefault("warned", True))

        def fake_load_with_progress(title: str, loader, path: str, **loader_kwargs):
            captured["loader_kwargs"] = dict(loader_kwargs)
            return {"data": np.zeros((2, 3), dtype=np.float32), "metadata": {}}

        monkeypatch.setattr(win, "_load_with_progress", fake_load_with_progress)

        win.import_csv_file()

        forwarded = cast(dict[str, Any], captured["loader_kwargs"])
        assert np.array_equal(
            forwarded["trace_timestamps_s"], np.array([0.0, 0.2, 0.4], dtype=np.float64)
        )
        assert forwarded["rtk_path"] == str(rtk_path)
        assert forwarded["imu_path"] == str(imu_path)
        assert "warned" not in captured
    finally:
        _close_window(app, win)


def test_import_csv_with_sidecars_and_existing_timestamps_for_same_source_forwards_kwargs(monkeypatch, tmp_path: Path) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        csv_path = tmp_path / "data.csv"
        rtk_path = tmp_path / "rtk.csv"
        imu_path = tmp_path / "imu.csv"
        csv_path.write_text("1,2\n3,4\n", encoding="utf-8")
        rtk_path.write_text("timestamp_s,lat,lon\n0,1,2\n", encoding="utf-8")
        imu_path.write_text("timestamp_s,roll_deg,pitch_deg,yaw_deg\n0,1,2,3\n", encoding="utf-8")
        trace_timestamps_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        captured: dict[str, Any] = {}

        win.shared_data.load_data(
            np.zeros((2, 3), dtype=np.float32),
            path=str(csv_path),
            header_info={"a_scan_length": 2, "num_traces": 3},
            trace_metadata={"trace_timestamp_s": trace_timestamps_s},
            source="test_seed",
        )
        win._set_sidecar_file("rtk", str(rtk_path))
        win._set_sidecar_file("imu", str(imu_path))

        monkeypatch.setattr(
            app_qt.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(csv_path), "CSV Files (*.csv)"),
        )
        monkeypatch.setattr(win, "_warn_sidecar_ignored", lambda *args, **kwargs: captured.setdefault("warned", True))

        def fake_load_with_progress(title: str, loader, path: str, **loader_kwargs):
            captured["loader_kwargs"] = dict(loader_kwargs)
            return {"data": np.zeros((2, 2), dtype=np.float32), "metadata": {}}

        monkeypatch.setattr(win, "_load_with_progress", fake_load_with_progress)

        win.import_csv_file()

        forwarded = cast(dict[str, Any], captured["loader_kwargs"])
        assert np.array_equal(forwarded["trace_timestamps_s"], trace_timestamps_s)
        assert forwarded["rtk_path"] == str(rtk_path)
        assert forwarded["imu_path"] == str(imu_path)
        assert win._trace_timestamps_s is not None
        assert np.array_equal(win._trace_timestamps_s, trace_timestamps_s)
        assert "warned" not in captured
    finally:
        _close_window(app, win)


def test_import_csv_with_sidecars_does_not_reuse_timestamps_from_different_source(monkeypatch, tmp_path: Path) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        csv_path = tmp_path / "new_data.csv"
        old_path = tmp_path / "old_data.csv"
        rtk_path = tmp_path / "rtk.csv"
        csv_path.write_text("1,2\n3,4\n", encoding="utf-8")
        old_path.write_text("1,2\n3,4\n", encoding="utf-8")
        rtk_path.write_text("timestamp_s,lat,lon\n0,1,2\n", encoding="utf-8")
        stale_timestamps_s = np.array([10.0, 11.0, 12.0], dtype=np.float64)
        captured: dict[str, Any] = {"warnings": []}

        win.shared_data.load_data(
            np.zeros((2, 3), dtype=np.float32),
            path=str(old_path),
            header_info={"a_scan_length": 2, "num_traces": 3},
            trace_metadata={"trace_timestamp_s": stale_timestamps_s},
            source="test_seed",
        )
        win._set_sidecar_file("rtk", str(rtk_path))

        monkeypatch.setattr(
            app_qt.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(csv_path), "CSV Files (*.csv)"),
        )
        monkeypatch.setattr(
            win,
            "_warn_sidecar_ignored",
            lambda kind, reason: captured["warnings"].append((kind, reason)),
        )

        def fake_load_with_progress(title: str, loader, path: str, **loader_kwargs):
            captured["loader_kwargs"] = dict(loader_kwargs)
            return {"data": np.zeros((2, 2), dtype=np.float32), "metadata": {}}

        monkeypatch.setattr(win, "_load_with_progress", fake_load_with_progress)

        win.import_csv_file()

        assert captured["loader_kwargs"] == {}
        assert len(captured["warnings"]) == 1
        assert captured["warnings"][0][0] == "RTK/IMU"
    finally:
        _close_window(app, win)


def test_malformed_sidecar_warns_and_falls_back_to_plain_csv(monkeypatch) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        warnings: list[tuple[str, str]] = []
        expected = (
            np.ones((2, 2), dtype=np.float32),
            {"longitude": np.array([1.0, 2.0])},
            {"num_traces": 2},
        )

        def fake_extract(*args, **kwargs):
            calls.append((args, dict(kwargs)))
            if kwargs.get("rtk_path"):
                raise ValueError("sidecar timestamp_s 列缺失")
            return expected

        monkeypatch.setattr(app_qt, "extract_airborne_csv_payload", fake_extract)
        monkeypatch.setattr(
            win,
            "_warn_sidecar_ignored",
            lambda kind, reason: warnings.append((kind, reason)),
        )

        result = win._extract_airborne_payload_with_optional_sidecars(
            np.zeros((2, 2), dtype=np.float32),
            {"num_traces": 2},
            {"trace_timestamps_s": [0.0, 0.1], "rtk_path": "bad_rtk.csv"},
        )

        assert result is expected
        assert len(calls) == 2
        assert calls[0][1]["rtk_path"] == "bad_rtk.csv"
        assert calls[1][1] == {}
        assert len(warnings) == 1
        assert warnings[0][0] == "RTK"
        assert "timestamp_s" in warnings[0][1]
    finally:
        _close_window(app, win)


def test_plain_csv_value_error_is_not_swallowed(monkeypatch) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        warnings: list[tuple[str, str]] = []

        def fake_extract(*args, **kwargs):
            raise ValueError("CSV 未读取到有效数据")

        monkeypatch.setattr(app_qt, "extract_airborne_csv_payload", fake_extract)
        monkeypatch.setattr(
            win,
            "_warn_sidecar_ignored",
            lambda kind, reason: warnings.append((kind, reason)),
        )

        with pytest.raises(ValueError, match="CSV 未读取到有效数据"):
            win._extract_airborne_payload_with_optional_sidecars(
                np.zeros((0, 0), dtype=np.float32), {}, {}
            )

        assert warnings == []
    finally:
        _close_window(app, win)


def test_primary_csv_metadata_error_with_sidecars_is_not_swallowed(monkeypatch) -> None:
    app = _get_app()
    win = GPRGuiQt()
    try:
        warnings: list[tuple[str, str]] = []

        def fake_extract(*args, **kwargs):
            raise ValueError("主 CSV 缺少 longitude 列")

        monkeypatch.setattr(app_qt, "extract_airborne_csv_payload", fake_extract)
        monkeypatch.setattr(
            win,
            "_warn_sidecar_ignored",
            lambda kind, reason: warnings.append((kind, reason)),
        )

        with pytest.raises(ValueError, match="longitude"):
            win._extract_airborne_payload_with_optional_sidecars(
                np.zeros((2, 2), dtype=np.float32),
                {"num_traces": 2},
                {
                    "trace_timestamps_s": np.array([0.0, 0.1], dtype=np.float64),
                    "rtk_path": "selected_rtk.csv",
                },
            )

        assert warnings == []
    finally:
        _close_window(app, win)
