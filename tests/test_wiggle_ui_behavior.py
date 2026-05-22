#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wiggle UI behavior tests."""

from __future__ import annotations

import json

import pytest
from PyQt6.QtWidgets import QApplication

from ui.gui_advanced_settings import AdvancedSettingsPage
from ui.gui_workbench import format_workbench_wiggle_sampling_notice
from app_qt import _load_app_settings_dict, _save_app_settings_dict


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_view_style_set_get_supports_wiggle(qapp):
    page = AdvancedSettingsPage()
    page.set_view_style("wiggle")
    assert page.get_view_style() == "wiggle"
    page.set_view_style("image")
    assert page.get_view_style() == "image"


def test_cmap_disabled_in_wiggle_enabled_in_image(qapp):
    page = AdvancedSettingsPage()
    page.set_view_style("wiggle")
    assert page.cmap_combo.isEnabled() is False
    page.set_view_style("image")
    assert page.cmap_combo.isEnabled() is True


def test_settings_round_trip_keeps_unrelated_keys(tmp_path, monkeypatch):
    settings_path = tmp_path / "gpr_gui_settings.json"
    monkeypatch.setattr("app_qt._get_settings_path", lambda: str(settings_path))
    original = {"last_data_path": "D:/demo.csv", "custom_flag": 1}
    _save_app_settings_dict(original)
    loaded = _load_app_settings_dict()
    loaded["view_style"] = "wiggle"
    _save_app_settings_dict(loaded)
    final_loaded = json.loads(settings_path.read_text(encoding="utf-8"))
    assert final_loaded["custom_flag"] == 1
    assert final_loaded["last_data_path"] == "D:/demo.csv"
    assert final_loaded["view_style"] == "wiggle"


def test_wiggle_sampling_summary_for_line9_traces():
    summary = AdvancedSettingsPage.compute_wiggle_sampling_summary(2378, max_traces=80)
    assert summary["step"] == 30
    assert summary["shown"] == 80
    assert summary["total"] == 2378


def test_workbench_wiggle_sampling_notice_is_display_only():
    notice = format_workbench_wiggle_sampling_notice(
        n_traces=2378,
        shown_traces=80,
        n_samples=501,
    )
    assert "80/2378" in notice
    assert "仅用于显示" in notice
    assert "不改变数据" in notice
