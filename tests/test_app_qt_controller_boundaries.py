# -*- coding: utf-8 -*-
"""Regression checks for V0.8 controller extraction boundaries."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_QT = ROOT / "app_qt.py"


def test_app_qt_line_budget_after_v08_controller_extraction():
    """The main window should remain within the V0.8 controller-boundary budget."""
    line_count = len(APP_QT.read_text(encoding="utf-8").splitlines())
    assert line_count <= 7700


def test_app_qt_no_longer_imports_moved_autotune_guard_helpers():
    """AutoTune/no-prior synchronization helpers belong to the dedicated controller."""
    text = APP_QT.read_text(encoding="utf-8")
    assert "from core.no_prior_qc_policy import" not in text
    assert "from core.no_prior_ui_guardrails import" not in text
    assert "from core.auto_tune_recommendation_labels import" not in text


def test_controller_modules_are_registered_in_app_qt():
    """All extracted V0.8 controllers should remain explicit app_qt dependencies."""
    text = APP_QT.read_text(encoding="utf-8")
    expected = [
        "from ui.report_export_controller import ReportExportController",
        "from ui.processing_lineage_controller import ProcessingLineageController",
        "from ui.bscan_interaction_controller import BscanInteractionController",
        "from ui.autotune_sync_controller import AutoTuneSyncController",
    ]
    for line in expected:
        assert line in text


def test_no_obsolete_noop_relocate_status_wrapper():
    """V0.8.5 removes a no-op compatibility method left after UI status cleanup."""
    text = APP_QT.read_text(encoding="utf-8")
    assert "def _relocate_basic_status_brief" not in text
    assert "_relocate_basic_status_brief()" not in text
