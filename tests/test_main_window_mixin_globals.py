#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for globals used by split main-window mixins."""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui import main_window_export_mixin, main_window_quality_mixin, main_window_worker_mixin
from ui.main_window_quality_mixin import MainWindowQualityMixin


def test_split_mixins_import_runtime_globals_used_after_worker_finish():
    assert hasattr(main_window_quality_mixin, "build_profile_workflow_summary")
    assert hasattr(main_window_quality_mixin, "os")
    assert hasattr(main_window_quality_mixin, "plt")
    assert hasattr(main_window_export_mixin, "BASE_DIR")
    assert hasattr(main_window_worker_mixin, "QMessageBox")


def test_set_last_run_summary_with_profile_key_does_not_raise_name_error():
    class Dummy(MainWindowQualityMixin):
        _last_quality_metrics = {"focus_ratio": 0.2}
        _no_prior_guard_events = []

        def _compute_airborne_qc_metrics(self):
            return {}

        def _build_no_prior_qc_policy(self, *, metrics=None, airborne_qc=None):
            return {"mode": "unit_test"}

    dummy = Dummy()
    dummy._set_last_run_summary(
        "recommended",
        "高质量 UAV-GPR",
        [{"method_key": "motion_compensation_v2", "params": {}}],
        profile_key="high_quality_uav_gpr",
    )
    assert dummy._last_run_summary["workflow_summary"]["profile_key"] == "high_quality_uav_gpr"
