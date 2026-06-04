#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoTune tuning page synthetic target_response contract tests."""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtWidgets import QApplication

from ui.autotune_tuning_page import AutoTuneTuningPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_simple_recommendation_uses_bound_target_response_for_rmse_metric():
    app = _get_app()
    page = AutoTuneTuningPage()
    try:
        raw = np.tile(np.linspace(0.0, 1.0, 24)[:, None], (1, 10))
        raw[8:15, 4:7] += 0.6
        target = raw - np.mean(raw, axis=1, keepdims=True)

        page.set_loaded_dataset(
            file_path="raw_Ey.npy",
            data_shape=raw.shape,
            data_type="NumPy",
            component="Ey",
            processing_stage="原始数据",
            source_label="raw_Ey.npy",
            data_array=raw,
            target_response_array=target,
            target_response_label="target_response_Ey.npy",
        )
        page.state.candidate_methods = {"baseline", "mean"}
        page.state.scoring_metrics = {"rmse"}

        page._on_run_recommendation_preview()

        assert page.state.recommendation_status == "已生成"
        assert page.state.backend_results
        assert page.state.synthetic_gt_available is True
        assert all("rmse" in row.get("scoring_terms", {}) for row in page.state.backend_results)
        assert all("RMSE 指标缺少 target_response" not in row.get("warning", "") for row in page.state.backend_results)
    finally:
        page.close()
        app.processEvents()
