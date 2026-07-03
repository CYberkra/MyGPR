#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the GPRPy baseline comparison report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import gprpy_baseline_report as report


def test_compare_gprpy_baseline_reports_zero_diff_for_aligned_kernels():
    rng = np.random.default_rng(123)
    data = rng.normal(size=(40, 16)).astype(np.float32)

    comparison = report.compare_gprpy_baseline(
        data,
        source_label="synthetic://aligned",
        dewow_window=7,
        ntraces=5,
        agc_window=9,
    )

    assert comparison["report_type"] == "gprpy_baseline_comparison"
    assert len(comparison["steps"]) == 3
    assert comparison["overall_metrics"]["max_abs_diff"] <= 1.0e-6
    assert "已与 GPRPy 基线对齐" in comparison["conclusion"]
    for step in comparison["steps"]:
        assert step["metrics"]["max_abs_diff"] <= 1.0e-6


def test_write_gprpy_baseline_report_outputs_html_and_summary(tmp_path: Path):
    data = np.arange(1, 241, dtype=np.float32).reshape(30, 8)
    comparison = report.compare_gprpy_baseline(
        data,
        source_label="synthetic://report",
        dewow_window=5,
        ntraces=3,
        agc_window=7,
    )

    summary = report.write_gprpy_baseline_report(
        comparison,
        tmp_path,
        save_images_flag=False,
    )

    html_path = tmp_path / "index.html"
    summary_path = tmp_path / "summary.json"
    assert html_path.exists()
    assert summary_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "MyGPR / GPRPy 基线对照报告" in html_text
    assert "remMeanTrace" in html_text

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["source_label"] == "synthetic://report"
    assert len(payload["steps"]) == 3
    assert payload["steps"][0]["images"] == {}
    assert summary["artifacts"]["html"] == str(html_path.resolve())
