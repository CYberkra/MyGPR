#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark runner smoke tests."""

from __future__ import annotations

from pathlib import Path

from core.benchmark_runner import run_benchmark_sample


def test_benchmark_runner_writes_summary_and_metrics(tmp_path: Path):
    summary = run_benchmark_sample(
        sample_id="zero_time_reference",
        method_keys=["set_zero_time", "dewow"],
        out_dir=tmp_path,
        save_images=False,
    )

    assert summary["sample_id"] == "zero_time_reference"
    assert summary["methods"] == ["set_zero_time", "dewow"]
    assert len(summary["steps"]) == 2
    assert summary["steps"][0]["method_key"] == "set_zero_time"
    assert "metrics" in summary["steps"][0]
    assert "baseline_bias_after" in summary["steps"][0]["metrics"]

    summary_path = tmp_path / "zero_time_reference-summary.json"
    assert summary_path.exists()
