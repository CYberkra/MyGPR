#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark runner smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


def test_benchmark_runner_tolerates_invalid_plot_metadata(monkeypatch, tmp_path: Path):
    raw = np.arange(48, dtype=np.float32).reshape(12, 4)

    def fake_generate(sample_id: str, seed: int = 42):
        return raw, {
            "header_info": {
                "total_time_ns": "bad",
                "trace_interval_m": "bad",
            }
        }

    monkeypatch.setattr("core.benchmark_runner.generate_benchmark_sample", fake_generate)

    summary = run_benchmark_sample(
        sample_id="zero_time_reference",
        method_keys=["dewow"],
        out_dir=tmp_path,
        save_images=True,
    )

    assert summary["sample_id"] == "zero_time_reference"
    assert (tmp_path / "zero_time_reference-00-raw.png").exists()
    assert (tmp_path / "zero_time_reference-01-dewow.png").exists()
