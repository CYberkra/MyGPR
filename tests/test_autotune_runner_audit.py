#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for AutoTune backend MVP."""

from __future__ import annotations

import numpy as np
import pytest

from core import autotune_background_runner as runner
from scripts.preflight_check import _is_generated_artifact_path


def test_background_runner_keeps_sliding_candidate_aligned_with_ui_options():
    data = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float64)[:, None], (1, 12))
    data[:, 4:7] += 0.5

    rows = runner.run_background_candidates(
        data,
        candidate_methods=["baseline", "sliding"],
        roi={"sample_start": 5, "sample_end": 25, "trace_start": 3, "trace_end": 8},
    )

    methods = {row["method"] for row in rows}
    assert methods == {"baseline", "sliding"}
    assert all(np.isfinite(row["score"]) for row in rows)


def test_background_runner_rejects_unknown_candidate_instead_of_silently_dropping_it():
    data = np.ones((8, 6), dtype=np.float64)

    with pytest.raises(ValueError, match="Unsupported candidate"):
        runner.run_background_candidates(data, candidate_methods=["baseline", "not_a_method"])


def test_background_runner_reuses_one_svd_for_rank_sweep(monkeypatch):
    data = np.arange(80, dtype=np.float64).reshape(10, 8)
    calls = {"count": 0}
    original = runner._compute_svd

    def counted_svd(arr):
        calls["count"] += 1
        return original(arr)

    monkeypatch.setattr(runner, "_compute_svd", counted_svd)

    rows = runner.run_background_candidates(data, candidate_methods=["svd"], svd_ranks=[1, 2, 3])

    assert calls["count"] == 1
    assert [row["method"] for row in rows] == ["svd", "svd", "svd"]


def test_background_runner_records_goal_metric_weights():
    data = np.tile(np.linspace(0.0, 1.0, 36, dtype=np.float64)[:, None], (1, 16))
    data[12:24, 6:10] += 0.8

    rows = runner.run_background_candidates(
        data,
        candidate_methods=["baseline", "median"],
        roi={"sample_start": 10, "sample_end": 26, "trace_start": 5, "trace_end": 11},
        target_goal="连续界面保留",
        scoring_metrics=["roi_retention", "shape"],
    )

    assert rows
    assert rows[0]["target_goal"] == "连续界面保留"
    assert set(rows[0]["scoring_weights"]) == {"roi_retention", "shape"}
    assert rows[0]["scoring_weights"]["shape"] > rows[0]["scoring_weights"]["roi_retention"]


def test_background_runner_skips_svd_when_trace_guard_exceeded():
    data = np.ones((8, 20), dtype=np.float64)

    rows = runner.run_background_candidates(data, candidate_methods=["svd"], max_svd_traces=4)

    assert rows[0]["status"] == "已跳过"
    assert "max_svd_traces" in rows[0]["params"]


def test_preflight_blocks_common_large_artifact_suffixes():
    blocked = [
        "experiments/gprmax/GX-001/raw.out",
        "experiments/gprmax/GX-001/raw.h5",
        "experiments/gprmax/GX-001/raw.hdf5",
        "experiments/gprmax/GX-001/raw.vti",
        "experiments/gprmax/GX-001/raw.vtk",
        "experiments/gprmax/GX-001/raw.vtu",
        "experiments/gprmax/GX-001/converted/target_response.npy",
        "experiments/gprmax/GX-001/paired_outputs/preview.jpg",
    ]

    assert all(_is_generated_artifact_path(path) for path in blocked)
    assert not _is_generated_artifact_path("experiments/gprmax/GX-001/models/scene.in")
    assert not _is_generated_artifact_path("assets/icons/logo.png")
