#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gprMax ground-truth-aware validation metrics."""

from __future__ import annotations

import numpy as np

from core.gprmax_truth_metrics import compute_ground_truth_metrics


def _truth_manifest() -> dict:
    return {
        "schema": "mygpr_gprmax_ground_truth_v1",
        "scenario_id": "truth_metric_demo",
        "analysis_roi": {
            "time_start_idx": 8,
            "time_end_idx": 56,
            "dist_start_idx": 4,
            "dist_end_idx": 28,
        },
        "targets": [
            {
                "target_id": "target_01",
                "type": "hyperbola",
                "roi": {
                    "time_start_idx": 24,
                    "time_end_idx": 34,
                    "dist_start_idx": 12,
                    "dist_end_idx": 20,
                },
                "must_preserve": True,
            }
        ],
    }


def _reference_bscan() -> np.ndarray:
    data = np.zeros((64, 32), dtype=np.float32)
    data[24:34, 12:20] = 4.0
    data[12:52, 5:8] = 2.0
    data += 0.05
    return data


def test_truth_metrics_reward_target_preservation_and_background_suppression():
    raw = _reference_bscan()
    good = raw.copy()
    good[12:52, 5:8] *= 0.1
    good[24:34, 12:20] *= 0.95

    bad = raw.copy()
    bad[12:52, 5:8] *= 0.9
    bad[24:34, 12:20] *= 0.15

    good_metrics = compute_ground_truth_metrics(raw, good, _truth_manifest())
    bad_metrics = compute_ground_truth_metrics(raw, bad, _truth_manifest())

    assert good_metrics["truth_score"] > bad_metrics["truth_score"]
    assert (
        good_metrics["truth_target_energy_preservation"]
        > bad_metrics["truth_target_energy_preservation"]
    )
    assert (
        good_metrics["truth_background_energy_reduction"]
        > bad_metrics["truth_background_energy_reduction"]
    )
    assert (
        good_metrics["truth_false_positive_ratio"]
        < bad_metrics["truth_false_positive_ratio"]
    )
    assert good_metrics["truth_target_count"] == 1.0


def test_truth_metrics_shift_processed_target_roi_with_zero_time_roi_change():
    raw = _reference_bscan()
    shifted = np.zeros_like(raw)
    shifted[19:29, 12:20] = 3.8
    shifted[7:47, 5:8] = 0.2
    shifted += 0.05

    metrics = compute_ground_truth_metrics(
        raw,
        shifted,
        _truth_manifest(),
        reference_roi={
            "time_start_idx": 8,
            "time_end_idx": 56,
            "dist_start_idx": 4,
            "dist_end_idx": 28,
        },
        processed_roi={
            "time_start_idx": 3,
            "time_end_idx": 51,
            "dist_start_idx": 4,
            "dist_end_idx": 28,
        },
    )

    assert metrics["truth_target_energy_preservation"] > 0.75
    assert metrics["truth_target_saliency_gain"] > 1.0
    assert metrics["truth_background_energy_reduction"] > 0.5
