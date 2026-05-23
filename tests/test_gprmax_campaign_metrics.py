#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for synthetic paired gprMax metrics."""

from __future__ import annotations

import numpy as np
import pytest

from core.gprmax_campaign.metrics import compute_paired_metrics


def test_compute_metrics_basic_expected_values():
    raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    background = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    target = raw - background
    metrics = compute_paired_metrics(raw, background, target)

    assert metrics["raw_shape"] == [2, 2]
    assert metrics["background_shape"] == [2, 2]
    assert metrics["target_response_shape"] == [2, 2]
    assert metrics["raw_energy"] == pytest.approx(30.0)
    assert metrics["background_energy"] == pytest.approx(6.0)
    assert metrics["target_response_energy"] == pytest.approx(10.0)
    assert metrics["target_to_background_energy_ratio"] == pytest.approx(10.0 / 6.0)
    assert metrics["raw_background_mae"] == pytest.approx(np.mean(np.abs(target)))
    assert metrics["raw_background_mse"] == pytest.approx(np.mean(np.square(target)))
    assert metrics["raw_background_rmse"] == pytest.approx(np.sqrt(metrics["raw_background_mse"]))


def test_compute_metrics_shape_mismatch_raises():
    raw = np.ones((2, 2), dtype=np.float64)
    background = np.ones((2, 3), dtype=np.float64)
    target = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="same shape"):
        compute_paired_metrics(raw, background, target)


def test_zero_denominator_warning_and_null_ratio():
    raw = np.zeros((2, 2), dtype=np.float64)
    background = np.zeros((2, 2), dtype=np.float64)
    target = np.ones((2, 2), dtype=np.float64)
    metrics = compute_paired_metrics(raw, background, target)
    assert metrics["target_to_background_energy_ratio"] is None
    assert metrics["target_to_raw_energy_ratio"] is None
    warning_codes = {item["code"] for item in metrics["warnings"]}
    assert "target_to_background_energy_ratio_denominator_zero" in warning_codes
    assert "target_to_raw_energy_ratio_denominator_zero" in warning_codes


def test_mse_zero_psnr_handling():
    raw = np.ones((3, 1), dtype=np.float64)
    background = np.ones((3, 1), dtype=np.float64)
    target = raw - background
    metrics = compute_paired_metrics(raw, background, target)
    assert metrics["raw_background_mse"] == pytest.approx(0.0)
    assert metrics["raw_background_psnr"] is None
    warning_codes = {item["code"] for item in metrics["warnings"]}
    assert "raw_background_psnr_mse_zero" in warning_codes


def test_roi_valid_ratio():
    raw = np.ones((4, 4), dtype=np.float64)
    background = np.zeros((4, 4), dtype=np.float64)
    target = raw - background
    roi = {"sample_range": [0, 2], "trace_range": [0, 2]}
    metrics = compute_paired_metrics(raw, background, target, roi=roi)
    assert metrics["roi_energy_ratio"] == pytest.approx(4.0 / 16.0)


def test_roi_invalid_warns():
    raw = np.ones((4, 4), dtype=np.float64)
    background = np.zeros((4, 4), dtype=np.float64)
    target = raw - background
    roi = {"sample_range": [3, 1], "trace_range": [0, 2]}
    metrics = compute_paired_metrics(raw, background, target, roi=roi)
    assert metrics["roi_energy_ratio"] is None
    warning_codes = {item["code"] for item in metrics["warnings"]}
    assert "roi_out_of_bounds" in warning_codes


def test_single_column_2d_supported():
    raw = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    background = np.array([[0.5], [1.0], [1.5]], dtype=np.float64)
    target = raw - background
    metrics = compute_paired_metrics(raw, background, target)
    assert metrics["target_response_shape"] == [3, 1]
