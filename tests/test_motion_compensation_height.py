#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for motion_compensation_height hardening."""

from __future__ import annotations

import numpy as np
import pytest

from PythonModule.motion_compensation_height import method_motion_compensation_height
from core.benchmark_registry import generate_benchmark_sample
from core.quality_metrics import ridge_error_metrics


def _ridge_amplitude_variance(data: np.ndarray, row_range: tuple[int, int]) -> float:
    """Variance of per-trace peak amplitudes inside a row range."""
    window = data[row_range[0] : row_range[1], :]
    peaks = np.max(np.abs(window), axis=0)
    return float(np.var(peaks))


def test_height_compensation_reduces_reflector_ridge_error():
    """Height compensation should flatten the reflector ridge to <= 1 sample RMSE."""
    raw, context = generate_benchmark_sample("motion_compensation_v1", seed=42)
    trace_metadata = context["trace_metadata"]
    gt_trace_metadata = context["ground_truth_trace_metadata"]
    ridge_row_range = tuple(context["expected_metrics"]["metric_config"]["ridge_row_range"])
    target_row_range = tuple(context["expected_metrics"]["metric_config"]["target_row_range"])

    # 原始 ridge 误差应大于 1 个样点
    raw_ridge = ridge_error_metrics(
        raw,
        gt_trace_metadata["reflector_ridge_idx"],
        row_range=ridge_row_range,
    )
    assert raw_ridge["raw_ridge_rmse_samples"] > 1.0

    corrected, meta = method_motion_compensation_height(
        raw,
        trace_metadata=trace_metadata,
        time_window_ns=trace_metadata["time_window_ns"],
        wave_speed_m_per_ns=context["wave_speed_m_per_ns"],
    )

    assert meta.get("skipped") is not True
    assert meta.get("input_height_valid") is True
    assert meta.get("amplitude_correction_applied") is True
    assert meta.get("time_shift_correction_applied") is True

    corrected_ridge = ridge_error_metrics(
        corrected,
        gt_trace_metadata["reflector_ridge_idx"],
        row_range=ridge_row_range,
    )
    # 计划阈值：residual ridge error <= 1 sample
    assert corrected_ridge["raw_ridge_rmse_samples"] <= 1.0

    # 振幅方差在反射体区域应下降 >= 40%
    raw_var = _ridge_amplitude_variance(raw, ridge_row_range)
    corr_var = _ridge_amplitude_variance(corrected, ridge_row_range)
    if raw_var > 0:
        reduction = (raw_var - corr_var) / raw_var
        assert reduction >= 0.40

    # target preservation: 目标带能量不应过度损失（>= 60%）
    from core.quality_metrics import target_preservation_ratio

    tpr = target_preservation_ratio(corrected, raw, row_range=target_row_range)
    assert tpr >= 0.60


def test_height_compensation_skips_nonpositive_heights():
    """零、负或 NaN 高度应安全跳过并返回拷贝数据。"""
    rng = np.random.default_rng(7)
    data = rng.normal(size=(64, 16)).astype(np.float32)
    base_meta = {
        "trace_index": np.arange(16, dtype=np.int32),
        "time_window_ns": 100.0,
    }

    # NaN
    nan_meta = {**base_meta, "flight_height_m": np.full(16, np.nan, dtype=np.float64)}
    out, meta = method_motion_compensation_height(data, trace_metadata=nan_meta)
    assert meta["skipped"] is True
    assert "NaN" in meta["reason"]
    assert meta["input_height_valid"] is False
    assert np.array_equal(out, data)
    assert out is not data

    # 零值
    zero_meta = {**base_meta, "flight_height_m": np.zeros(16, dtype=np.float64)}
    out, meta = method_motion_compensation_height(data, trace_metadata=zero_meta)
    assert meta["skipped"] is True
    assert "零或负值" in meta["reason"]
    assert meta["input_height_valid"] is False
    assert np.array_equal(out, data)

    # 负值
    neg_meta = {**base_meta, "flight_height_m": np.full(16, -1.5, dtype=np.float64)}
    out, meta = method_motion_compensation_height(data, trace_metadata=neg_meta)
    assert meta["skipped"] is True
    assert "零或负值" in meta["reason"]
    assert meta["input_height_valid"] is False
    assert np.array_equal(out, data)


def test_height_compensation_skips_missing_flight_height_m():
    """缺少 flight_height_m 时应安全跳过。"""
    rng = np.random.default_rng(8)
    data = rng.normal(size=(32, 8)).astype(np.float32)

    out, meta = method_motion_compensation_height(data, trace_metadata=None)
    assert meta["skipped"] is True
    assert "flight_height_m" in meta["reason"]
    assert np.array_equal(out, data)

    out2, meta2 = method_motion_compensation_height(data, trace_metadata={"time_window_ns": 50.0})
    assert meta2["skipped"] is True
    assert "flight_height_m" in meta2["reason"]
    assert np.array_equal(out2, data)


def test_height_compensation_respects_max_shift_samples():
    """max_shift_samples 应对时移进行截断并记录。"""
    rng = np.random.default_rng(9)
    samples, traces = 128, 24
    data = rng.normal(size=(samples, traces)).astype(np.float32)
    # 构造一个明显的高度变化，使得 shift > 5 samples
    heights = 2.0 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, traces))
    trace_metadata = {
        "flight_height_m": heights.astype(np.float64),
        "time_window_ns": 200.0,
    }

    _, meta_unclamped = method_motion_compensation_height(
        data,
        trace_metadata=trace_metadata,
        max_shift_samples=None,
    )
    max_shift_unclamped = meta_unclamped["max_shift_samples_applied"]
    assert max_shift_unclamped > 5.0

    clamp = 3.0
    _, meta_clamped = method_motion_compensation_height(
        data,
        trace_metadata=trace_metadata,
        max_shift_samples=clamp,
    )
    assert meta_clamped["max_shift_samples_applied"] <= clamp + 1e-6
    assert meta_clamped["shift_clamped"] is True
    assert meta_clamped["time_shift_correction_applied"] is True

    # clamp 足够大时不应触发截断
    _, meta_loose = method_motion_compensation_height(
        data,
        trace_metadata=trace_metadata,
        max_shift_samples=100.0,
    )
    assert meta_loose["shift_clamped"] is False


def test_height_compensation_rejects_unsupported_interpolation_mode():
    """V1 仅支持 linear 插值。"""
    rng = np.random.default_rng(10)
    data = rng.normal(size=(32, 8)).astype(np.float32)
    trace_metadata = {
        "flight_height_m": np.full(8, 1.5, dtype=np.float64),
        "time_window_ns": 100.0,
    }

    with pytest.raises(ValueError, match="不受支持"):
        method_motion_compensation_height(
            data,
            trace_metadata=trace_metadata,
            interpolation_mode="cubic",
        )


def test_height_compensation_time_window_from_kwargs_first():
    """kwargs 中的 time_window_ns 应优先于 trace_metadata 中的值。"""
    rng = np.random.default_rng(11)
    data = rng.normal(size=(64, 12)).astype(np.float32)
    trace_metadata = {
        "flight_height_m": np.full(12, 1.5, dtype=np.float64),
        "time_window_ns": 50.0,
    }

    _, meta = method_motion_compensation_height(
        data,
        trace_metadata=trace_metadata,
        time_window_ns=120.0,
    )
    assert meta["time_window_ns"] == pytest.approx(120.0)


def test_height_compensation_empty_flight_height_array():
    """空的 flight_height_m 数组应安全跳过。"""
    rng = np.random.default_rng(12)
    data = rng.normal(size=(32, 8)).astype(np.float32)
    trace_metadata = {
        "flight_height_m": np.array([], dtype=np.float64),
        "time_window_ns": 100.0,
    }

    out, meta = method_motion_compensation_height(data, trace_metadata=trace_metadata)
    assert meta["skipped"] is True
    assert "为空" in meta["reason"]
    assert np.array_equal(out, data)


def test_height_compensation_skips_shorter_flight_height_array_instead_of_resizing():
    """Short flight_height_m coverage must skip safely instead of inventing traces via resize."""
    rng = np.random.default_rng(120)
    data = rng.normal(size=(32, 4)).astype(np.float32)
    trace_metadata = {
        "flight_height_m": np.array([1.5, 1.6], dtype=np.float64),
        "time_window_ns": 80.0,
    }

    out, meta = method_motion_compensation_height(data, trace_metadata=trace_metadata)

    assert meta["skipped"] is True
    assert meta["input_height_valid"] is False
    assert meta["height_length_mismatch"] is True
    assert "长度与道数不一致" in meta["reason"]
    assert meta["metadata_trace_count"] == 2
    assert meta["data_trace_count"] == 4
    assert np.array_equal(out, data)


def test_height_compensation_manual_reference_height():
    """manual 参考高度模式应使用给定值。"""
    rng = np.random.default_rng(13)
    data = rng.normal(size=(64, 16)).astype(np.float32)
    heights = np.full(16, 2.0, dtype=np.float64)
    trace_metadata = {
        "flight_height_m": heights,
        "time_window_ns": 100.0,
    }

    _, meta = method_motion_compensation_height(
        data,
        trace_metadata=trace_metadata,
        reference_height_mode="manual",
        manual_height=1.5,
    )
    assert meta["reference_height_m"] == pytest.approx(1.5)
    assert meta["amplitude_correction_applied"] is True
    # 所有高度均为 2.0，参考高度 1.5，振幅因子 (1.5/2.0)^2 < 1
    # 时移因子 2*(2.0-1.5)/0.1 = 10 ns，在 100ns/63samples 下约 6.3 samples


def test_height_compensation_does_not_mutate_input():
    """输入数组不应被原地修改。"""
    rng = np.random.default_rng(14)
    data = rng.normal(size=(64, 16)).astype(np.float32)
    original = data.copy()
    trace_metadata = {
        "flight_height_m": np.full(16, 2.0, dtype=np.float64),
        "time_window_ns": 100.0,
    }

    corrected, _ = method_motion_compensation_height(data, trace_metadata=trace_metadata)
    assert np.array_equal(data, original)
    assert corrected is not data
