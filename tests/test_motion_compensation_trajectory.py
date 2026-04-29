#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for trajectory_smoothing non-mutating lateral geometry correction."""

from __future__ import annotations

import numpy as np
import pytest

from PythonModule.trajectory_smoothing import method_trajectory_smoothing
from core.quality_metrics import path_rmse
from core.trace_metadata_utils import derive_local_xy_m  # type: ignore[import]


def _copy_metadata(trace_metadata: dict) -> dict:
    """Deep-copy a trace_metadata dict of arrays."""
    return {
        key: np.array(value, copy=True) for key, value in trace_metadata.items()
    }


def test_trajectory_smoothing_does_not_mutate_input_metadata():
    """输入 trace_metadata 的所有数组在调用后必须保持不变。"""
    rng = np.random.default_rng(20)
    data = rng.normal(size=(64, 16)).astype(np.float32)
    trace_metadata = {
        "trace_index": np.arange(16, dtype=np.int32),
        "longitude": 116.3913 + 0.0001 * np.cumsum(rng.normal(size=16)).astype(np.float64),
        "latitude": 39.9075 + 0.0001 * np.cumsum(rng.normal(size=16)).astype(np.float64),
    }
    originals = _copy_metadata(trace_metadata)

    result, meta = method_trajectory_smoothing(
        data, trace_metadata=trace_metadata, method="savgol", window_length=5
    )

    assert meta.get("skipped") is not True
    assert result is not data
    assert np.array_equal(result, data)

    for key in originals:
        assert key in trace_metadata
        assert np.array_equal(trace_metadata[key], originals[key]), f"字段 {key} 被原地修改"


def test_trajectory_smoothing_returns_trace_metadata_updates():
    """meta 中必须包含 trace_metadata_updates 且含预期键。"""
    rng = np.random.default_rng(21)
    data = rng.normal(size=(32, 8)).astype(np.float32)
    trace_metadata = {
        "trace_index": np.arange(8, dtype=np.int32),
        "longitude": 116.3913 + 0.0002 * np.cumsum(rng.normal(size=8)).astype(np.float64),
        "latitude": 39.9075 + 0.0002 * np.cumsum(rng.normal(size=8)).astype(np.float64),
    }

    _, meta = method_trajectory_smoothing(data, trace_metadata=trace_metadata)

    updates = meta.get("trace_metadata_updates")
    assert isinstance(updates, dict)
    required_keys = {
        "longitude_raw",
        "latitude_raw",
        "longitude",
        "latitude",
        "local_x_m",
        "local_y_m",
        "trace_distance_m",
    }
    assert required_keys.issubset(set(updates.keys()))

    # trace_distance_m 应为单调非递减
    td = updates["trace_distance_m"]
    assert np.all(np.diff(td) >= -1e-12)
    assert td[0] == pytest.approx(0.0)


def test_trajectory_smoothing_preserves_existing_longitude_raw():
    """若输入已含 longitude_raw / latitude_raw，应继承而非覆盖。"""
    rng = np.random.default_rng(22)
    data = rng.normal(size=(32, 8)).astype(np.float32)
    raw_lon = np.array([116.0, 116.1, 116.2, 116.3, 116.4, 116.5, 116.6, 116.7], dtype=np.float64)
    raw_lat = np.array([39.0, 39.1, 39.2, 39.3, 39.4, 39.5, 39.6, 39.7], dtype=np.float64)
    trace_metadata = {
        "longitude": raw_lon + 0.001,
        "latitude": raw_lat + 0.001,
        "longitude_raw": raw_lon.copy(),
        "latitude_raw": raw_lat.copy(),
    }

    _, meta = method_trajectory_smoothing(data, trace_metadata=trace_metadata)

    updates = meta["trace_metadata_updates"]
    assert np.array_equal(updates["longitude_raw"], raw_lon)
    assert np.array_equal(updates["latitude_raw"], raw_lat)


def test_trajectory_smoothing_reduces_path_rmse_on_noisy_line():
    """直线上叠加高频 GPS 噪声：平滑后 path_rmse 应显著下降（>=60%）。"""
    rng = np.random.default_rng(30)
    n = 200
    # 地面真值：带缓弯的直线（模拟真实 UAV 航线低频变化）
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    gt_local_x = 50.0 * t
    gt_local_y = 2.0 * np.sin(2.0 * np.pi * 0.5 * t)

    # 高频 GPS 噪声（单频 20 周期 / 200 道 = 0.1 cycles/sample，
    # 远高于 Savitzky-Golay wl=21 的通带上限，应被大幅衰减）
    noise_freq = 20.0
    noise_amp = 0.8
    noise_x = noise_amp * np.sin(
        2.0 * np.pi * noise_freq * t + rng.uniform(0.0, 2.0 * np.pi)
    )
    noise_y = noise_amp * np.cos(
        2.0 * np.pi * noise_freq * t + rng.uniform(0.0, 2.0 * np.pi)
    )
    noise_x += 0.02 * rng.normal(size=n)
    noise_y += 0.02 * rng.normal(size=n)

    obs_local_x = gt_local_x + noise_x
    obs_local_y = gt_local_y + noise_y

    # 与 benchmark 一致的 lon/lat 换算
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(39.9075))
    lon0, lat0 = 116.3913, 39.9075
    obs_lon = lon0 + obs_local_x / meters_per_deg_lon
    obs_lat = lat0 + obs_local_y / meters_per_deg_lat
    gt_lon = lon0 + gt_local_x / meters_per_deg_lon
    gt_lat = lat0 + gt_local_y / meters_per_deg_lat

    # 预计算 obs 的 local_xy（原点为 obs 首点）
    obs_x, obs_y = derive_local_xy_m(obs_lon, obs_lat)
    # 地面真值使用与平滑输出相同的原点，保证坐标系统一
    gt_x, gt_y = derive_local_xy_m(
        gt_lon, gt_lat, origin_longitude=float(obs_lon[0]), origin_latitude=float(obs_lat[0])
    )

    trace_metadata = {
        "longitude": obs_lon.astype(np.float64),
        "latitude": obs_lat.astype(np.float64),
        "local_x_m": obs_x.astype(np.float64),
        "local_y_m": obs_y.astype(np.float64),
    }
    gt_trace_metadata: dict[str, object] = {
        "local_x_m": gt_x.astype(np.float64),
        "local_y_m": gt_y.astype(np.float64),
    }

    data = rng.normal(size=(64, n)).astype(np.float32)

    before_rmse = path_rmse(trace_metadata, gt_trace_metadata)
    assert before_rmse > 0.5, f"before_rmse={before_rmse} 应明显大于 0"

    _, meta = method_trajectory_smoothing(
        data,
        trace_metadata=trace_metadata,
        method="savgol",
        window_length=21,
    )

    assert meta.get("skipped") is not True
    assert "trace_metadata_updates" in meta

    updated_metadata: dict[str, object] = _copy_metadata(trace_metadata)
    for key, value in meta["trace_metadata_updates"].items():
        updated_metadata[key] = np.array(value, copy=True)

    after_rmse = path_rmse(updated_metadata, gt_trace_metadata)

    # 平滑后路径误差应下降至少 60%
    reduction = (before_rmse - after_rmse) / before_rmse
    assert reduction >= 0.60, (
        f"path_rmse 下降不足: before={before_rmse:.6f}, "
        f"after={after_rmse:.6f}, reduction={reduction:.4f}"
    )

    # 位移指标应为正且合理
    assert meta["max_displacement_m"] > 0.0
    assert meta["mean_displacement_m"] > 0.0
    assert meta["mean_displacement_m"] <= meta["max_displacement_m"]


def test_trajectory_smoothing_skips_missing_lon_or_lat():
    """缺少 longitude 或 latitude 时应安全跳过。"""
    rng = np.random.default_rng(23)
    data = rng.normal(size=(32, 8)).astype(np.float32)

    out, meta = method_trajectory_smoothing(data, trace_metadata=None)
    assert meta["skipped"] is True
    assert "longitude" in meta["reason"] or "latitude" in meta["reason"]
    assert np.array_equal(out, data)

    out2, meta2 = method_trajectory_smoothing(
        data, trace_metadata={"trace_index": np.arange(8, dtype=np.int32)}
    )
    assert meta2["skipped"] is True
    assert "longitude" in meta2["reason"] or "latitude" in meta2["reason"]
    assert np.array_equal(out2, data)


def test_trajectory_smoothing_skips_too_short_input():
    """少于 3 道时应安全跳过。"""
    rng = np.random.default_rng(24)
    data = rng.normal(size=(32, 2)).astype(np.float32)
    trace_metadata = {
        "longitude": np.array([116.0, 116.1], dtype=np.float64),
        "latitude": np.array([39.0, 39.1], dtype=np.float64),
    }

    out, meta = method_trajectory_smoothing(data, trace_metadata=trace_metadata)
    assert meta["skipped"] is True
    assert "过短" in meta["reason"]
    assert np.array_equal(out, data)


def test_trajectory_smoothing_skips_empty_trajectory():
    """空轨迹数组时应安全跳过。"""
    rng = np.random.default_rng(25)
    data = rng.normal(size=(32, 8)).astype(np.float32)
    trace_metadata = {
        "longitude": np.array([], dtype=np.float64),
        "latitude": np.array([], dtype=np.float64),
    }

    out, meta = method_trajectory_smoothing(data, trace_metadata=trace_metadata)
    assert meta["skipped"] is True
    assert "为空" in meta["reason"]
    assert np.array_equal(out, data)


def test_trajectory_smoothing_skips_lon_lat_length_mismatch():
    """Lon/lat length mismatch must skip instead of truncating updates."""
    rng = np.random.default_rng(250)
    data = rng.normal(size=(32, 5)).astype(np.float32)
    trace_metadata = {
        "longitude": np.array([116.0, 116.1, 116.2], dtype=np.float64),
        "latitude": np.array([39.0, 39.1, 39.2, 39.3, 39.4], dtype=np.float64),
    }

    out, meta = method_trajectory_smoothing(data, trace_metadata=trace_metadata)

    assert meta["skipped"] is True
    assert meta["metadata_length_mismatch"] is True
    assert "长度不一致" in meta["reason"]
    assert "trace_metadata_updates" not in meta
    assert np.array_equal(out, data)


def test_trajectory_smoothing_skips_trace_count_mismatch():
    """Trajectory metadata shorter than the B-scan trace count must skip safely."""
    rng = np.random.default_rng(251)
    data = rng.normal(size=(32, 5)).astype(np.float32)
    trace_metadata = {
        "longitude": np.array([116.0, 116.1, 116.2], dtype=np.float64),
        "latitude": np.array([39.0, 39.1, 39.2], dtype=np.float64),
    }

    out, meta = method_trajectory_smoothing(data, trace_metadata=trace_metadata)

    assert meta["skipped"] is True
    assert meta["metadata_length_mismatch"] is True
    assert meta["metadata_trace_count"] == 3
    assert meta["data_trace_count"] == 5
    assert "道数不一致" in meta["reason"]
    assert "trace_metadata_updates" not in meta
    assert np.array_equal(out, data)


def test_trajectory_smoothing_unsupported_method_skips():
    """不支持的平滑方法应安全跳过。"""
    rng = np.random.default_rng(26)
    data = rng.normal(size=(32, 8)).astype(np.float32)
    trace_metadata = {
        "longitude": np.array([116.0, 116.1, 116.2, 116.3, 116.4, 116.5, 116.6, 116.7], dtype=np.float64),
        "latitude": np.array([39.0, 39.1, 39.2, 39.3, 39.4, 39.5, 39.6, 39.7], dtype=np.float64),
    }

    out, meta = method_trajectory_smoothing(
        data, trace_metadata=trace_metadata, method="exponential"
    )
    assert meta["skipped"] is True
    assert "exponential" in meta["reason"]
    assert np.array_equal(out, data)


def test_trajectory_smoothing_moving_average_method_works():
    """moving_average 方法应正常产生平滑结果和 updates。"""
    rng = np.random.default_rng(27)
    data = rng.normal(size=(32, 16)).astype(np.float32)
    trace_metadata = {
        "trace_index": np.arange(16, dtype=np.int32),
        "longitude": 116.3913 + 0.0001 * np.cumsum(rng.normal(size=16)).astype(np.float64),
        "latitude": 39.9075 + 0.0001 * np.cumsum(rng.normal(size=16)).astype(np.float64),
    }

    _, meta = method_trajectory_smoothing(
        data, trace_metadata=trace_metadata, method="moving_average", window_length=5
    )

    assert meta.get("skipped") is not True
    assert "trace_metadata_updates" in meta
    assert "window_length" in meta
    assert "polyorder" not in meta
    updates = meta["trace_metadata_updates"]
    assert "local_x_m" in updates
    assert "local_y_m" in updates
    assert "trace_distance_m" in updates
