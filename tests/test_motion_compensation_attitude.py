#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for metadata-only attitude/APC footprint correction."""

from __future__ import annotations

import numpy as np

from PythonModule.motion_compensation_attitude import method_motion_compensation_attitude  # type: ignore[import]
from core.quality_metrics import footprint_rmse


def _copy_metadata(trace_metadata: dict) -> dict:
    copied: dict[str, object] = {}
    for key, values in trace_metadata.items():
        copied[key] = np.array(values, copy=True) if isinstance(values, np.ndarray) else values
    return copied


def _expected_updates(
    trace_metadata: dict,
    *,
    apc_offset_x_m: float = 0.0,
    apc_offset_y_m: float = 0.0,
    apc_offset_z_m: float = 0.0,
    max_abs_tilt_deg: float = 20.0,
) -> dict[str, np.ndarray]:
    local_x_m = np.asarray(trace_metadata["local_x_m"], dtype=np.float64)
    local_y_m = np.asarray(trace_metadata["local_y_m"], dtype=np.float64)
    yaw_rad = np.deg2rad(np.asarray(trace_metadata["yaw_deg"], dtype=np.float64))
    roll_deg = np.asarray(trace_metadata["roll_deg"], dtype=np.float64)
    pitch_deg = np.asarray(trace_metadata["pitch_deg"], dtype=np.float64)
    roll_used_rad = np.deg2rad(np.clip(roll_deg, -max_abs_tilt_deg, max_abs_tilt_deg))
    pitch_used_rad = np.deg2rad(np.clip(pitch_deg, -max_abs_tilt_deg, max_abs_tilt_deg))

    apc_x = apc_offset_x_m * np.cos(yaw_rad) - apc_offset_y_m * np.sin(yaw_rad)
    apc_y = apc_offset_x_m * np.sin(yaw_rad) + apc_offset_y_m * np.cos(yaw_rad)

    if "flight_height_m" in trace_metadata:
        projection_height_m = (
            np.asarray(trace_metadata["flight_height_m"], dtype=np.float64) + apc_offset_z_m
        )
    else:
        projection_height_m = np.zeros(local_x_m.size, dtype=np.float64)

    pitch_body_m = projection_height_m * np.tan(pitch_used_rad)
    roll_body_m = projection_height_m * np.tan(roll_used_rad)
    footprint_dx = pitch_body_m * np.cos(yaw_rad) - roll_body_m * np.sin(yaw_rad)
    footprint_dy = pitch_body_m * np.sin(yaw_rad) + roll_body_m * np.cos(yaw_rad)

    corrected_x = local_x_m + apc_x + footprint_dx
    corrected_y = local_y_m + apc_y + footprint_dy
    distance = np.concatenate(
        ([0.0], np.cumsum(np.sqrt(np.diff(corrected_x) ** 2 + np.diff(corrected_y) ** 2)))
    ).astype(np.float64)
    return {
        "local_x_m": corrected_x,
        "local_y_m": corrected_y,
        "trace_distance_m": distance,
        "footprint_x_m": corrected_x.copy(),
        "footprint_y_m": corrected_y.copy(),
    }


def test_attitude_compensation_zero_lever_arm_preserves_amplitudes():
    """Zero lever arm should keep amplitudes unchanged and emit deterministic metadata updates."""
    data = np.arange(30, dtype=np.float32).reshape(6, 5)
    trace_metadata = {
        "local_x_m": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "local_y_m": np.array([0.0, 0.2, 0.1, -0.1, -0.2], dtype=np.float64),
        "roll_deg": np.array([0.0, 2.0, -1.0, 0.5, -0.5], dtype=np.float64),
        "pitch_deg": np.array([1.0, 0.5, -2.0, 0.0, 1.5], dtype=np.float64),
        "yaw_deg": np.array([0.0, 15.0, 30.0, 45.0, 60.0], dtype=np.float64),
        "flight_height_m": np.full(5, 1.8, dtype=np.float64),
    }

    corrected_a, meta_a = method_motion_compensation_attitude(data, trace_metadata=trace_metadata)
    corrected_b, meta_b = method_motion_compensation_attitude(data, trace_metadata=trace_metadata)

    expected = _expected_updates(trace_metadata)

    assert np.array_equal(corrected_a, data)
    assert np.array_equal(corrected_b, data)
    assert corrected_a is not data
    assert corrected_b is not data
    assert meta_a.get("skipped") is not True
    assert meta_b.get("skipped") is not True
    assert set(meta_a["trace_metadata_updates"].keys()) == {
        "local_x_m",
        "local_y_m",
        "trace_distance_m",
        "footprint_x_m",
        "footprint_y_m",
    }

    for key, expected_values in expected.items():
        assert np.allclose(meta_a["trace_metadata_updates"][key], expected_values)
        assert np.array_equal(meta_a["trace_metadata_updates"][key], meta_b["trace_metadata_updates"][key])


def test_attitude_compensation_nonzero_lever_arm_reduces_footprint_rmse():
    """Synthetic APC offset case should improve corrected-footprint RMSE by at least 50%."""
    n = 64
    phase = np.linspace(0.0, 1.0, n, dtype=np.float64)
    data = np.zeros((32, n), dtype=np.float32)
    trace_metadata = {
        "local_x_m": np.linspace(0.0, 30.0, n, dtype=np.float64),
        "local_y_m": 0.4 * np.sin(2.0 * np.pi * phase),
        "roll_deg": 3.0 * np.sin(2.0 * np.pi * 0.8 * phase),
        "pitch_deg": 2.5 * np.cos(2.0 * np.pi * 1.1 * phase),
        "yaw_deg": 12.0 * np.sin(2.0 * np.pi * 0.4 * phase),
        "flight_height_m": 1.6 + 0.05 * np.cos(2.0 * np.pi * phase),
    }
    target_updates = _expected_updates(
        trace_metadata,
        apc_offset_x_m=0.35,
        apc_offset_y_m=-0.18,
        apc_offset_z_m=0.12,
    )
    target_metadata: dict[str, object] = {
        "footprint_x_m": target_updates["footprint_x_m"],
        "footprint_y_m": target_updates["footprint_y_m"],
    }

    before_input: dict[str, object] = {
        "footprint_x_m": np.asarray(trace_metadata["local_x_m"], dtype=np.float64),
        "footprint_y_m": np.asarray(trace_metadata["local_y_m"], dtype=np.float64),
    }
    before_rmse = footprint_rmse(before_input, target_metadata)
    assert before_rmse > 0.10

    _, meta = method_motion_compensation_attitude(
        data,
        trace_metadata=trace_metadata,
        apc_offset_x_m=0.35,
        apc_offset_y_m=-0.18,
        apc_offset_z_m=0.12,
    )

    assert meta.get("skipped") is not True
    updates = meta["trace_metadata_updates"]
    after_input: dict[str, object] = {
        "footprint_x_m": np.asarray(updates["footprint_x_m"], dtype=np.float64),
        "footprint_y_m": np.asarray(updates["footprint_y_m"], dtype=np.float64),
    }
    after_rmse = footprint_rmse(after_input, target_metadata)
    reduction = (before_rmse - after_rmse) / before_rmse
    assert reduction >= 0.50


def test_attitude_compensation_skips_missing_imu_fields():
    """Missing attitude fields should skip safely with an explicit reason."""
    data = np.ones((8, 4), dtype=np.float32)
    trace_metadata = {
        "local_x_m": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "local_y_m": np.zeros(4, dtype=np.float64),
        "roll_deg": np.zeros(4, dtype=np.float64),
        "flight_height_m": np.full(4, 1.5, dtype=np.float64),
    }

    corrected, meta = method_motion_compensation_attitude(data, trace_metadata=trace_metadata)

    assert meta["skipped"] is True
    assert "pitch_deg" in meta["reason"]
    assert "yaw_deg" in meta["reason"]
    assert "trace_metadata_updates" not in meta
    assert np.array_equal(corrected, data)
    assert corrected is not data


def test_attitude_compensation_clamps_excessive_tilt_with_warning_and_provenance():
    """Excessive roll/pitch spikes should be clamped deterministically with warnings."""
    data = np.zeros((16, 4), dtype=np.float32)
    trace_metadata = {
        "local_x_m": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "local_y_m": np.zeros(4, dtype=np.float64),
        "roll_deg": np.array([0.0, 45.0, -35.0, 1.0], dtype=np.float64),
        "pitch_deg": np.array([0.0, 5.0, 28.0, -40.0], dtype=np.float64),
        "yaw_deg": np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64),
        "flight_height_m": np.full(4, 1.7, dtype=np.float64),
    }

    _, meta = method_motion_compensation_attitude(
        data,
        trace_metadata=trace_metadata,
        max_abs_tilt_deg=20.0,
    )

    expected = _expected_updates(trace_metadata, max_abs_tilt_deg=20.0)

    assert meta.get("skipped") is not True
    assert meta["attitude_clamped"] is True
    assert meta["clamped_trace_count"] == 3
    assert meta["warnings"]
    assert "钳制" in meta["warnings"][0]
    assert meta["provenance"]["attitude_handling"] == "clamp"
    assert meta["projection_height_source"] == "flight_height_m"
    assert np.allclose(meta["trace_metadata_updates"]["local_x_m"], expected["local_x_m"])
    assert np.allclose(meta["trace_metadata_updates"]["local_y_m"], expected["local_y_m"])


def test_attitude_compensation_does_not_mutate_input_metadata():
    """Input trace_metadata arrays should remain unchanged after correction."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(16, 6)).astype(np.float32)
    trace_metadata = {
        "local_x_m": np.linspace(0.0, 5.0, 6, dtype=np.float64),
        "local_y_m": rng.normal(scale=0.1, size=6).astype(np.float64),
        "roll_deg": rng.normal(scale=2.0, size=6).astype(np.float64),
        "pitch_deg": rng.normal(scale=2.0, size=6).astype(np.float64),
        "yaw_deg": rng.normal(scale=5.0, size=6).astype(np.float64),
        "flight_height_m": np.full(6, 1.6, dtype=np.float64),
    }
    original_metadata = _copy_metadata(trace_metadata)
    original_data = data.copy()

    corrected, meta = method_motion_compensation_attitude(data, trace_metadata=trace_metadata)

    assert meta.get("skipped") is not True
    assert np.array_equal(data, original_data)
    assert corrected is not data
    assert np.array_equal(corrected, data)
    for key, original_value in original_metadata.items():
        assert np.array_equal(trace_metadata[key], original_value), f"字段 {key} 被原地修改"


def test_attitude_compensation_skips_invalid_projection_height():
    """Non-positive projection height should skip without inventing metadata updates."""
    data = np.zeros((12, 3), dtype=np.float32)
    trace_metadata = {
        "local_x_m": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        "local_y_m": np.zeros(3, dtype=np.float64),
        "roll_deg": np.zeros(3, dtype=np.float64),
        "pitch_deg": np.zeros(3, dtype=np.float64),
        "yaw_deg": np.zeros(3, dtype=np.float64),
        "flight_height_m": np.full(3, 0.05, dtype=np.float64),
    }

    corrected, meta = method_motion_compensation_attitude(
        data,
        trace_metadata=trace_metadata,
        apc_offset_z_m=-0.10,
    )

    assert meta["skipped"] is True
    assert "投影高度" in meta["reason"]
    assert "trace_metadata_updates" not in meta
    assert np.array_equal(corrected, data)
