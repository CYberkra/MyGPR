#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for unified UAV-GPR motion compensation V2."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from PythonModule.motion_compensation_v2 import method_motion_compensation_v2


AIR_WAVE_SPEED_M_PER_NS = 0.299792458


def _base_metadata(trace_count: int = 5) -> dict[str, np.ndarray | float]:
    distance = np.linspace(0.0, 4.0, trace_count, dtype=np.float64)
    return {
        "trace_index": np.arange(trace_count, dtype=np.int32),
        "trace_timestamp_s": np.linspace(0.0, 0.4, trace_count, dtype=np.float64),
        "trace_distance_m": distance.copy(),
        "local_x_m": distance.copy(),
        "local_y_m": np.zeros(trace_count, dtype=np.float64),
        "roll_deg": np.linspace(-2.0, 2.0, trace_count, dtype=np.float64),
        "pitch_deg": np.linspace(1.0, -1.0, trace_count, dtype=np.float64),
        "yaw_deg": np.zeros(trace_count, dtype=np.float64),
        "height_agl_m": np.linspace(1.2, 1.8, trace_count, dtype=np.float64),
        "height_source": np.full(trace_count, "nar15", dtype="<U8"),
        "height_confidence": np.ones(trace_count, dtype=np.float32),
        "time_window_ns": 120.0,
    }


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for key, value in metadata.items():
        copied[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value)
    return copied


def test_v2_uses_agl_height_and_air_velocity_for_time_shift():
    data = np.zeros((64, 5), dtype=np.float32)
    metadata = _base_metadata(5)
    height = np.asarray(metadata["height_agl_m"], dtype=np.float64)
    expected_ref = float(np.mean(height))
    expected_shift_ns = 2.0 * (height - expected_ref) / AIR_WAVE_SPEED_M_PER_NS

    corrected, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        time_window_ns=120.0,
        compensate_time_shift=True,
        compensate_amplitude=False,
    )

    assert corrected.shape == data.shape
    assert meta.get("skipped") is not True
    assert meta["height_source_used"] == "height_agl_m"
    assert meta["height_reference_m"] == pytest.approx(expected_ref)
    assert meta["air_wave_speed_m_per_ns"] == pytest.approx(AIR_WAVE_SPEED_M_PER_NS)
    assert np.allclose(meta["time_shift_ns"], expected_shift_ns)
    assert meta["time_shift_correction_applied"] is True


def test_v2_falls_back_to_flight_height_with_warning():
    data = np.zeros((32, 4), dtype=np.float32)
    metadata = _base_metadata(4)
    metadata.pop("height_agl_m")
    metadata.pop("height_source")
    metadata.pop("height_confidence")
    metadata["flight_height_m"] = np.linspace(1.0, 1.3, 4, dtype=np.float64)

    _, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        compensate_time_shift=False,
        compensate_amplitude=False,
    )

    warning_codes = {item["code"] for item in meta.get("runtime_warnings", [])}
    assert meta.get("skipped") is not True
    assert meta["height_source_used"] == "flight_height_m"
    assert "height_source_fallback" in warning_codes
    assert "height_from_legacy_flight_height" in meta["quality_flags"]


def test_v2_missing_height_skips_height_correction_without_failing():
    data = np.arange(24, dtype=np.float32).reshape(6, 4)
    metadata = {
        "trace_index": np.arange(4, dtype=np.int32),
        "trace_distance_m": np.arange(4, dtype=np.float64),
    }

    corrected, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        compensate_time_shift=True,
        compensate_amplitude=True,
    )

    assert np.array_equal(corrected, data)
    assert corrected is not data
    assert meta.get("skipped") is not True
    assert meta["height_correction_applied"] is False
    assert "missing_height_agl" in meta["quality_flags"]


def test_v2_nonpositive_height_skips_height_correction():
    data = np.ones((12, 4), dtype=np.float32)
    metadata = _base_metadata(4)
    metadata["height_agl_m"] = np.array([1.2, 0.0, 1.4, 1.6], dtype=np.float64)

    corrected, meta = method_motion_compensation_v2(data, trace_metadata=metadata)

    assert np.array_equal(corrected, data)
    assert meta["height_correction_applied"] is False
    assert "invalid_height_agl" in meta["quality_flags"]
    assert "trace_metadata_updates" not in meta


def test_v2_flags_extrapolated_alignment_and_low_height_confidence():
    data = np.zeros((16, 4), dtype=np.float32)
    metadata = _base_metadata(4)
    metadata["alignment_status"] = np.array(
        ["aligned", "extrapolated", "aligned", "aligned"], dtype="<U16"
    )
    metadata["height_confidence"] = np.array([1.0, 0.4, 0.9, 0.8], dtype=np.float32)

    _, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        compensate_time_shift=False,
        compensate_amplitude=False,
    )

    warning_codes = {item["code"] for item in meta.get("runtime_warnings", [])}
    assert "sidecar_extrapolated" in meta["quality_flags"]
    assert "low_height_confidence" in meta["quality_flags"]
    assert "sidecar_extrapolated" in warning_codes
    assert "low_height_confidence" in warning_codes
    assert meta["input_quality"]["alignment_extrapolated_traces"] == 1
    assert meta["input_quality"]["height_confidence_low_traces"] == 1


def test_v2_warns_when_height_time_shift_is_clamped():
    data = np.zeros((64, 4), dtype=np.float32)
    metadata = _base_metadata(4)
    metadata["height_agl_m"] = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    _, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        time_window_ns=10.0,
        compensate_time_shift=True,
        compensate_amplitude=False,
        max_shift_samples=0.5,
    )

    warning_codes = {item["code"] for item in meta.get("runtime_warnings", [])}
    assert meta["time_shift_clamped"] is True
    assert "time_shift_clamped" in meta["quality_flags"]
    assert "time_shift_clamped" in warning_codes
    assert meta["raw_time_shift_samples_min"] < -0.5
    assert meta["raw_time_shift_samples_max"] > 0.5


def test_v2_uses_time_window_shift_limit_when_sample_limit_is_auto():
    data = np.zeros((64, 4), dtype=np.float32)
    metadata = _base_metadata(4)
    metadata["height_agl_m"] = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    _, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        time_window_ns=10.0,
        compensate_time_shift=True,
        compensate_amplitude=False,
        max_shift_samples=0.0,
        max_shift_ns=0.5,
    )

    expected_limit = 0.5 / (10.0 / 63.0)
    assert meta["max_shift_samples_effective"] == pytest.approx(expected_limit)
    assert meta["max_shift_limit_source"] == "max_shift_ns"
    assert meta["time_shift_clamped"] is True
    assert np.max(np.abs(meta["time_shift_samples"])) <= expected_limit + 1.0e-6


def test_v2_flags_timestamp_distance_gap_and_speed_outlier():
    data = np.zeros((32, 6), dtype=np.float32)
    metadata = _base_metadata(6)
    metadata["trace_timestamp_s"] = np.array([0.0, 0.1, 0.2, 0.3, 2.0, 2.1])
    metadata["trace_distance_m"] = np.array([0.0, 0.1, 0.2, 0.3, 5.0, 5.1])
    metadata["local_x_m"] = metadata["trace_distance_m"].copy()

    _, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        compensate_time_shift=False,
        compensate_amplitude=False,
    )

    warning_codes = {item["code"] for item in meta.get("runtime_warnings", [])}
    assert "trace_timestamp_gap" in meta["quality_flags"]
    assert "trace_distance_gap" in meta["quality_flags"]
    assert "trace_speed_outlier" in meta["quality_flags"]
    assert "trace_timestamp_gap" in warning_codes
    assert "trace_distance_gap" in warning_codes
    assert "trace_speed_outlier" in warning_codes
    assert meta["input_quality"]["trace_timestamp_gap_ratio"] > 3.0
    assert meta["input_quality"]["trace_distance_gap_ratio"] > 3.0
    assert meta["input_quality"]["trace_speed_max_mps"] > meta["input_quality"]["trace_speed_median_mps"]


def test_v2_flags_nonmonotonic_trace_timestamps():
    data = np.zeros((16, 4), dtype=np.float32)
    metadata = _base_metadata(4)
    metadata["trace_timestamp_s"] = np.array([0.0, 0.2, 0.1, 0.3])

    _, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        compensate_time_shift=False,
        compensate_amplitude=False,
    )

    warning_codes = {item["code"] for item in meta.get("runtime_warnings", [])}
    assert "trace_timestamp_nonmonotonic" in meta["quality_flags"]
    assert "trace_timestamp_nonmonotonic" in warning_codes
    assert meta["input_quality"]["trace_timestamp_nonpositive_steps"] == 1


def test_v2_resamples_data_and_metadata_to_uniform_trace_spacing():
    source_distance = np.array([0.0, 0.4, 1.1, 2.0], dtype=np.float64)
    data = np.vstack(
        [source_distance, source_distance**2, source_distance + 2.0]
    ).astype(np.float32)
    metadata = _base_metadata(4)
    metadata["trace_distance_m"] = source_distance.copy()
    metadata["local_x_m"] = source_distance.copy()
    metadata["local_y_m"] = np.zeros(4, dtype=np.float64)
    metadata["roll_deg"] = np.zeros(4, dtype=np.float64)
    metadata["pitch_deg"] = np.zeros(4, dtype=np.float64)
    metadata["yaw_deg"] = np.zeros(4, dtype=np.float64)

    corrected, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        compensate_time_shift=False,
        compensate_amplitude=False,
        resample_spacing_m=0.5,
    )

    trace_metadata_out = meta["trace_metadata_out"]
    out_distance = np.asarray(trace_metadata_out["trace_distance_m"], dtype=np.float64)
    assert corrected.shape == (3, out_distance.size)
    assert np.allclose(out_distance, np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert len(trace_metadata_out["trace_index"]) == corrected.shape[1]
    assert len(trace_metadata_out["height_agl_m"]) == corrected.shape[1]
    assert meta["resampling_applied"] is True


def test_v2_does_not_mutate_input_data_or_metadata():
    rng = np.random.default_rng(123)
    data = rng.normal(size=(32, 6)).astype(np.float32)
    metadata = _base_metadata(6)
    metadata["trace_distance_m"] = np.array([0.0, 0.5, 1.3, 2.2, 3.1, 4.0])
    metadata["local_x_m"] = np.asarray(metadata["trace_distance_m"]).copy()
    original_data = data.copy()
    original_metadata = _copy_metadata(metadata)

    corrected, meta = method_motion_compensation_v2(
        data,
        trace_metadata=metadata,
        compensate_time_shift=True,
        compensate_amplitude=True,
        resample_spacing_m=0.5,
    )

    assert corrected is not data
    assert meta.get("skipped") is not True
    assert np.array_equal(data, original_data)
    for key, original in original_metadata.items():
        current = metadata[key]
        if isinstance(original, np.ndarray):
            assert np.array_equal(current, original), key
        else:
            assert current == original
