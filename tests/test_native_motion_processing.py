#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression and integration tests for the native motion-processing backend."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from core.methods_registry import PROCESSING_METHODS
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.native_adapter import (
    NativeProcessingCatalog,
    NativeProcessingExecutor,
)


def _fixture() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(20260722)
    data = rng.normal(size=(64, 25)).astype(np.float32)
    traces = data.shape[1]
    t = np.linspace(0.0, 2.4, traces)
    metadata = {
        "trace_index": np.arange(traces, dtype=np.int32),
        "trace_timestamp_s": t,
        "trace_distance_m": np.cumsum(
            np.r_[0.0, np.clip(rng.normal(0.1, 0.012, traces - 1), 0.04, None)]
        ),
        "local_x_m": t + 0.01 * np.sin(t * 4.0),
        "local_y_m": 0.1 * np.sin(t),
        "longitude": 106.8 + np.linspace(0.0, 2.0e-4, traces) + rng.normal(0.0, 1.0e-7, traces),
        "latitude": 31.25 + np.linspace(0.0, 1.0e-4, traces) + rng.normal(0.0, 1.0e-7, traces),
        "height_agl_m": 1.5 + 0.15 * np.sin(t),
        "flight_height_m": 1.5 + 0.15 * np.sin(t),
        "height_confidence": np.ones(traces, dtype=np.float32),
        "roll_deg": 2.0 * np.sin(t),
        "pitch_deg": 1.5 * np.cos(t),
        "yaw_deg": np.linspace(0.0, 5.0, traces),
        "angular_rate_x": np.sin(t * 8.0),
        "angular_rate_y": np.cos(t * 7.0),
        "angular_rate_z": np.sin(t * 6.0),
    }
    return data, metadata, {"total_time_ns": 120.0, "time_window_ns": 120.0}


CASES = {
    "motion_compensation_height": (
        {
            "reference_height_mode": "mean",
            "manual_height": 10.0,
            "height_source": "auto",
            "compensate_amplitude": True,
            "compensate_time_shift": True,
            "wave_speed_m_per_ns": 0.299792458,
        },
        (64, 25),
        "00f2c510ab30f51d4a43b1493487add3649118a02c43e9a3910b6ef80805e658",
    ),
    "motion_compensation_speed": (
        {"spacing_m": 0.1},
        (64, 25),
        "9fa4fe5bcaab5bf26123b2b6b2c13d901a16c8626b5fb21eca8bf18b539489b7",
    ),
    "trajectory_smoothing": (
        {"method": "savgol", "window_length": 11, "polyorder": 3},
        (64, 25),
        "08ba6d8753359cfc89578a7f20d7c567e4e29be454f5d2890b6e6f37a5df5255",
    ),
    "motion_compensation_attitude": (
        {
            "apc_offset_x_m": 0.1,
            "apc_offset_y_m": -0.05,
            "apc_offset_z_m": 0.02,
            "max_abs_tilt_deg": 20.0,
        },
        (64, 25),
        "08ba6d8753359cfc89578a7f20d7c567e4e29be454f5d2890b6e6f37a5df5255",
    ),
    "motion_compensation_vibration": (
        {
            "smooth_window": 9,
            "preserve_row_percentile": 94.0,
            "preserve_mix": 0.35,
            "background_mix": 0.02,
            "max_restore_gain": 1.25,
        },
        (64, 25),
        "8c208e17904752e86f51d6205c0986e64cb1a5eaefbce83fb3d61e63e27fa9f4",
    ),
    "motion_compensation_v2": (
        {
            "height_reference_mode": "mean",
            "manual_height_m": 1.5,
            "height_source": "auto",
            "compensate_time_shift": True,
            "compensate_amplitude": True,
            "resample_spacing_m": 0.1,
            "apc_offset_x_m": 0.1,
            "apc_offset_y_m": -0.05,
            "apc_offset_z_m": 0.02,
            "max_abs_tilt_deg": 20.0,
        },
        (64, 24),
        "4857d60bdcb92919f19ee36f53e1e7b82e61353968691b191fc206b9f12f18e8",
    ),
}


@pytest.mark.parametrize("method_id", tuple(CASES))
def test_native_motion_golden_regression(method_id: str):
    data, trace_metadata, header_info = _fixture()
    params, expected_shape, expected_digest = CASES[method_id]
    result = NativeProcessingExecutor().execute(
        ProcessingRequest(
            data=data,
            method_id=method_id,
            params=params,
            header_info=header_info,
            trace_metadata=trace_metadata,
        )
    )

    assert result.data.shape == expected_shape
    assert result.data.dtype == np.float32
    assert hashlib.sha256(np.ascontiguousarray(result.data).tobytes()).hexdigest() == expected_digest
    assert result.metadata["implementation_version"] == "native-motion-2.0"
    assert result.header_info["a_scan_length"] == expected_shape[0]
    assert result.header_info["num_traces"] == expected_shape[1]


def test_vibration_receives_imu_guidance_through_processing_runtime():
    data, trace_metadata, header_info = _fixture()
    params = CASES["motion_compensation_vibration"][0]
    result = NativeProcessingExecutor().execute(
        ProcessingRequest(
            data=data,
            method_id="motion_compensation_vibration",
            params=params,
            header_info=header_info,
            trace_metadata=trace_metadata,
        )
    )

    assert result.metadata["guidance_source"] == "angular_rate_guided"
    assert result.metadata["fallback_used"] is False
    assert not np.array_equal(
        result.trace_metadata["angular_rate_x"], trace_metadata["angular_rate_x"]
    )


def test_motion_catalog_and_legacy_registry_point_to_native_implementations():
    catalog = NativeProcessingCatalog()
    for method_id in CASES:
        descriptor = catalog.get(method_id)
        assert descriptor is not None
        assert descriptor.implementation_version == "native-motion-2.0"
        assert "loaded_global" in descriptor.capabilities
        module_name = PROCESSING_METHODS[method_id]["func"].__module__
        assert module_name.startswith("mygpr.infrastructure.processing.algorithms.motion")
