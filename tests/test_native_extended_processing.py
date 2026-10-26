#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coverage and golden regressions for the remaining native public methods."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from core.methods_registry import HAS_PYWAVELETS, PROCESSING_METHODS, is_public_method
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor


def _fixture():
    rng = np.random.default_rng(20260723)
    data = rng.normal(size=(72, 29)).astype(np.float32)
    data[:, 3] = 0.0
    data[7, 8] = 20.0
    traces = data.shape[1]
    distance = np.cumsum(np.r_[0.0, np.linspace(0.08, 0.12, traces - 1)])
    metadata = {
        "trace_index": np.arange(traces, dtype=np.int32),
        "trace_distance_m": distance,
        "local_x_m": distance,
        "local_y_m": np.zeros(traces),
    }
    header = {
        "total_time_ns": 144.0,
        "time_window_ns": 144.0,
        "trace_interval_m": 0.1,
    }
    return data, metadata, header


CASES = {
    "time_cut": (
        {"mode": "keep_range", "time_start_ns": 18.0, "time_end_ns": 99.0},
        (41, 29),
        "a3301db17739cffdad5438b95888140fc2dcd0a3dcd3a51cb72bd462fb9f783d",
    ),
    "trace_qc": (
        {
            "mode": "remove",
            "empty_rms_threshold": 0.001,
            "spike_zscore": 3.0,
            "manual_trace_indices": "5",
        },
        (72, 26),
        "64403af46f6db9bb501f0af5470742f2bb6729fa1e3d55c5b81b736c296ab16d",
    ),
    "energy_decay_gain": (
        {
            "strength": 0.8,
            "smoothing_samples": 11,
            "min_gain": 0.5,
            "max_gain": 5.0,
            "floor_ratio": 0.05,
        },
        (72, 29),
        "8d7274f19dfb920c183d7c9647e43f99f8862d9fc035b18f15a177f08708476f",
    ),
    "amplitude_scale": (
        {"mode": "rms", "target": 0.8},
        (72, 29),
        "94e7a3d8d8cd089591ae73d9f251ed4f36575607c1c138bc94b81ccb0cce2b79",
    ),
    "median_background_2D": (
        {
            "ntraces": 9,
            "time_start_ns": 10.0,
            "time_end_ns": 120.0,
            "edge_taper_samples": 2,
        },
        (72, 29),
        "eb0f11f866a4da414ed852fa47206169a8d8f7e4a5cade4d1ed97e6949ff0c54",
    ),
    "hilbert_envelope": (
        {"normalize": True, "log_compress": True},
        (72, 29),
        # FFT（scipy.signal.hilbert）在不同平台 BLAS 下末位浮点有差异，
        # float32 舍入后字节摘要不同；首个摘要是 Linux CI 的基准，
        # 其余是已登记的平台等价摘要（Windows / Linux py3.12），
        # 列表之外的任何摘要仍失败。
        (
            "8679b6f07793a70deff673d373874d4290d6fb57ff2a56b001aa55afb936f1e4",
            "14a9c4194c71b8d8bf59f1d2da0d15c9fae31ae1a26d4eec2378516957bca150",
            "dcee79c7e0c77b22b6ac6846dd6f30b031af68e9fcc3fa04e33150688cf31e37",
        ),
    ),
    "ccbs": (
        {},
        (72, 29),
        "cd2f02d7cca2f356bba1b24bedb77398939b549e4d60681f104627ab813376fb",
    ),
    "time_to_depth": (
        {"dt": 0.1, "v": 0.1, "dz": 0.03},
        (12, 29),
        # 同上：深插值路径对平台舍入敏感，第二个摘要是 Windows 等价摘要。
        (
            "78579e1fd7b979a27a9f90fd4ba7cb0ffb860abae6d5d94c6dc79f01955e1b85",
            "96192cc14e922399772fad20b9f24de6b4020c6d2b52c63ffc5f4ac583caf750",
        ),
    ),
}


@pytest.mark.parametrize("method_id", tuple(CASES))
def test_extended_native_golden_regression(method_id: str):
    data, trace_metadata, header_info = _fixture()
    params, expected_shape, expected_digests = CASES[method_id]
    if isinstance(expected_digests, str):
        expected_digests = (expected_digests,)
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
    assert np.isfinite(result.data).all()
    digest = hashlib.sha256(np.ascontiguousarray(result.data).tobytes()).hexdigest()
    assert digest in expected_digests, (
        f"{method_id}: unexpected golden digest {digest!r} "
        f"(expected one of {expected_digests}); if a platform digest was "
        "rebaselined deliberately, register it in CASES."
    )
    assert result.metadata["implementation_version"] == "native-extended-1.0"


def test_every_public_registry_method_has_a_native_backend():
    public_ids = {method_id for method_id in PROCESSING_METHODS if is_public_method(method_id)}
    assert public_ids <= set(NATIVE_ALGORITHMS)


def test_optional_wavelet_backend_is_lazy_and_reports_missing_dependency():
    if HAS_PYWAVELETS:
        pytest.skip("PyWavelets is installed in this environment")
    data, _, header = _fixture()
    with pytest.raises(ImportError, match="PyWavelets"):
        NativeProcessingExecutor().execute(
            ProcessingRequest(
                data=data,
                method_id="wavelet_2d",
                params={"wavelet": "db4", "levels": 2, "threshold": 0.1},
                header_info=header,
            )
        )


def test_trace_cut_keeps_inclusive_trace_range_and_aligned_metadata():
    data = np.arange(30, dtype=np.float64).reshape(3, 10)
    metadata = {
        "trace_index": np.arange(10),
        "trace_distance_m": np.linspace(2.0, 11.0, 10),
        "x_m": np.arange(10) * 3,
        "trace_flags": np.arange(20).reshape(10, 2),
        "instrument": "demo",
    }
    result = NativeProcessingExecutor().execute(
        ProcessingRequest(
            data=data,
            method_id="trace_cut",
            params={"trace_start": 3, "trace_end": 6},
            header_info={"source_trace_offset": 4, "track_length_m": 9.0, "total_time_ns": 30.0},
            trace_metadata=metadata,
        )
    )

    np.testing.assert_array_equal(result.data, data[:, 2:6])
    assert result.data.dtype == np.float32
    assert result.metadata["trace_start_index"] == 2
    assert result.metadata["trace_end_index"] == 6
    assert result.metadata["input_traces"] == 10
    assert result.metadata["output_traces"] == 4
    np.testing.assert_array_equal(result.trace_metadata["trace_index"], [2, 3, 4, 5])
    np.testing.assert_array_equal(result.trace_metadata["x_m"], [6, 9, 12, 15])
    np.testing.assert_array_equal(
        result.trace_metadata["trace_flags"], metadata["trace_flags"][2:6]
    )
    assert result.trace_metadata["instrument"] == "demo"
    assert result.header_info["track_length_m"] == pytest.approx(3.0)
    assert result.header_info["trace_interval_m"] == pytest.approx(1.0)
    assert result.header_info["source_trace_offset"] == 6
    assert result.header_info["total_time_ns"] == 30.0


def test_trace_cut_supports_single_trace_and_open_ended_default():
    data = np.arange(20, dtype=np.float32).reshape(2, 10)
    executor = NativeProcessingExecutor()
    single = executor.execute(ProcessingRequest(
        data=data, method_id="trace_cut", params={"trace_start": 4, "trace_end": 4},
    ))
    np.testing.assert_array_equal(single.data, data[:, 3:4])
    complete = executor.execute(ProcessingRequest(
        data=data, method_id="trace_cut", params={"trace_start": 1, "trace_end": 0},
    ))
    np.testing.assert_array_equal(complete.data, data)


@pytest.mark.parametrize("params", [
    {"trace_start": 0, "trace_end": 2},
    {"trace_start": 4, "trace_end": 2},
    {"trace_start": 1, "trace_end": 11},
    {"trace_start": 11, "trace_end": 0},
])
def test_trace_cut_rejects_invalid_ranges(params):
    with pytest.raises(ValueError):
        NativeProcessingExecutor().execute(ProcessingRequest(
            data=np.ones((2, 10), dtype=np.float32),
            method_id="trace_cut",
            params=params,
        ))
