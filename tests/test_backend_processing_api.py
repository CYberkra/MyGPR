#!/usr/bin/env python3
"""Contracts for the UI-independent backend processing façade."""
from __future__ import annotations

import numpy as np

from mygpr.domain.processing.models import (
    PipelineDefinition,
    PipelineStep,
    ProcessingRequest,
)
from mygpr.interfaces.backend import BACKEND_API_VERSION, MyGPRBackend
from tests.qt_import_isolation import assert_qt_imports_unchanged, qt_module_snapshot


def _bscan(samples: int = 64, traces: int = 24) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples)[:, None]
    x = np.linspace(0.0, 1.0, traces)[None, :]
    return (0.2 * t + np.sin(5.0 * t) + 0.05 * np.cos(3.0 * x)).astype(np.float32)


def test_backend_import_and_processing_do_not_require_qt() -> None:
    assert BACKEND_API_VERSION == "1.0"
    qt_before = qt_module_snapshot()

    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        raw = _bscan()
        methods = backend.processing.list_methods(public_only=True)
        assert any(method.method_id == "dewow" for method in methods)

        request = ProcessingRequest(raw, "dewow", {"window": 9})
        estimate = backend.processing.estimate(request)
        assert estimate.memory_bytes >= raw.nbytes
        assert estimate.relative_cost in {"low", "medium", "high", "unknown"}

        result = backend.processing.execute_method(request)
        assert result.method_id == "dewow"
        assert result.data.shape == raw.shape
        assert np.isfinite(result.data).all()
        assert result.header_info["a_scan_length"] == raw.shape[0]
        assert result.header_info["num_traces"] == raw.shape[1]
        assert_qt_imports_unchanged(qt_before)
    finally:
        backend.shutdown()


def test_backend_pipeline_preserves_lineage_and_metadata() -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        raw = _bscan()
        result = backend.processing.execute_pipeline(
            raw,
            PipelineDefinition(
                name="backend contract",
                steps=(
                    PipelineStep("dewow", {"window": 9}),
                    PipelineStep("agcGain", {"window": 11}),
                ),
            ),
            header_info={"total_time_ns": 64.0},
            trace_metadata={"distance_m": np.arange(raw.shape[1], dtype=float)},
        )
        assert [item.method_id for item in result.step_results] == ["dewow", "agcGain"]
        assert result.data.shape == raw.shape
        assert result.header_info["a_scan_length"] == raw.shape[0]
        np.testing.assert_array_equal(
            result.trace_metadata["distance_m"],
            np.arange(raw.shape[1], dtype=float),
        )
    finally:
        backend.shutdown()
