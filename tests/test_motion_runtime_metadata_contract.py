#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for motion runtime metadata propagation."""

from __future__ import annotations

import numpy as np
import pytest

from core.methods_registry import PROCESSING_METHODS
from core.workflow_data import WorkflowMethod
from core.workflow_executor import WorkflowExecutor


def _register_method(monkeypatch, method_id: str, func, *, motion: bool) -> None:
    method_info = {
        "name": method_id,
        "type": "local",
        "func": func,
        "params": [],
    }
    if motion:
        method_info["auto_tune_family"] = "motion_comp"
    monkeypatch.setitem(PROCESSING_METHODS, method_id, method_info)


def test_trace_metadata_updates_persist_to_next_workflow_step(monkeypatch):
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    header_info = {"total_time_ns": 240.0, "track_length_m": 3.0}
    trace_metadata = {
        "trace_index": np.arange(4, dtype=np.int32),
        "trace_distance_m": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    }
    source_distance = trace_metadata["trace_distance_m"].copy()
    observed: dict[str, np.ndarray | float] = {}
    updated_distance = np.array([0.25, 1.25, 2.25, 3.25], dtype=np.float64)

    def first_motion(data, trace_metadata=None, header_info=None, time_window_ns=None, **kwargs):
        assert trace_metadata is not None
        assert header_info is not None
        assert np.array_equal(trace_metadata["trace_distance_m"], source_distance)
        assert time_window_ns == pytest.approx(240.0)

        trace_metadata["trace_distance_m"][0] = -999.0
        header_info["total_time_ns"] = -1.0

        return data + 1.0, {
            "trace_metadata_updates": {"trace_distance_m": updated_distance}
        }

    def second_motion(data, trace_metadata=None, header_info=None, time_window_ns=None, **kwargs):
        assert trace_metadata is not None
        assert header_info is not None
        assert time_window_ns is not None
        observed["trace_distance_m"] = trace_metadata["trace_distance_m"].copy()
        observed["time_window_ns"] = float(time_window_ns)
        observed["header_total_time_ns"] = float(header_info["total_time_ns"])
        return data + 2.0, {}

    _register_method(monkeypatch, "test_motion_stage_one", first_motion, motion=True)
    _register_method(monkeypatch, "test_motion_stage_two", second_motion, motion=True)

    executor = WorkflowExecutor(header_info=header_info, trace_metadata=trace_metadata)
    result = executor.execute_all(
        raw,
        [
            WorkflowMethod("motion", "test_motion_stage_one"),
            WorkflowMethod("motion", "test_motion_stage_two"),
        ],
    )

    assert np.array_equal(result, raw + 3.0)
    assert np.array_equal(observed["trace_distance_m"], updated_distance)
    assert observed["time_window_ns"] == pytest.approx(240.0)
    assert observed["header_total_time_ns"] == pytest.approx(240.0)
    assert np.array_equal(executor.current_trace_metadata["trace_distance_m"], updated_distance)
    assert np.array_equal(trace_metadata["trace_distance_m"], source_distance)
    assert header_info["total_time_ns"] == pytest.approx(240.0)


def test_trace_metadata_out_replaces_runtime_metadata_without_mutating_source(monkeypatch):
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    header_info = {"total_time_ns": 120.0}
    trace_metadata = {
        "trace_distance_m": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "trace_index": np.arange(4, dtype=np.int32),
    }
    original_distance = trace_metadata["trace_distance_m"].copy()
    replacement_metadata = {
        "trace_distance_m": np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64),
        "alignment_status": np.array(["ok", "ok", "ok", "ok"], dtype="<U8"),
    }

    def motion_stage(data, trace_metadata=None, header_info=None, time_window_ns=None, **kwargs):
        assert trace_metadata is not None
        assert header_info is not None
        assert time_window_ns == pytest.approx(120.0)
        trace_metadata["trace_distance_m"][:] = -5.0
        header_info["total_time_ns"] = -10.0
        return data, {"trace_metadata_out": replacement_metadata}

    _register_method(monkeypatch, "test_motion_trace_metadata_out", motion_stage, motion=True)

    executor = WorkflowExecutor(header_info=header_info, trace_metadata=trace_metadata)
    result, _ = executor.execute_single(
        raw,
        WorkflowMethod("motion", "test_motion_trace_metadata_out"),
    )

    replacement_metadata["trace_distance_m"][0] = 999.0

    assert np.array_equal(result, raw)
    assert np.array_equal(
        executor.current_trace_metadata["trace_distance_m"],
        np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64),
    )
    assert set(executor.current_trace_metadata["alignment_status"].tolist()) == {"ok"}
    assert np.array_equal(trace_metadata["trace_distance_m"], original_distance)
    assert "alignment_status" not in trace_metadata
    assert header_info["total_time_ns"] == pytest.approx(120.0)


def test_non_motion_methods_do_not_mutate_trace_metadata(monkeypatch):
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    header_info = {"total_time_ns": 96.0}
    trace_metadata = {
        "trace_distance_m": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    }
    source_distance = trace_metadata["trace_distance_m"].copy()
    observed_kwargs: dict[str, object] = {}

    def non_motion_stage(data, **kwargs):
        observed_kwargs.update(kwargs)
        return data + 1.0, {}

    _register_method(monkeypatch, "test_non_motion_stage", non_motion_stage, motion=False)

    executor = WorkflowExecutor(header_info=header_info, trace_metadata=trace_metadata)
    result, _ = executor.execute_single(
        raw,
        WorkflowMethod("preprocessing", "test_non_motion_stage"),
    )

    assert np.array_equal(result, raw + 1.0)
    assert "trace_metadata" not in observed_kwargs
    assert "header_info" not in observed_kwargs
    assert "time_window_ns" not in observed_kwargs
    assert np.array_equal(trace_metadata["trace_distance_m"], source_distance)
    assert header_info["total_time_ns"] == pytest.approx(96.0)
