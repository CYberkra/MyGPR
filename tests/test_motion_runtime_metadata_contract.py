#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for motion runtime metadata propagation."""

from __future__ import annotations

import numpy as np
import pytest

from core.processing_engine import prepare_runtime_params


def test_prepare_runtime_params_accepts_numpy_scalar_header_values():
    raw_shape = (10, 4)
    header_info = {
        "total_time_ns": np.array([100.0]),
        "track_length_m": np.array([3.0]),
        "trace_interval_m": np.array([0.5]),
    }
    trace_metadata = {
        "trace_index": np.arange(4, dtype=np.int32),
        "trace_distance_m": np.arange(4, dtype=np.float64),
    }

    time_cut_params = prepare_runtime_params(
        "time_cut",
        {},
        header_info,
        trace_metadata,
        raw_shape,
    )
    motion_params = prepare_runtime_params(
        "motion_compensation_v2",
        {},
        header_info,
        trace_metadata,
        raw_shape,
    )
    migration_params = prepare_runtime_params(
        "kirchhoff_migration",
        {},
        header_info,
        trace_metadata,
        raw_shape,
    )

    assert time_cut_params["time_step_s"] == pytest.approx(10.0e-9)
    assert motion_params["time_window_ns"] == pytest.approx(100.0)
    assert migration_params["length_m"] == pytest.approx(3.0)
