#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for evidence export array helpers."""

from __future__ import annotations

import numpy as np

from core.evidence_export import _resample_columns_linear


def test_resample_columns_linear_matches_numpy_interp_per_row():
    data = np.array(
        [
            [0.0, 1.0, 3.0, 6.0],
            [2.0, 4.0, 8.0, 16.0],
            [-1.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    target_traces = 7
    source_axis = np.linspace(0.0, 1.0, data.shape[1], dtype=np.float32)
    target_axis = np.linspace(0.0, 1.0, target_traces, dtype=np.float32)
    expected = np.vstack(
        [np.interp(target_axis, source_axis, row) for row in data]
    ).astype(np.float32)

    result = _resample_columns_linear(data, target_traces)

    assert result.dtype == np.float32
    assert result.shape == (3, target_traces)
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_resample_columns_linear_returns_copy_when_trace_count_matches():
    data = np.arange(12, dtype=np.float32).reshape(3, 4)

    result = _resample_columns_linear(data, 4)
    result[0, 0] = -99.0

    assert data[0, 0] == 0.0
