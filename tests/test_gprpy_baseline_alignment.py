#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for GPRPy-style baseline processing alignment."""

from __future__ import annotations

import numpy as np

from core.gprpy_compat import (
    apply_gprpy_agc_gain,
    apply_gprpy_dewow,
    apply_gprpy_rem_mean_trace,
)
from core.processing_engine import run_processing_method


def test_dewow_matches_gprpy_baseline_windowed_edges():
    raw = np.arange(1, 49, dtype=np.float32).reshape(8, 6)

    expected = apply_gprpy_dewow(raw, 3)
    result, meta = run_processing_method(raw, "dewow", {"window": 3})

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert np.allclose(result, expected)
    assert meta["method_id"] == "dewow"


def test_rem_mean_trace_matches_gprpy_baseline_windowed_edges():
    raw = np.arange(1, 49, dtype=np.float32).reshape(8, 6)

    expected = apply_gprpy_rem_mean_trace(raw, 3)
    result, meta = run_processing_method(
        raw,
        "subtracting_average_2D",
        {"ntraces": 3},
    )

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert np.allclose(result, expected)
    assert meta["ntraces"] == 3


def test_agc_matches_gprpy_baseline_for_normal_window():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(10, 5)).astype(np.float32)

    expected = apply_gprpy_agc_gain(raw, 5)
    result, meta = run_processing_method(raw, "agcGain", {"window": 5})

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert np.allclose(result, expected)
    assert meta["method_id"] == "agcGain"
