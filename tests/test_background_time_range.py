#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for background-suppression time-range parameter parsing."""

from __future__ import annotations

import numpy as np

from core.background_time_range import resolve_time_range_selection


def test_resolve_time_range_accepts_numpy_sample_bounds():
    selection = resolve_time_range_selection(
        (100, 8),
        time_start_idx=np.array([3]),
        time_end_idx=np.array([9]),
    )

    assert selection.start_idx == 3
    assert selection.end_idx == 9
    assert selection.source == "samples"


def test_resolve_time_range_accepts_numpy_time_bounds():
    selection = resolve_time_range_selection(
        (100, 8),
        time_start_ns=np.array([10.0]),
        time_end_ns=np.array([25.0]),
        time_window_ns=np.array([100.0]),
    )

    assert selection.start_idx == 10
    assert selection.end_idx == 25
    assert selection.source == "ns"


def test_resolve_time_range_ignores_empty_numpy_bounds():
    selection = resolve_time_range_selection(
        (100, 8),
        time_start_idx=np.array([], dtype=np.float64),
        time_end_idx=np.array([], dtype=np.float64),
        time_start_ns=np.array([], dtype=np.float64),
        time_end_ns=np.array([], dtype=np.float64),
    )

    assert selection.start_idx == 0
    assert selection.end_idx == 100
    assert selection.source == "full"
