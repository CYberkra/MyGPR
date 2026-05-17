#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stolt migration regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from PythonModule.stolt_migration import method_stolt_migration


def test_stolt_migration_returns_finite_array_for_small_input():
    raw = np.zeros((16, 8), dtype=np.float32)
    raw[4:8, 2:6] = 1.0

    result, meta = method_stolt_migration(raw, dx=0.05, dt=0.1, v=0.10)

    assert result.shape == raw.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert meta["mapped_params"]["dx"] == 0.05


def test_stolt_migration_rejects_non_finite_geometry_params():
    raw = np.ones((8, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="dx"):
        method_stolt_migration(raw, dx=np.nan)

    with pytest.raises(ValueError, match="pad_x"):
        method_stolt_migration(raw, pad_x=np.inf)

    with pytest.raises(ValueError, match="stolt_clip_percentile"):
        method_stolt_migration(raw, stolt_clip_percentile=np.nan)
