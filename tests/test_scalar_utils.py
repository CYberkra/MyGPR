#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scalar runtime parameter helpers."""

from __future__ import annotations

import numpy as np

from mygpr.domain.common.scalars import (
    first_scalar,
    first_two_floats,
    to_float,
    to_float_or_none,
    to_int,
    to_int_or_none,
)


def test_scalar_helpers_accept_numpy_scalar_arrays():
    assert first_scalar(np.float64(2.5)) == np.float64(2.5)
    assert to_float(np.float64(2.5), default=0.0) == 2.5
    assert to_int(np.int64(4), default=0) == 4
    assert first_scalar(np.array([3.5])) == 3.5
    assert to_float(np.array([3.5]), default=0.0) == 3.5
    assert to_int(np.array([3.5]), default=0) == 3


def test_scalar_helpers_use_defaults_for_empty_or_invalid_values():
    assert first_scalar(np.array([], dtype=np.float64)) is None
    assert to_float(np.array([], dtype=np.float64), default=2.0) == 2.0
    assert to_float(np.array([np.inf]), default=2.0) == 2.0
    assert to_int("", default=7) == 7
    assert to_int(np.array([np.inf]), default=7) == 7
    assert to_float_or_none("not-a-number") is None
    assert to_float_or_none(np.array([np.nan])) is None
    assert to_int_or_none("not-a-number") is None
    assert to_int_or_none(np.array([np.inf])) is None
    assert first_two_floats([1.0]) is None


def test_first_two_floats_requires_finite_ordered_inputs_only_for_parsing():
    assert first_two_floats(np.array([20.0, 170.0, 300.0])) == (20.0, 170.0)
    assert first_two_floats(np.array([np.nan, 170.0])) is None
