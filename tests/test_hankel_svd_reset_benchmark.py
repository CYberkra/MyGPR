#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Hankel-SVD reset benchmark artifact serialization."""

from __future__ import annotations

import json

import numpy as np

from scripts.benchmark_hankel_svd_reset import _to_jsonable


def test_hankel_benchmark_jsonable_removes_nonfinite_values():
    payload = {
        "float": float("inf"),
        "numpy_float": np.float64(np.nan),
        "nested": [np.float32(-np.inf), 1.25],
    }

    safe = _to_jsonable(payload)

    json.dumps(safe, allow_nan=False)
    assert safe == {
        "float": None,
        "numpy_float": None,
        "nested": [None, 1.25],
    }
