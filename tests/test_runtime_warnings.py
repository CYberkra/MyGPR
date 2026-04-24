#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""处理引擎结构化告警测试。"""

from __future__ import annotations

import numpy as np

from core.processing_engine import run_processing_method


def test_agc_gain_emits_clamp_and_fallback_warning():
    data = np.ones((16, 4), dtype=np.float32)
    result, meta = run_processing_method(data, "agcGain", {"window": 99})

    warnings = meta.get("runtime_warnings", [])
    codes = {item.get("code") for item in warnings}
    assert result.shape == data.shape
    assert "parameter_clamped" in codes
    assert "global_gain_fallback" in codes


def test_running_average_preserves_shape_and_emits_warning_when_window_too_large():
    data = np.arange(60, dtype=np.float32).reshape(10, 6)
    result, meta = run_processing_method(data, "running_average_2D", {"ntraces": 999})

    warnings = meta.get("runtime_warnings", [])
    codes = {item.get("code") for item in warnings}
    assert result.shape == data.shape
    assert "window_clamped" in codes


def test_normalize_result_emits_data_sanitized_warning():
    data = np.array([[1.0, np.nan], [np.inf, 2.0]], dtype=np.float32)
    result, meta = run_processing_method(data, "dewow", {"window": 2})

    warnings = meta.get("runtime_warnings", [])
    codes = {item.get("code") for item in warnings}
    assert np.isfinite(result).all()
    assert "data_sanitized" in codes
