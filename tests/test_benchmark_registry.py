#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark registry smoke tests."""

from __future__ import annotations

from core.benchmark_registry import BENCHMARK_SAMPLES
from core.benchmark_registry import (
    BENCHMARK_SCENARIOS,
    generate_benchmark_sample,
    list_benchmark_sample_ids,
)
from core.methods_registry import METHOD_METADATA


def test_benchmark_registry_exposes_expected_samples():
    sample_ids = list_benchmark_sample_ids()

    assert "zero_time_reference" in sample_ids
    assert "drift_background_reference" in sample_ids
    assert "clutter_gain_reference" in sample_ids


def test_generate_benchmark_sample_returns_deterministic_metadata():
    sample, meta = generate_benchmark_sample("drift_background_reference", seed=42)

    assert sample.ndim == 2
    assert sample.shape[0] > 100
    assert sample.shape[1] > 40
    assert meta["scenario"] in BENCHMARK_SCENARIOS
    assert meta["default_methods"] == [
        "dewow",
        "subtracting_average_2D",
        "median_background_2D",
        "rpca_background",
    ]
    assert "header_info" in meta


def test_wnnm_is_explicitly_deferred_and_excluded_from_benchmark_defaults():
    assert METHOD_METADATA["wnnm_placeholder"]["maturity"] == "deferred"
    assert METHOD_METADATA["wnnm_placeholder"]["visibility"] == "hidden"
    assert all(
        "wnnm_placeholder" not in spec.default_methods
        for spec in BENCHMARK_SAMPLES.values()
    )
