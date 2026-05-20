#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for relative background-window candidate policy."""

from __future__ import annotations

from scripts.auto_tune_validation.background_window_policy import (
    generate_relative_background_candidates,
    policy_label_for_ratio,
)


def test_generate_candidates_are_odd_dedup_clamped():
    items = generate_relative_background_candidates(
        trace_count=90,
        trace_spacing_m=0.01,
        ratio_candidates=[0.05, 0.10, 0.20, 0.40, 0.70, 1.0],
        explicit_ntraces=[73, 74, 91, 999],
    )
    values = [x.ntraces for x in items]
    assert values == sorted(values)
    assert all(v % 2 == 1 for v in values)
    assert all(3 <= v <= 89 for v in values)
    assert len(values) == len(set(values))
    assert 73 in values
    assert 89 in values
    assert all(0 < x.ntraces_ratio <= 1.0 for x in items)
    assert all(x.window_length_m is not None for x in items)


def test_label_mapping_thresholds():
    assert policy_label_for_ratio(0.05) == "local"
    assert policy_label_for_ratio(0.10) == "local"
    assert policy_label_for_ratio(0.20) == "medium"
    assert policy_label_for_ratio(0.40) == "large"
    assert policy_label_for_ratio(0.70) == "near_full_line"
    assert policy_label_for_ratio(1.00) == "full_line"
