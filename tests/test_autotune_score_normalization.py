#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for core.autotune_score_normalization — deterministic scoring helpers."""

from __future__ import annotations

import math

import pytest

from core.autotune_score_normalization import (
    clamp01,
    logistic01,
    norm01,
    normalize_weights,
    ratio_similarity,
    weighted_sum,
)


# ── clamp01 ─────────────────────────────────────────────────────────────────

class TestClamp01:
    def test_zero_stays_zero(self) -> None:
        assert clamp01(0.0) == 0.0

    def test_one_stays_one(self) -> None:
        assert clamp01(1.0) == 1.0

    def test_half_stays_half(self) -> None:
        assert clamp01(0.5) == 0.5

    def test_negative_clamped_to_zero(self) -> None:
        assert clamp01(-0.1) == 0.0

    def test_above_one_clamped_to_one(self) -> None:
        assert clamp01(1.5) == 1.0

    def test_large_negative_clamped(self) -> None:
        assert clamp01(-999.0) == 0.0

    def test_large_positive_clamped(self) -> None:
        assert clamp01(999.0) == 1.0

    def test_nan_returns_zero(self) -> None:
        assert clamp01(float("nan")) == 0.0

    def test_inf_returns_one(self) -> None:
        assert clamp01(float("inf")) == 1.0

    def test_negative_inf_returns_zero(self) -> None:
        assert clamp01(float("-inf")) == 0.0

    def test_string_input_returns_zero(self) -> None:
        assert clamp01("not a number") == 0.0  # type: ignore[arg-type]

    def test_none_input_returns_zero(self) -> None:
        assert clamp01(None) == 0.0  # type: ignore[arg-type]

    def test_integer_input_works(self) -> None:
        assert clamp01(0) == 0.0
        assert clamp01(1) == 1.0


# ── norm01 ─────────────────────────────────────────────────────────────────

class TestNorm01:
    def test_lo_maps_to_zero(self) -> None:
        assert norm01(0.0, 0.0, 10.0) == 0.0

    def test_hi_maps_to_one(self) -> None:
        assert norm01(10.0, 0.0, 10.0) == 1.0

    def test_midpoint_maps_to_half(self) -> None:
        assert norm01(5.0, 0.0, 10.0) == 0.5

    def test_below_lo_clamped(self) -> None:
        assert norm01(-5.0, 0.0, 10.0) == 0.0

    def test_above_hi_clamped(self) -> None:
        assert norm01(15.0, 0.0, 10.0) == 1.0

    def test_equal_bounds_returns_zero(self) -> None:
        assert norm01(5.0, 5.0, 5.0) == 0.0

    def test_inverted_bounds_returns_zero(self) -> None:
        """hi < lo should return 0.0 safely."""
        assert norm01(5.0, 10.0, 0.0) == 0.0

    def test_negative_range(self) -> None:
        """Works with negative ranges."""
        assert norm01(-5.0, -10.0, 0.0) == 0.5


# ── logistic01 ───────────────────────────────────────────────────────────────

class TestLogistic01:
    def test_center_returns_half(self) -> None:
        assert logistic01(0.0, center=0.0, slope=1.0) == pytest.approx(0.5)

    def test_positive_input_above_half(self) -> None:
        result = logistic01(2.0, center=0.0, slope=1.0)
        assert result > 0.5

    def test_negative_input_below_half(self) -> None:
        result = logistic01(-2.0, center=0.0, slope=1.0)
        assert result < 0.5

    def test_large_positive_approaches_one(self) -> None:
        result = logistic01(60.0, center=0.0, slope=1.0)
        assert result > 0.999

    def test_large_negative_approaches_zero(self) -> None:
        result = logistic01(-60.0, center=0.0, slope=1.0)
        assert result < 0.001

    def test_output_is_bounded_0_1(self) -> None:
        for val in [-100.0, -10.0, 0.0, 10.0, 100.0]:
            result = logistic01(val)
            assert 0.0 <= result <= 1.0

    def test_higher_slope_steepens(self) -> None:
        """Higher slope should push values further from 0.5."""
        gentle = logistic01(1.0, slope=1.0)
        steep = logistic01(1.0, slope=5.0)
        assert steep > gentle  # steeper slope pushes further toward 1

    def test_different_center_shifts(self) -> None:
        """Shifting center changes where 0.5 occurs."""
        at_center = logistic01(3.0, center=3.0, slope=1.0)
        assert at_center == pytest.approx(0.5)

    def test_clamped_at_60_boundary(self) -> None:
        """Input beyond ±60 is clamped internally."""
        huge = logistic01(1e6)
        capped = logistic01(60.0)
        assert huge == pytest.approx(capped)

    def test_default_slope_and_center(self) -> None:
        """Should work with no kwargs."""
        result = logistic01(0.0)
        assert 0.0 <= result <= 1.0


# ── ratio_similarity ────────────────────────────────────────────────────────

class TestRatioSimilarity:
    def test_exact_match_returns_one(self) -> None:
        assert ratio_similarity(1.0) == 1.0

    def test_double_ratio_with_default_tolerance(self) -> None:
        result = ratio_similarity(2.0)  # tolerance_ratio=3.0 default
        assert 0.0 < result < 1.0

    def test_half_ratio_symmetry(self) -> None:
        """Ratio of 0.5 and 2.0 should give the same similarity (log symmetry)."""
        sim_half = ratio_similarity(0.5)
        sim_double = ratio_similarity(2.0)
        assert sim_half == pytest.approx(sim_double)

    def test_at_tolerance_ratio_returns_zero(self) -> None:
        """At exactly tolerance_ratio, similarity should be 0."""
        assert ratio_similarity(3.0, tolerance_ratio=3.0) == pytest.approx(0.0, abs=1e-10)

    def test_beyond_tolerance_returns_zero(self) -> None:
        assert ratio_similarity(5.0, tolerance_ratio=3.0) == 0.0

    def test_zero_ratio_guarded(self) -> None:
        """Zero ratio is guarded to 1e-12 to avoid log(0)."""
        result = ratio_similarity(0.0)
        assert 0.0 <= result <= 1.0

    def test_tiny_tolerance_guard(self) -> None:
        """tolerance_ratio below 1.01 is clamped."""
        result = ratio_similarity(2.0, tolerance_ratio=1.0)
        assert 0.0 <= result <= 1.0

    def test_negative_ratio_handled(self) -> None:
        """Negative ratio should be guarded (clamped to small positive)."""
        result = ratio_similarity(-1.0)
        assert 0.0 <= result <= 1.0


# ── normalize_weights ───────────────────────────────────────────────────────

class TestNormalizeWeights:
    def test_equal_weights_normalize(self) -> None:
        result = normalize_weights({"a": 1.0, "b": 1.0})
        assert result == {"a": 0.5, "b": 0.5}

    def test_single_weight_returns_one(self) -> None:
        result = normalize_weights({"a": 42.0})
        assert result == {"a": 1.0}

    def test_proportional_weights(self) -> None:
        result = normalize_weights({"a": 2.0, "b": 3.0})
        assert result["a"] == pytest.approx(0.4)
        assert result["b"] == pytest.approx(0.6)

    def test_empty_dict_returns_empty(self) -> None:
        assert normalize_weights({}) == {}

    def test_all_zeros_returns_empty(self) -> None:
        assert normalize_weights({"a": 0.0, "b": 0.0}) == {}

    def test_negative_weights_clipped_to_zero(self) -> None:
        result = normalize_weights({"a": -1.0, "b": 2.0})
        assert result == {"a": 0.0, "b": 1.0}

    def test_all_negative_returns_empty(self) -> None:
        assert normalize_weights({"a": -1.0, "b": -2.0}) == {}

    def test_sums_to_one(self) -> None:
        weights = {"a": 3.0, "b": 1.5, "c": 0.5}
        result = normalize_weights(weights)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_none_input_returns_empty(self) -> None:
        assert normalize_weights(None) == {}  # type: ignore[arg-type]

    def test_keys_are_stringified(self) -> None:
        result = normalize_weights({1: 1.0, 2: 1.0})  # type: ignore[dict-item]
        assert "1" in result
        assert "2" in result


# ── weighted_sum ─────────────────────────────────────────────────────────────

class TestWeightedSum:
    def test_equal_terms_equal_weights(self) -> None:
        result = weighted_sum({"a": 0.8, "b": 0.4}, {"a": 1.0, "b": 1.0})
        assert result == pytest.approx(0.6)

    def test_empty_weights_returns_zero(self) -> None:
        assert weighted_sum({"a": 1.0}, {}) == 0.0

    def test_all_zero_weights_returns_zero(self) -> None:
        assert weighted_sum({"a": 1.0}, {"a": 0.0}) == 0.0

    def test_terms_clamped_to_01(self) -> None:
        """Terms outside [0,1] should be clamped before weighting."""
        result = weighted_sum({"a": 2.0}, {"a": 1.0})
        assert result == 1.0  # term clamped to 1.0

    def test_result_is_bounded(self) -> None:
        for _ in range(100):
            result = weighted_sum(
                {"a": 0.7, "b": 0.3, "c": 0.9},
                {"a": 0.5, "b": 0.3, "c": 0.2},
            )
            assert 0.0 <= result <= 1.0

    def test_missing_term_treated_as_zero(self) -> None:
        # weights normalize to a:0.5 b:0.5; b term defaults to 0.0
        # result = 0.5*0.9 + 0.5*0.0 = 0.45
        result = weighted_sum({"a": 0.9}, {"a": 0.5, "b": 0.5})
        assert result == pytest.approx(0.45)

    def test_single_term_single_weight(self) -> None:
        result = weighted_sum({"a": 0.75}, {"a": 1.0})
        assert result == 0.75

    def test_negative_terms_clamped_to_zero(self) -> None:
        result = weighted_sum({"a": -0.5}, {"a": 1.0})
        assert result == 0.0

    def test_nan_term_returns_zero_for_that_term(self) -> None:
        """NaN terms are clamped to 0 by clamp01."""
        result = weighted_sum({"a": float("nan"), "b": 1.0}, {"a": 0.0, "b": 1.0})
        assert result == 1.0


# ── Roundtrip property tests ────────────────────────────────────────────────

class TestNormalizationRoundtrip:
    """Test that normalize_weights + weighted_sum compose correctly."""

    def test_weighted_sum_with_normalized_weights_equals_direct_average(self) -> None:
        weights = {"signal": 3.0, "noise": 2.0, "contrast": 5.0}
        terms = {"signal": 0.8, "noise": 0.2, "contrast": 0.6}
        result = weighted_sum(terms, weights)
        # Manual calculation
        norm = normalize_weights(weights)
        expected = sum(norm[k] * clamp01(terms[k]) for k in norm)
        assert result == pytest.approx(expected)

    def test_normalized_weights_preserve_relative_ratios(self) -> None:
        """Normalized weights should maintain the same proportions."""
        weights = {"a": 2.0, "b": 3.0, "c": 5.0}
        norm = normalize_weights(weights)
        # a:b ratio preserved
        assert norm["b"] / norm["a"] == pytest.approx(3.0 / 2.0)
