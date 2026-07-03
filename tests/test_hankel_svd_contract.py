#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hankel-SVD compatibility and cancellation contract tests.

These tests define the expected contract for ``method_hankel_svd``.
Some assertions target metadata keys that will be added in a future
rewrite; those tests are intentionally *red* (failing) against the
current implementation so they guard the new behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from PythonModule.hankel_svd import method_hankel_svd, ProcessingCancelled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_noisy_bscan(seed: int, rows: int = 64, cols: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.sin(np.linspace(0.0, 4.0 * np.pi, rows, dtype=np.float32))[:, None]
    return np.repeat(base, cols, axis=1) + 0.05 * rng.standard_normal(
        size=(rows, cols)
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Metadata contract (old keys – should be green now)
# ---------------------------------------------------------------------------

OLD_METADATA_KEYS = (
    "method",
    "window_length",
    "rank_requested",
    "rank_mode",
    "effective_rank_min",
    "effective_rank_max",
    "svd_backend",
    "fallback_columns",
)

NEW_METADATA_KEYS = (
    "window_selection_mode",
    "rank_selection_mode",
    "rho",
    "recovery_mode",
    "candidate_windows",
    "calibration_traces",
    "warnings",
)


def test_old_metadata_keys_are_present():
    raw = _make_noisy_bscan(0)
    result, meta = method_hankel_svd(raw, window_length=8, rank=3)

    assert result.shape == raw.shape
    assert result.dtype == np.float64  # current impl up-casts to float
    for key in OLD_METADATA_KEYS:
        assert key in meta, f"missing old metadata key: {key}"

    assert meta["method"] == "hankel_svd"
    assert meta["window_length"] == 8
    assert meta["rank_requested"] == 3
    assert meta["rank_mode"] == "fixed"
    assert meta["svd_backend"] in {"truncated", "full", "mixed", "none"}
    assert isinstance(meta["fallback_columns"], int)
    assert meta["fallback_columns"] >= 0


# ---------------------------------------------------------------------------
# Metadata contract (new keys – intentionally red until rewrite)
# ---------------------------------------------------------------------------

def test_new_metadata_keys_are_present():
    """RED until rewrite adds window_selection_mode, warnings, etc."""
    raw = _make_noisy_bscan(1)
    _result, meta = method_hankel_svd(raw, window_length=8, rank=3)

    for key in NEW_METADATA_KEYS:
        assert key in meta, f"missing new metadata key: {key}"


def test_warnings_is_a_list():
    """RED until rewrite populates a warnings list."""
    raw = _make_noisy_bscan(2)
    _result, meta = method_hankel_svd(raw, window_length=8, rank=3)

    assert "warnings" in meta
    assert isinstance(meta["warnings"], list)


# ---------------------------------------------------------------------------
# Shape / dtype / deterministic contract
# ---------------------------------------------------------------------------

def test_output_preserves_shape_and_upcasts_dtype():
    raw = _make_noisy_bscan(3)
    result, meta = method_hankel_svd(raw, window_length=8, rank=3)

    assert result.shape == raw.shape
    assert result.dtype == np.float64


def test_deterministic_output_for_fixed_seed():
    raw = _make_noisy_bscan(4)
    r1, m1 = method_hankel_svd(raw.copy(), window_length=8, rank=3)
    r2, m2 = method_hankel_svd(raw.copy(), window_length=8, rank=3)

    # Structural equality, not exact fp – the algorithm is deterministic
    assert r1.shape == r2.shape
    assert r1.dtype == r2.dtype
    assert np.allclose(r1, r2)
    assert m1 == m2


# ---------------------------------------------------------------------------
# Legacy kwarg acceptance
# ---------------------------------------------------------------------------

def test_legacy_batch_size_kwarg_is_accepted_silently():
    raw = _make_noisy_bscan(5)
    # batch_size used to be an experimental parameter; it must not crash.
    result, meta = method_hankel_svd(raw, window_length=8, rank=3, batch_size=16)
    baseline_result, baseline_meta = method_hankel_svd(raw, window_length=8, rank=3)

    assert result.shape == raw.shape
    assert result.dtype == baseline_result.dtype
    assert meta["method"] == "hankel_svd"
    assert np.allclose(result, baseline_result)
    assert meta == baseline_meta


# ---------------------------------------------------------------------------
# Cancellation propagation
# ---------------------------------------------------------------------------

def test_cancel_checker_raises_processing_cancelled():
    raw = _make_noisy_bscan(6, rows=64, cols=24)
    call_count = 0

    def _cancel_after_16():
        nonlocal call_count
        call_count += 1
        return call_count >= 2  # second polled call (col 8 or 16)

    with pytest.raises(ProcessingCancelled):
        method_hankel_svd(raw, window_length=8, rank=3, cancel_checker=_cancel_after_16)

    assert call_count == 2


def test_cancel_checker_none_does_not_crash():
    raw = _make_noisy_bscan(7)
    result, meta = method_hankel_svd(
        raw, window_length=8, rank=3, cancel_checker=None
    )
    assert result.shape == raw.shape
    assert meta["method"] == "hankel_svd"


def test_preview_mode_records_bounded_caps_and_limits_auto_search():
    raw = _make_noisy_bscan(11, rows=96, cols=32)
    _result, meta = method_hankel_svd(raw, window_length=None, rank=None, preview=True)

    assert meta["preview"] is True
    assert meta["caps"]["max_candidate_windows"] == 6
    assert meta["caps"]["max_calibration_traces"] == 8
    assert meta["caps"]["auto_rank_fallback_cap"] == 3
    assert len(meta["candidate_windows"]) <= 6
    assert len(meta["calibration_traces"]) <= 8
    assert meta["window_length"] <= meta["caps"]["preview_window_cap"]


def test_preview_fixed_window_emits_warning_when_cap_applies():
    raw = _make_noisy_bscan(12, rows=96, cols=8)
    _result, meta = method_hankel_svd(raw, window_length=80, rank=3, preview=True)

    assert meta["window_length"] == 24
    assert any("preview capped requested window_length" in warning for warning in meta["warnings"])


def test_infeasible_fixed_rank_and_window_emit_metadata_warnings():
    raw = _make_noisy_bscan(13, rows=32, cols=8)
    _result, meta = method_hankel_svd(raw, window_length=128, rank=128)

    assert meta["window_length"] == 31
    assert meta["effective_rank_max"] == 1
    assert any("requested window_length 128 was adjusted" in warning for warning in meta["warnings"])
    assert any("requested rank 128 was adjusted" in warning for warning in meta["warnings"])


def test_auto_rank_fallback_is_deterministic_when_heuristic_path_produces_no_rank(monkeypatch: pytest.MonkeyPatch):
    raw = _make_noisy_bscan(15, rows=64, cols=16)

    def _empty_trajectory(_data, _window_size):
        return np.zeros((0, 0), dtype=np.float64)

    monkeypatch.setattr("PythonModule.hankel_svd._build_vmssa_trajectory", _empty_trajectory)
    _result, meta = method_hankel_svd(raw, window_length=None, rank=None, preview=True)

    assert meta["preview"] is True
    assert any("trajectory matrix is empty" in warning for warning in meta["warnings"])


def test_auto_calibration_loop_honors_cancellation_checker():
    raw = _make_noisy_bscan(14, rows=96, cols=32)
    call_count = 0

    def _cancel_in_calibration_loop():
        nonlocal call_count
        call_count += 1
        return call_count >= 3

    with pytest.raises(ProcessingCancelled):
        method_hankel_svd(raw, window_length=None, rank=None, preview=True, cancel_checker=_cancel_in_calibration_loop)

    assert call_count == 3


# ---------------------------------------------------------------------------
# Infeasible rank / window fallback
# ---------------------------------------------------------------------------

def test_infeasible_large_window_is_capped_and_does_not_crash():
    """Window > ny is capped to ny-1 by _resolve_window_length; must not crash."""
    raw = _make_noisy_bscan(8, rows=10, cols=4)
    result, meta = method_hankel_svd(raw, window_length=100, rank=3)

    assert result.shape == raw.shape
    assert np.isfinite(result).all()
    # Current impl caps window, so early-return (svd_backend=="none") is unreachable.
    assert meta["svd_backend"] in {"truncated", "full", "mixed"}


def test_infeasible_large_rank_is_capped():
    raw = _make_noisy_bscan(9, rows=64, cols=8)
    result, meta = method_hankel_svd(raw, window_length=8, rank=500)

    assert result.shape == raw.shape
    assert meta["effective_rank_max"] <= 7  # cannot exceed min(m, window_length)-1


# ---------------------------------------------------------------------------
# Edge-case robustness (must not crash)
# ---------------------------------------------------------------------------

def test_flat_input_does_not_crash():
    raw = np.ones((32, 8), dtype=np.float32)
    result, meta = method_hankel_svd(raw, window_length=8, rank=3)

    assert result.shape == raw.shape
    assert np.isfinite(result).all()
    assert meta["method"] == "hankel_svd"


def test_nan_input_does_not_crash():
    """RED – current implementation crashes on NaN because scipy.svd uses check_finite=True."""
    raw = _make_noisy_bscan(10)
    raw[5, 3] = np.nan
    result, meta = method_hankel_svd(raw, window_length=8, rank=3)

    assert result.shape == raw.shape
    assert meta["method"] == "hankel_svd"


def test_short_input_does_not_crash():
    """Very small inputs are handled without crash (window is capped to ny-1)."""
    raw = np.arange(4, dtype=np.float32).reshape(2, 2)
    result, meta = method_hankel_svd(raw, window_length=8, rank=3)

    assert result.shape == raw.shape
    assert meta["method"] == "hankel_svd"
    # Current impl caps window so m >= 2; early-return path never triggers.
    assert meta["svd_backend"] in {"truncated", "full", "mixed"}
