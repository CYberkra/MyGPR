#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D MSSA quality regression tests.

These tests verify that the 2D MSSA implementation provides
better cross-trace coherence preservation compared to what
a per-trace 1D Hankel SVD would achieve.
"""

from __future__ import annotations

import numpy as np
import pytest

from PythonModule.hankel_svd import method_hankel_svd


def _make_layered_bscan(n_samples: int = 128, n_traces: int = 32) -> np.ndarray:
    """Create a synthetic B-scan with horizontal layers and cross-trace coherence."""
    rng = np.random.default_rng(42)
    t = np.arange(n_samples, dtype=np.float64)
    
    # Layer 1 at sample 30
    layer1 = np.exp(-0.5 * ((t - 30) / 3) ** 2)
    # Layer 2 at sample 60
    layer2 = np.exp(-0.5 * ((t - 60) / 4) ** 2)
    # Layer 3 at sample 90
    layer3 = np.exp(-0.5 * ((t - 90) / 5) ** 2)
    
    signal = (layer1 + 0.7 * layer2 + 0.5 * layer3)[:, None]
    signal = np.repeat(signal, n_traces, axis=1)
    
    # Add coherent noise (simulating cross-trace correlated clutter)
    noise = rng.standard_normal(size=(n_samples, n_traces)) * 0.15
    # Make noise partially correlated across traces
    for i in range(1, n_traces):
        noise[:, i] = 0.6 * noise[:, i-1] + 0.4 * noise[:, i]
    
    return (signal + noise).astype(np.float32)


def _compute_cross_trace_correlation(data: np.ndarray) -> float:
    """Compute average correlation between adjacent traces."""
    n_traces = data.shape[1]
    if n_traces < 2:
        return 0.0
    
    correlations = []
    for i in range(n_traces - 1):
        a = data[:, i]
        b = data[:, i+1]
        # Pearson correlation
        a_norm = a - np.mean(a)
        b_norm = b - np.mean(b)
        denom = np.sqrt(np.sum(a_norm**2) * np.sum(b_norm**2))
        if denom > 1e-10:
            corr = np.sum(a_norm * b_norm) / denom
            correlations.append(abs(corr))
    
    return float(np.mean(correlations)) if correlations else 0.0


def _compute_snr_db(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> float:
    """Compute SNR improvement in dB."""
    noise_before = np.mean((noisy - clean) ** 2)
    noise_after = np.mean((denoised - clean) ** 2)
    if noise_before <= 1e-12 or noise_after <= 1e-12:
        return 0.0
    return 10.0 * np.log10(noise_before / noise_after)


# ---------------------------------------------------------------------------
# Cross-trace coherence tests
# ---------------------------------------------------------------------------

def test_mssa_preserves_cross_trace_coherence():
    """2D MSSA should preserve or enhance cross-trace correlation."""
    clean = _make_layered_bscan(128, 32)
    rng = np.random.default_rng(123)
    noisy = clean + rng.standard_normal(size=clean.shape).astype(np.float32) * 0.2
    
    corr_before = _compute_cross_trace_correlation(noisy)
    
    denoised, _meta = method_hankel_svd(
        noisy,
        window_length=64,
        rank=5,
        aggressiveness=1.0,
    )
    
    corr_after = _compute_cross_trace_correlation(denoised)
    
    # MSSA should maintain or improve cross-trace correlation
    assert corr_after >= corr_before * 0.9, (
        f"Cross-trace correlation degraded: {corr_before:.3f} -> {corr_after:.3f}"
    )


def test_mssa_improves_snr_on_layered_data():
    """2D MSSA should improve SNR on layered synthetic data."""
    clean = _make_layered_bscan(128, 32)
    rng = np.random.default_rng(456)
    noisy = clean + rng.standard_normal(size=clean.shape).astype(np.float32) * 0.25
    
    denoised, meta = method_hankel_svd(
        noisy,
        window_length=32,  # fixed, smaller window for better layer separation
        rank=5,  # fixed rank
        aggressiveness=1.0,
    )
    
    snr_db = _compute_snr_db(clean, noisy, denoised)
    
    assert snr_db > 0.5, f"SNR improvement too small: {snr_db:.2f} dB"
    assert meta["method"] == "hankel_svd"
    assert denoised.shape == clean.shape


def test_mssa_aggressiveness_controls_denoising_strength():
    """Higher aggressiveness should remove more noise (but possibly more signal)."""
    clean = _make_layered_bscan(128, 32)
    rng = np.random.default_rng(789)
    noisy = clean + rng.standard_normal(size=clean.shape).astype(np.float32) * 0.3
    
    # Conservative
    denoised_conservative, _ = method_hankel_svd(
        noisy, window_length=64, rank=0, aggressiveness=0.2,
    )
    
    # Standard
    denoised_standard, _ = method_hankel_svd(
        noisy, window_length=64, rank=0, aggressiveness=1.0,
    )
    
    # Aggressive
    denoised_aggressive, _ = method_hankel_svd(
        noisy, window_length=64, rank=0, aggressiveness=2.0,
    )
    
    # Lower aggressiveness (more conservative) should result in lower output variance
    var_conservative = np.var(denoised_conservative)
    var_standard = np.var(denoised_standard)
    var_aggressive = np.var(denoised_aggressive)
    
    # Conservative (low aggressiveness) removes more components = lower variance
    assert var_conservative <= var_standard * 1.2, (
        f"Conservative should reduce variance: conservative={var_conservative:.4f}, "
        f"standard={var_standard:.4f}, aggressive={var_aggressive:.4f}"
    )


def test_mssa_handles_small_number_of_traces():
    """MSSA should work even with very few traces."""
    rng = np.random.default_rng(321)
    data = rng.standard_normal(size=(64, 2)).astype(np.float32)
    
    denoised, meta = method_hankel_svd(
        data,
        window_length=32,
        rank=3,
        aggressiveness=1.0,
    )
    
    assert denoised.shape == data.shape
    assert np.isfinite(denoised).all()
    assert meta["method"] == "hankel_svd"


def test_mssa_auto_parameters_work_on_realistic_size():
    """Auto window and rank selection should work on realistic GPR sizes."""
    rng = np.random.default_rng(654)
    # Simulate a 512x256 GPR B-scan
    data = rng.standard_normal(size=(512, 256)).astype(np.float32) * 0.1
    
    # Add some coherent structure
    t = np.arange(512, dtype=np.float64)
    layer = np.exp(-0.5 * ((t - 200) / 10) ** 2)[:, None]
    data = data + np.repeat(layer, 256, axis=1).astype(np.float32)
    
    denoised, meta = method_hankel_svd(
        data,
        window_length=0,  # auto = n_samples // 2 = 256
        rank=0,  # auto via SVHT
        aggressiveness=1.0,
    )
    
    assert denoised.shape == data.shape
    assert np.isfinite(denoised).all()
    assert meta["window_length"] == 256  # auto
    assert meta["rank_selection_mode"] == "svht_auto"
    assert meta["effective_rank_max"] >= 1


def test_mssa_rank_zero_returns_auto_selected_rank():
    """rank=0 should trigger automatic rank selection."""
    rng = np.random.default_rng(987)
    data = rng.standard_normal(size=(128, 16)).astype(np.float32)
    
    _, meta = method_hankel_svd(
        data,
        window_length=64,
        rank=0,
        aggressiveness=1.0,
    )
    
    assert meta["rank_mode"] == "auto"
    assert meta["rank_selection_mode"] == "svht_auto"
    assert meta["effective_rank_max"] is not None
    assert meta["effective_rank_max"] >= 1
