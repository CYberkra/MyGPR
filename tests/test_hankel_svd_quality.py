#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quality regression tests for Hankel-SVD synthetic denoising cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from PythonModule.hankel_svd import method_hankel_svd


@dataclass(frozen=True)
class SyntheticDenoiseCase:
    """Deterministic synthetic input / target pair for denoising tests."""

    clean: np.ndarray
    noisy: np.ndarray
    target_mask: np.ndarray | None = None


def compute_snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return reconstruction SNR in dB against a clean reference."""
    signal_norm = float(np.linalg.norm(reference))
    residual_norm = float(np.linalg.norm(reference - estimate))
    if residual_norm == 0.0:
        return float("inf")
    if signal_norm == 0.0:
        return float("-inf")
    return 20.0 * np.log10(signal_norm / residual_norm)


def compute_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return root-mean-square error against a clean reference."""
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


def target_preservation_ratio(
    clean: np.ndarray,
    estimate: np.ndarray,
    target_mask: np.ndarray,
) -> float:
    """Measure how much target-zone energy remains after denoising."""
    clean_energy = float(np.linalg.norm(clean[target_mask]))
    if clean_energy == 0.0:
        raise ValueError("target_mask must cover non-zero clean energy")
    return float(np.linalg.norm(estimate[target_mask]) / clean_energy)


def has_only_finite_values(matrix: np.ndarray) -> bool:
    """Return True when the output contains no NaN or inf."""
    return bool(np.isfinite(matrix).all())


def _expand_vertical_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand a target mask vertically to tolerate slight timing shifts."""
    expanded = mask.copy()
    for shift in range(1, radius + 1):
        expanded[shift:, :] |= mask[:-shift, :]
        expanded[:-shift, :] |= mask[shift:, :]
    return expanded


def _build_point_target_hyperbola_case() -> SyntheticDenoiseCase:
    """Create a noisy point-target hyperbola similar to Xue-style examples."""
    np.random.seed(7)
    samples = 96
    traces = 64
    sample_axis = np.arange(samples, dtype=np.float64)
    trace_axis = np.arange(traces, dtype=np.float64)
    trace_center = (traces - 1) / 2.0

    clean = np.zeros((samples, traces), dtype=np.float64)
    for col, lateral in enumerate(trace_axis):
        arrival = 22.0 + 0.055 * (lateral - trace_center) ** 2
        clean[:, col] += 1.40 * np.exp(-0.5 * ((sample_axis - arrival) / 1.5) ** 2)
        clean[:, col] -= 0.50 * np.exp(
            -0.5 * ((sample_axis - (arrival + 3.2)) / 2.2) ** 2
        )

    coherent_noise = 0.22 * np.sin(2.0 * np.pi * sample_axis / 13.0)[:, None]
    white_noise = 0.28 * np.random.normal(size=clean.shape)
    noisy = clean + coherent_noise + white_noise
    target_mask = _expand_vertical_mask(np.abs(clean) > 0.05, radius=3)
    return SyntheticDenoiseCase(clean=clean, noisy=noisy, target_mask=target_mask)


def _build_horizontal_layer_case() -> SyntheticDenoiseCase:
    """Create a noisy layered reflector field with lateral continuity."""
    np.random.seed(11)
    samples = 96
    traces = 64
    sample_axis = np.arange(samples, dtype=np.float64)
    trace_axis = np.arange(traces, dtype=np.float64)

    clean = np.zeros((samples, traces), dtype=np.float64)
    clean[40:43, :] += np.array([[0.60], [1.25], [0.65]], dtype=np.float64)
    clean[68:70, :] -= np.array([[0.35], [0.22]], dtype=np.float64)

    horizontal_banding = 0.20 * np.sin(2.0 * np.pi * sample_axis / 9.0)[:, None]
    lateral_trend = 0.08 * np.cos(2.0 * np.pi * trace_axis / 17.0)[None, :]
    white_noise = 0.20 * np.random.normal(size=clean.shape)
    noisy = clean + horizontal_banding + lateral_trend + white_noise
    target_mask = _expand_vertical_mask(np.abs(clean) > 0.05, radius=3)
    return SyntheticDenoiseCase(clean=clean, noisy=noisy, target_mask=target_mask)


def _build_correlated_noise_case(correlation_length: int, seed: int) -> SyntheticDenoiseCase:
    """Create a correlated-noise analog with controllable trace correlation length."""
    np.random.seed(seed)
    samples = 96
    traces = 64

    clean = np.zeros((samples, traces), dtype=np.float64)
    clean[24:27, 20:44] += np.array([[0.20], [0.80], [0.20]], dtype=np.float64)
    clean[57:60, 10:30] -= np.array([[0.12], [0.40], [0.12]], dtype=np.float64)

    white_noise = np.random.normal(size=(samples, traces + correlation_length - 1))
    kernel = np.ones(correlation_length, dtype=np.float64) / float(correlation_length)
    correlated_noise = np.array(
        [np.convolve(row, kernel, mode="valid") for row in white_noise],
        dtype=np.float64,
    )
    noisy = clean + 0.38 * correlated_noise
    return SyntheticDenoiseCase(clean=clean, noisy=noisy)


@pytest.fixture
def point_target_hyperbola_case() -> SyntheticDenoiseCase:
    """Deterministic point-target hyperbola with coherent and white noise."""
    return _build_point_target_hyperbola_case()


@pytest.fixture
def horizontal_layer_case() -> SyntheticDenoiseCase:
    """Deterministic horizontal-layer reflector field with structured noise."""
    return _build_horizontal_layer_case()


@pytest.fixture
def correlated_noise_len10_case() -> SyntheticDenoiseCase:
    """Deterministic correlated-noise analog with length-10 smoothing."""
    return _build_correlated_noise_case(correlation_length=10, seed=17)


@pytest.fixture
def correlated_noise_len20_case() -> SyntheticDenoiseCase:
    """Deterministic correlated-noise analog with length-20 smoothing."""
    return _build_correlated_noise_case(correlation_length=20, seed=19)


@pytest.fixture
def flat_matrix() -> np.ndarray:
    """Deterministic flat matrix that should remain unchanged."""
    return np.full((64, 24), 3.5, dtype=np.float64)


@pytest.fixture
def nan_contaminated_matrix() -> np.ndarray:
    """Deterministic matrix with a single NaN contamination."""
    matrix = np.tile(np.linspace(-1.0, 1.0, 24, dtype=np.float64), (48, 1))
    matrix[17, 9] = np.nan
    return matrix


@pytest.fixture
def very_short_trace() -> np.ndarray:
    """Single-sample trace matrix for degenerate boundary behavior."""
    return np.array([[1.0, -1.0, 0.5, 0.25]], dtype=np.float64)


def test_hankel_svd_preserves_flat_matrix(flat_matrix: np.ndarray) -> None:
    """Happy-path sanity check for finite output on trivial low-rank data."""
    denoised, metadata = method_hankel_svd(flat_matrix.copy(), window_length=16, rank=2)

    assert denoised.shape == flat_matrix.shape
    assert has_only_finite_values(denoised)
    assert np.allclose(denoised, flat_matrix, atol=1e-10)
    assert metadata["method"] == "hankel_svd"


@pytest.mark.parametrize(
    ("fixture_name", "window_length", "rank", "min_snr_gain_db"),
    [
        ("correlated_noise_len10_case", 24, 3, 1.5),
        ("correlated_noise_len20_case", 24, 3, 1.0),
    ],
)
def test_hankel_svd_improves_correlated_noise_analogs(
    request: pytest.FixtureRequest,
    fixture_name: str,
    window_length: int,
    rank: int,
    min_snr_gain_db: float,
) -> None:
    """Happy-path checks for correlated-noise analogs inspired by Xue 2019."""
    case = request.getfixturevalue(fixture_name)

    denoised, _ = method_hankel_svd(
        case.noisy.copy(),
        window_length=window_length,
        rank=rank,
    )
    snr_before = compute_snr_db(case.clean, case.noisy)
    snr_after = compute_snr_db(case.clean, denoised)
    rmse_before = compute_rmse(case.clean, case.noisy)
    rmse_after = compute_rmse(case.clean, denoised)

    assert has_only_finite_values(denoised)
    assert denoised.shape == case.clean.shape
    assert snr_after > snr_before + min_snr_gain_db
    assert rmse_after < rmse_before


@pytest.mark.parametrize(
    ("fixture_name", "window_length", "rank", "min_target_ratio"),
    [
        ("point_target_hyperbola_case", 32, 4, 0.35),
        ("horizontal_layer_case", 32, 4, 0.35),
    ],
)
def test_hankel_svd_should_denoise_without_erasing_targets(
    request: pytest.FixtureRequest,
    fixture_name: str,
    window_length: int,
    rank: int,
    min_target_ratio: float,
) -> None:
    """Red quality regression: denoising should not wipe out target energy."""
    case = request.getfixturevalue(fixture_name)

    denoised, _ = method_hankel_svd(
        case.noisy.copy(),
        window_length=window_length,
        rank=rank,
    )
    snr_before = compute_snr_db(case.clean, case.noisy)
    snr_after = compute_snr_db(case.clean, denoised)
    rmse_before = compute_rmse(case.clean, case.noisy)
    rmse_after = compute_rmse(case.clean, denoised)
    preserved_ratio = target_preservation_ratio(
        case.clean,
        denoised,
        case.target_mask,
    )

    assert has_only_finite_values(denoised)
    assert snr_after > snr_before + 1.0
    assert rmse_after < rmse_before
    assert preserved_ratio >= min_target_ratio


def test_hankel_svd_handles_nan_contamination_without_crashing(
    nan_contaminated_matrix: np.ndarray,
) -> None:
    """Red robustness regression: a single NaN should not crash the method."""
    try:
        denoised, _ = method_hankel_svd(
            nan_contaminated_matrix.copy(),
            window_length=16,
            rank=2,
        )
    except Exception as exc:  # pragma: no cover - intentional red path today
        pytest.fail(f"NaN-contaminated input should not raise, got: {exc!r}")

    assert denoised.shape == nan_contaminated_matrix.shape
    assert has_only_finite_values(denoised)


def test_hankel_svd_keeps_single_sample_trace_finite(
    very_short_trace: np.ndarray,
) -> None:
    """Red boundary regression: single-sample traces should stay finite."""
    denoised, _ = method_hankel_svd(very_short_trace.copy(), window_length=8, rank=2)

    assert denoised.shape == very_short_trace.shape
    assert has_only_finite_values(denoised)
    assert np.allclose(denoised, very_short_trace, atol=1e-12)
