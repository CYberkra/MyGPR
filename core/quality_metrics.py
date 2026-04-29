#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-tune and benchmark quality metrics for GPR preprocessing steps."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.ndimage import uniform_filter1d


EPS = 1.0e-12


def relative_reduction(before: float, after: float) -> float:
    """Normalized improvement for metrics where lower is better."""
    base = max(float(before), EPS)
    return float(np.clip((base - float(after)) / base, -1.0, 1.0))


def ratio_fidelity(ratio: float, target: float = 1.0, tol: float = 0.15) -> float:
    """Score how close a ratio is to the desired target value."""
    r = max(float(ratio), EPS)
    t = max(float(target), EPS)
    k = max(np.log1p(float(tol)), EPS)
    return float(np.exp(-abs(np.log(r / t)) / k))


def _as_clean_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是二维数组，当前 shape={arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _normalized_abs(data: np.ndarray) -> np.ndarray:
    arr = np.abs(_as_clean_2d(data))
    trace_scale = np.max(arr, axis=0, keepdims=True)
    trace_scale = np.where(trace_scale > EPS, trace_scale, 1.0)
    return arr / trace_scale


def _gradient_energy(data: np.ndarray) -> np.ndarray:
    arr = _as_clean_2d(data)
    grad_t = np.diff(arr, axis=0, prepend=arr[[0], :])
    grad_x = np.diff(arr, axis=1, prepend=arr[:, [0]])
    return np.sqrt(grad_t**2 + grad_x**2)


def build_saliency_map(data: np.ndarray) -> np.ndarray:
    """Build a lightweight saliency map from amplitude and gradient energy."""
    arr = _as_clean_2d(data)
    grad = _gradient_energy(arr)
    abs_norm = _normalized_abs(arr)
    saliency = 0.65 * grad + 0.35 * abs_norm
    return uniform_filter1d(saliency, size=5, axis=0, mode="nearest")


def auto_roi_bounds(data: np.ndarray, padding_ratio: float = 0.12) -> dict[str, int]:
    """Estimate a rectangular ROI from the saliency distribution."""
    arr = _as_clean_2d(data)
    n_samples, n_traces = arr.shape
    saliency = build_saliency_map(arr)

    row_score = uniform_filter1d(
        np.mean(saliency, axis=1), size=max(5, n_samples // 24), mode="nearest"
    )
    col_score = uniform_filter1d(
        np.mean(saliency, axis=0), size=max(3, n_traces // 24), mode="nearest"
    )

    row_thresh = float(np.percentile(row_score, 78.0)) if row_score.size else 0.0
    col_thresh = float(np.percentile(col_score, 72.0)) if col_score.size else 0.0
    row_idx = np.flatnonzero(row_score >= row_thresh)
    col_idx = np.flatnonzero(col_score >= col_thresh)

    if row_idx.size == 0:
        center = int(np.argmax(row_score)) if row_score.size else n_samples // 2
        half = max(8, n_samples // 10)
        row_start, row_end = max(0, center - half), min(n_samples, center + half)
    else:
        pad = max(4, int(round((row_idx[-1] - row_idx[0] + 1) * padding_ratio)))
        row_start = max(0, int(row_idx[0]) - pad)
        row_end = min(n_samples, int(row_idx[-1]) + pad + 1)

    if col_idx.size == 0:
        col_start, col_end = 0, n_traces
    else:
        pad = max(2, int(round((col_idx[-1] - col_idx[0] + 1) * padding_ratio)))
        col_start = max(0, int(col_idx[0]) - pad)
        col_end = min(n_traces, int(col_idx[-1]) + pad + 1)

    if row_end <= row_start:
        row_start, row_end = 0, n_samples
    if col_end <= col_start:
        col_start, col_end = 0, n_traces

    return {
        "time_start_idx": int(row_start),
        "time_end_idx": int(row_end),
        "dist_start_idx": int(col_start),
        "dist_end_idx": int(col_end),
    }


def extract_roi_and_context(
    data: np.ndarray,
    roi_bounds: dict[str, int] | None = None,
    padding_ratio: float = 0.18,
) -> dict[str, object]:
    """Extract ROI and its expanded context window."""
    arr = _as_clean_2d(data)
    n_samples, n_traces = arr.shape
    bounds = dict(roi_bounds or auto_roi_bounds(arr))
    time_start = bounds.get("time_start_idx", 0)
    time_end = bounds.get("time_end_idx", n_samples)
    dist_start = bounds.get("dist_start_idx", 0)
    dist_end = bounds.get("dist_end_idx", n_traces)
    t0 = max(0, min(int(0 if time_start is None else time_start), n_samples - 1))
    t1 = max(t0 + 1, min(int(n_samples if time_end is None else time_end), n_samples))
    d0 = max(0, min(int(0 if dist_start is None else dist_start), n_traces - 1))
    d1 = max(d0 + 1, min(int(n_traces if dist_end is None else dist_end), n_traces))

    pad_t = max(2, int(round((t1 - t0) * padding_ratio)))
    pad_d = max(2, int(round((d1 - d0) * padding_ratio)))
    c_t0 = max(0, t0 - pad_t)
    c_t1 = min(n_samples, t1 + pad_t)
    c_d0 = max(0, d0 - pad_d)
    c_d1 = min(n_traces, d1 + pad_d)

    return {
        "bounds": {
            "time_start_idx": t0,
            "time_end_idx": t1,
            "dist_start_idx": d0,
            "dist_end_idx": d1,
        },
        "roi_data": arr[t0:t1, d0:d1],
        "context_bounds": {
            "time_start_idx": c_t0,
            "time_end_idx": c_t1,
            "dist_start_idx": c_d0,
            "dist_end_idx": c_d1,
        },
        "context_data": arr[c_t0:c_t1, c_d0:c_d1],
    }


def estimate_lateral_correlation_length(
    data: np.ndarray, max_lag: int | None = None
) -> int:
    """Estimate correlation length along traces using lagged trace correlation."""
    arr = _as_clean_2d(data)
    n_traces = arr.shape[1]
    if n_traces <= 2:
        return 1
    max_lag_value = max(6, n_traces // 6) if max_lag is None else int(max_lag)
    resolved_max_lag = min(n_traces - 1, max_lag_value)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    std = np.std(centered, axis=0, keepdims=True)
    std = np.where(std > EPS, std, 1.0)
    normalized = centered / std

    corrs = []
    for lag in range(1, resolved_max_lag + 1):
        left = normalized[:, :-lag]
        right = normalized[:, lag:]
        corr = float(np.mean(left * right))
        corrs.append(corr)
        if corr < 0.55:
            return lag
    if not corrs:
        return 1
    return int(np.argmax(corrs) + 1)


def estimate_depth_attenuation_curve(data: np.ndarray) -> np.ndarray:
    """Estimate smoothed per-depth attenuation envelope."""
    arr = _as_clean_2d(data)
    rms = np.sqrt(np.mean(arr**2, axis=1))
    window = max(5, min(len(rms) // 20, 41))
    if window % 2 == 0:
        window += 1
    return uniform_filter1d(rms, size=window, mode="nearest")


def estimate_singular_elbow_rank(data: np.ndarray, max_rank: int = 12) -> int:
    """Estimate low-rank elbow from singular values."""
    arr = _as_clean_2d(data)
    ds_t = max(1, arr.shape[0] // 256)
    ds_x = max(1, arr.shape[1] // 128)
    proxy = arr[::ds_t, ::ds_x]
    if min(proxy.shape) <= 2:
        return 1
    singular = np.linalg.svd(proxy, full_matrices=False, compute_uv=False)
    singular = singular[: max_rank + 2]
    if singular.size <= 2:
        return 1
    second_diff = np.diff(np.log(np.maximum(singular, EPS)), n=2)
    if second_diff.size == 0:
        return 1
    rank = int(np.argmax(np.abs(second_diff)) + 2)
    return max(1, min(rank, max_rank))


def weighted_score_parts(
    roi_score: float,
    full_score: float,
    guard_score: float,
    use_roi: bool,
) -> float:
    """Combine ROI/full/guard score parts into a final score."""
    if not use_roi:
        return float(full_score)
    return float(0.60 * roi_score + 0.25 * full_score + 0.15 * guard_score)


def _band_mask(n_samples: int, band: tuple[float, float]) -> np.ndarray:
    freqs = np.fft.rfftfreq(max(n_samples, 2), d=1.0)
    nyquist = max(float(freqs[-1]), EPS)
    low = max(0.0, float(band[0])) * nyquist
    high = min(1.0, float(band[1])) * nyquist
    if high <= low:
        high = min(nyquist, low + nyquist * 0.1)
    return (freqs >= low) & (freqs <= high)


def detect_first_break_indices(
    data: np.ndarray,
    method: str = "threshold",
    threshold: float = 0.05,
    search_ratio: float = 0.35,
) -> np.ndarray:
    """Detect per-trace first-break indices using lightweight rules."""
    arr = _as_clean_2d(data)
    n_samples, n_traces = arr.shape
    search_end = max(4, min(n_samples, int(np.ceil(n_samples * float(search_ratio)))))
    abs_norm = _normalized_abs(arr)
    gradient_norm = _normalized_abs(np.diff(arr, axis=0, prepend=arr[[0], :]))
    smooth_env = uniform_filter1d(abs_norm, size=5, axis=0, mode="nearest")

    threshold = float(np.clip(threshold, 1.0e-4, 0.95))
    indices = np.zeros(n_traces, dtype=np.int32)

    for trace_idx in range(n_traces):
        if method == "peak":
            local = smooth_env[:search_end, trace_idx]
            peak_idx = int(np.argmax(local))
            peak_val = float(local[peak_idx])
            gate = max(threshold * peak_val, peak_val * 0.25)
            candidates = np.flatnonzero(local >= gate)
        elif method == "first_break":
            local = gradient_norm[:search_end, trace_idx]
            gate = max(float(np.percentile(local, 92.0)), threshold)
            candidates = np.flatnonzero(local >= gate)
        else:
            local = abs_norm[:search_end, trace_idx]
            gate = max(threshold, float(np.percentile(local, 75.0)) * 0.25)
            candidates = np.flatnonzero(local >= gate)

        indices[trace_idx] = int(candidates[0]) if candidates.size else 0

    return indices


def pre_zero_energy_ratio(data: np.ndarray, zero_idx: int) -> float:
    """Energy before zero-time divided by total energy."""
    arr = _as_clean_2d(data)
    idx = max(0, min(int(zero_idx), arr.shape[0]))
    total = float(np.sum(arr**2))
    if total <= EPS or idx <= 0:
        return 0.0
    return float(np.sum(arr[:idx, :] ** 2) / total)


def first_break_std(
    data: np.ndarray, method: str = "threshold", threshold: float = 0.05
) -> float:
    """Std of first-break indices across traces."""
    indices = detect_first_break_indices(data, method=method, threshold=threshold)
    return float(np.std(indices))


def first_break_sharpness(data: np.ndarray, zero_idx: int) -> float:
    """Local mean gradient magnitude around the zero-time zone."""
    arr = _as_clean_2d(data)
    idx = max(1, min(int(zero_idx), arr.shape[0] - 2))
    grad = np.abs(np.diff(arr, axis=0, prepend=arr[[0], :]))
    start = max(0, idx - 2)
    stop = min(arr.shape[0], idx + 3)
    return float(np.mean(grad[start:stop, :]))


def baseline_bias(
    data: np.ndarray, pre_zero_only: bool = False, zero_idx: int | None = None
) -> float:
    """Absolute baseline bias measured by trace means."""
    arr = _as_clean_2d(data)
    if pre_zero_only and zero_idx is not None and zero_idx > 0:
        arr = arr[: max(1, int(zero_idx)), :]
    trace_means = np.mean(arr, axis=0)
    return float(np.mean(np.abs(trace_means)))


def low_freq_energy_ratio(
    data: np.ndarray, fs: float | None = None, cutoff_ratio: float = 0.08
) -> float:
    """Low-frequency energy ratio along the time axis."""
    arr = _as_clean_2d(data)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    if spec.size == 0:
        return 0.0
    freqs = np.fft.rfftfreq(arr.shape[0], d=1.0 / fs if fs and fs > 0 else 1.0)
    cutoff = float(cutoff_ratio) * float(freqs[-1] if freqs.size > 0 else 1.0)
    mask = freqs <= cutoff
    total = float(np.sum(spec))
    if total <= EPS:
        return 0.0
    return float(np.sum(spec[mask, :]) / total)


def target_band_energy_ratio(
    before: np.ndarray,
    after: np.ndarray,
    band: tuple[float, float] = (0.08, 0.45),
) -> float:
    """Energy preservation ratio in a target mid-frequency band."""
    arr_before = _as_clean_2d(before)
    arr_after = _as_clean_2d(after)
    mask = _band_mask(arr_before.shape[0], band)
    spec_before = np.abs(np.fft.rfft(arr_before, axis=0)) ** 2
    spec_after = np.abs(np.fft.rfft(arr_after, axis=0)) ** 2
    band_before = float(np.sum(spec_before[mask, :]))
    band_after = float(np.sum(spec_after[mask, :]))
    return float(band_after / max(band_before, EPS))


def horizontal_coherence(data: np.ndarray) -> float:
    """Energy ratio of horizontally coherent background component."""
    arr = _as_clean_2d(data)
    background = np.mean(arr, axis=1, keepdims=True)
    total = float(np.sum(arr**2))
    if total <= EPS:
        return 0.0
    return float(np.sum(background**2) / total)


def local_saliency_preservation(before: np.ndarray, after: np.ndarray) -> float:
    """Preservation ratio of locally salient structures."""
    saliency_before = _gradient_energy(before)
    saliency_after = _gradient_energy(after)
    q = float(np.percentile(saliency_before, 85.0)) if saliency_before.size else 0.0
    mask = saliency_before >= q
    if not np.any(mask):
        return 1.0
    preserved = float(np.mean(saliency_after[mask]))
    base = float(np.mean(saliency_before[mask]))
    return float(np.clip(preserved / max(base, EPS), 0.0, 1.5))


def depth_rms_cv(data: np.ndarray) -> float:
    """Coefficient of variation of per-depth RMS values."""
    arr = _as_clean_2d(data)
    depth_rms = np.sqrt(np.mean(arr**2, axis=1))
    mean_rms = float(np.mean(depth_rms))
    if mean_rms <= EPS:
        return 0.0
    return float(np.std(depth_rms) / mean_rms)


def deep_zone_contrast(data: np.ndarray, deep_ratio: float = 0.4) -> float:
    """Contrast estimate for the deep zone."""
    arr = _as_clean_2d(data)
    start = int(np.floor(arr.shape[0] * (1.0 - float(deep_ratio))))
    deep = arr[max(0, start) :, :]
    if deep.size == 0:
        return 0.0
    local = deep - np.median(deep, axis=1, keepdims=True)
    return float(np.std(local))


def clipping_ratio(data: np.ndarray, high_quantile: float = 0.999) -> float:
    """Ratio of near-saturated pixels."""
    arr = np.abs(_as_clean_2d(data))
    max_abs = float(np.max(arr)) if arr.size else 0.0
    if max_abs <= EPS:
        return 0.0
    q = float(np.quantile(arr, min(max(high_quantile, 0.9), 0.9999)))
    gate = max(q, 0.98 * max_abs)
    return float(np.mean(arr >= gate))


def hot_pixel_ratio(data: np.ndarray, z: float = 6.0) -> float:
    """Ratio of extreme bright outliers."""
    arr = _as_clean_2d(data)
    centered = arr - np.mean(arr)
    std = float(np.std(centered))
    if std <= EPS:
        return 0.0
    return float(np.mean(np.abs(centered / std) >= float(z)))


def kurtosis_or_spikiness(data: np.ndarray) -> float:
    """Excess kurtosis-like spikiness measure."""
    arr = _as_clean_2d(data)
    centered = arr - np.mean(arr)
    std = float(np.std(centered))
    if std <= EPS:
        return 0.0
    z = centered / std
    return float(max(0.0, np.mean(z**4) - 3.0))


def edge_preservation(before: np.ndarray, after: np.ndarray) -> float:
    """Ratio of preserved edge energy after processing."""
    before_edge = _gradient_energy(before)
    after_edge = _gradient_energy(after)
    before_energy = float(np.mean(before_edge))
    if before_energy <= EPS:
        return 1.0
    return float(np.clip(np.mean(after_edge) / before_energy, 0.0, 1.5))


def median_first_break(indices: Iterable[int]) -> int:
    """Robust aggregate for detected first-break indices."""
    arr = np.asarray(list(indices), dtype=np.float64)
    if arr.size == 0:
        return 0
    return int(np.median(arr))


def compute_benchmark_metrics(
    before: np.ndarray,
    after: np.ndarray,
    zero_idx: int | None = None,
) -> dict[str, float]:
    """Compute a stable benchmark metric bundle for before/after comparisons."""
    before_arr = _as_clean_2d(before)
    after_arr = _as_clean_2d(after)

    baseline_before = baseline_bias(before_arr)
    baseline_after = baseline_bias(after_arr)
    low_freq_before = low_freq_energy_ratio(before_arr)
    low_freq_after = low_freq_energy_ratio(after_arr)
    coherence_before = horizontal_coherence(before_arr)
    coherence_after = horizontal_coherence(after_arr)
    deep_before = deep_zone_contrast(before_arr)
    deep_after = deep_zone_contrast(after_arr)

    metrics = {
        "baseline_bias_before": baseline_before,
        "baseline_bias_after": baseline_after,
        "baseline_bias_reduction": relative_reduction(baseline_before, baseline_after),
        "low_freq_energy_ratio_before": low_freq_before,
        "low_freq_energy_ratio_after": low_freq_after,
        "low_freq_energy_reduction": relative_reduction(
            low_freq_before, low_freq_after
        ),
        "horizontal_coherence_before": coherence_before,
        "horizontal_coherence_after": coherence_after,
        "horizontal_coherence_reduction": relative_reduction(
            coherence_before, coherence_after
        ),
        "target_band_energy_ratio": target_band_energy_ratio(before_arr, after_arr),
        "local_saliency_preservation": local_saliency_preservation(
            before_arr, after_arr
        ),
        "edge_preservation": edge_preservation(before_arr, after_arr),
        "deep_zone_contrast_before": deep_before,
        "deep_zone_contrast_after": deep_after,
        "deep_zone_contrast_gain": float(deep_after / max(deep_before, EPS)),
        "clipping_ratio_after": clipping_ratio(after_arr),
        "hot_pixel_ratio_after": hot_pixel_ratio(after_arr),
        "kurtosis_or_spikiness_after": kurtosis_or_spikiness(after_arr),
    }

    if zero_idx is not None:
        idx = int(zero_idx)
        metrics.update(
            {
                "pre_zero_energy_ratio_before": pre_zero_energy_ratio(before_arr, idx),
                "pre_zero_energy_ratio_after": pre_zero_energy_ratio(after_arr, idx),
                "first_break_sharpness_before": first_break_sharpness(before_arr, idx),
                "first_break_sharpness_after": first_break_sharpness(after_arr, idx),
                "baseline_bias_pre_zero_after": baseline_bias(
                    after_arr, pre_zero_only=True, zero_idx=idx
                ),
            }
        )

    return {key: float(value) for key, value in metrics.items()}


def _metadata_array(metadata: dict[str, object], key: str) -> np.ndarray:
    """Fetch a 1D numeric metadata array with validation."""
    if key not in metadata:
        raise KeyError(f"缺少 trace_metadata['{key}']")
    values = np.asarray(metadata[key], dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"trace_metadata['{key}'] 不能为空")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_row_range(
    row_range: tuple[int, int] | list[int] | None,
    total_rows: int,
) -> tuple[int, int]:
    """Clamp an optional row window to valid array bounds."""
    if row_range is None:
        return 0, total_rows
    start = max(0, min(int(row_range[0]), total_rows - 1))
    stop = max(start + 1, min(int(row_range[1]), total_rows))
    return start, stop


def detect_ridge_indices(
    data: np.ndarray,
    row_range: tuple[int, int] | list[int] | None = None,
) -> np.ndarray:
    """Return the strongest absolute-amplitude row index per trace."""
    arr = _as_clean_2d(data)
    start, stop = _resolve_row_range(row_range, arr.shape[0])
    return start + np.argmax(np.abs(arr[start:stop, :]), axis=0)


def ridge_error_metrics(
    data: np.ndarray,
    ground_truth_ridge_idx: np.ndarray,
    row_range: tuple[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    """Compare detected reflector ridge indices against ground truth."""
    detected = detect_ridge_indices(data, row_range=row_range).astype(np.float64)
    truth = np.asarray(ground_truth_ridge_idx, dtype=np.float64).reshape(-1)
    n = min(detected.size, truth.size)
    if n == 0:
        raise ValueError("reflector ridge ground truth不能为空")
    residual = detected[:n] - truth[:n]
    return {
        "detected_ridge_idx": detected[:n].astype(np.int32),
        "ridge_residual_samples": residual,
        "raw_ridge_mae_samples": float(np.mean(np.abs(residual))),
        "raw_ridge_rmse_samples": float(np.sqrt(np.mean(residual**2))),
        "raw_ridge_max_abs_samples": float(np.max(np.abs(residual))),
    }


def trace_spacing_std(trace_metadata: dict[str, object], key: str = "trace_distance_m") -> float:
    """Std of inter-trace spacing from cumulative distance metadata."""
    distance = _metadata_array(trace_metadata, key)
    if distance.size <= 1:
        return 0.0
    return float(np.std(np.diff(distance)))


def path_rmse(
    trace_metadata: dict[str, object],
    ground_truth_trace_metadata: dict[str, object],
) -> float:
    """RMSE between observed and ground-truth antenna path."""
    obs_x = _metadata_array(trace_metadata, "local_x_m")
    obs_y = _metadata_array(trace_metadata, "local_y_m")
    ref_x = _metadata_array(ground_truth_trace_metadata, "local_x_m")
    ref_y = _metadata_array(ground_truth_trace_metadata, "local_y_m")
    n = min(obs_x.size, obs_y.size, ref_x.size, ref_y.size)
    if n == 0:
        raise ValueError("路径元数据不能为空")
    dx = obs_x[:n] - ref_x[:n]
    dy = obs_y[:n] - ref_y[:n]
    return float(np.sqrt(np.mean(dx**2 + dy**2)))


def footprint_rmse(
    trace_metadata: dict[str, object],
    ground_truth_trace_metadata: dict[str, object],
) -> float:
    """RMSE between observed and ground-truth footprint coordinates."""
    obs_x = _metadata_array(trace_metadata, "footprint_x_m")
    obs_y = _metadata_array(trace_metadata, "footprint_y_m")
    ref_x = _metadata_array(ground_truth_trace_metadata, "footprint_x_m")
    ref_y = _metadata_array(ground_truth_trace_metadata, "footprint_y_m")
    n = min(obs_x.size, obs_y.size, ref_x.size, ref_y.size)
    if n == 0:
        raise ValueError("footprint 元数据不能为空")
    dx = obs_x[:n] - ref_x[:n]
    dy = obs_y[:n] - ref_y[:n]
    return float(np.sqrt(np.mean(dx**2 + dy**2)))


def periodic_banding_ratio(
    data: np.ndarray,
    trace_band: tuple[float, float] = (0.05, 0.18),
    row_range: tuple[int, int] | list[int] | None = None,
) -> float:
    """Ratio of lateral spectral energy inside a periodic striping band."""
    arr = _as_clean_2d(data)
    start, stop = _resolve_row_range(row_range, arr.shape[0])
    window = arr[start:stop, :]
    if window.shape[1] <= 2:
        return 0.0
    centered = window - np.mean(window, axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    if spec.shape[1] <= 1:
        return 0.0
    freqs = np.fft.rfftfreq(window.shape[1], d=1.0)
    low = max(0.0, float(trace_band[0]))
    high = min(float(freqs[-1]), float(trace_band[1]))
    if high <= low:
        high = min(float(freqs[-1]), low + 0.04)
    band_mask = (freqs >= low) & (freqs <= high)
    band_mask[0] = False
    total = float(np.sum(spec[:, 1:]))
    if total <= EPS:
        return 0.0
    return float(np.sum(spec[:, band_mask]) / total)


def target_preservation_ratio(
    data: np.ndarray,
    reference_data: np.ndarray,
    row_range: tuple[int, int] | list[int],
) -> float:
    """Energy ratio inside the target zone relative to a clean reference sample."""
    arr = _as_clean_2d(data)
    ref = _as_clean_2d(reference_data)
    n_rows = min(arr.shape[0], ref.shape[0])
    n_cols = min(arr.shape[1], ref.shape[1])
    start, stop = _resolve_row_range(row_range, n_rows)
    window = arr[start:stop, :n_cols]
    ref_window = ref[start:stop, :n_cols]
    denom = float(np.sum(ref_window**2))
    if denom <= EPS:
        return 0.0
    return float(np.sum(window**2) / denom)


def compute_motion_quality_metrics(
    data: np.ndarray,
    trace_metadata: dict[str, object],
    ground_truth_trace_metadata: dict[str, object],
    *,
    ground_truth_data: np.ndarray | None = None,
    ridge_row_range: tuple[int, int] | list[int] | None = None,
    target_row_range: tuple[int, int] | list[int] | None = None,
    banding_trace_band: tuple[float, float] = (0.05, 0.18),
    banding_row_range: tuple[int, int] | list[int] | None = None,
) -> dict[str, float]:
    """Compute scalar benchmark metrics for raw or corrected motion-compensation data."""
    ground_truth_ridge_idx = np.asarray(
        ground_truth_trace_metadata["reflector_ridge_idx"],
        dtype=np.float64,
    )
    ridge = ridge_error_metrics(
        data,
        ground_truth_ridge_idx,
        row_range=ridge_row_range,
    )
    metrics = {
        "raw_ridge_mae_samples": float(ridge["raw_ridge_mae_samples"]),
        "raw_ridge_rmse_samples": float(ridge["raw_ridge_rmse_samples"]),
        "raw_ridge_max_abs_samples": float(ridge["raw_ridge_max_abs_samples"]),
        "trace_spacing_std_m": float(trace_spacing_std(trace_metadata)),
        "path_rmse_m": float(path_rmse(trace_metadata, ground_truth_trace_metadata)),
        "footprint_rmse_m": float(
            footprint_rmse(trace_metadata, ground_truth_trace_metadata)
        ),
        "periodic_banding_ratio": float(
            periodic_banding_ratio(
                data,
                trace_band=banding_trace_band,
                row_range=banding_row_range,
            )
        ),
    }
    if ground_truth_data is not None and target_row_range is not None:
        metrics["target_preservation_ratio"] = float(
            target_preservation_ratio(data, ground_truth_data, row_range=target_row_range)
        )
    return metrics
