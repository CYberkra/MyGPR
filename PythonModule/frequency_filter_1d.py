#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero-phase 1D frequency filtering along the GPR time axis."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.runtime_warnings import build_runtime_warning
from core.scalar_utils import to_float


def method_frequency_filter_1d(
    data: np.ndarray,
    filter_type: str = "bandpass",
    low_freq_mhz: float = 10.0,
    high_freq_mhz: float = 800.0,
    notch_freq_mhz: float = 50.0,
    notch_width_mhz: float = 5.0,
    notch_depth: float = 1.0,
    taper_ratio: float = 0.08,
    sample_rate_hz: float | None = None,
    sample_rate_mhz: float | None = None,
    time_step_s: float | None = None,
    sample_interval_ns: float | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a zero-phase FFT-domain filter to each A-scan.

    The method keeps the B-scan shape unchanged and only filters along axis 0
    (time/depth samples). Frequencies are specified in MHz to match GPR use.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    resolved_sample_rate_hz = _resolve_sample_rate_hz(
        sample_rate_hz=sample_rate_hz,
        sample_rate_mhz=sample_rate_mhz,
        time_step_s=time_step_s,
        sample_interval_ns=sample_interval_ns,
    )
    if resolved_sample_rate_hz <= 0.0:
        return arr.astype(np.float32, copy=True), {
            "method": "frequency_filter_1d",
            "filter_type": str(filter_type or "bandpass").strip().lower(),
            "skipped": True,
            "runtime_warnings": [
                build_runtime_warning(
                    "frequency_sampling_missing",
                    "缺少采样率或时间窗信息，频域滤波已跳过以避免错误截止频率。",
                    method_id="frequency_filter_1d",
                )
            ],
        }

    filter_key = str(filter_type or "bandpass").strip().lower()
    if filter_key not in {"bandpass", "lowpass", "highpass", "notch"}:
        raise ValueError(f"Unsupported filter_type: {filter_type}")

    n_samples = int(arr.shape[0])
    freqs_hz = np.fft.rfftfreq(n_samples, d=1.0 / resolved_sample_rate_hz)
    nyquist_mhz = float(resolved_sample_rate_hz / 2.0e6)
    warnings: list[dict[str, Any]] = []

    mask, effective = _build_filter_mask(
        freqs_hz,
        filter_key,
        low_freq_mhz=to_float(low_freq_mhz, default=10.0),
        high_freq_mhz=to_float(high_freq_mhz, default=800.0),
        notch_freq_mhz=to_float(notch_freq_mhz, default=50.0),
        notch_width_mhz=to_float(notch_width_mhz, default=5.0),
        notch_depth=to_float(notch_depth, default=1.0),
        taper_ratio=to_float(taper_ratio, default=0.08),
        nyquist_mhz=nyquist_mhz,
        warnings=warnings,
    )

    if bool(effective.get("skipped", False)):
        result = np.array(arr, copy=True)
    else:
        spectrum = np.fft.rfft(arr, axis=0)
        filtered = spectrum * mask[:, None]
        result = np.fft.irfft(filtered, n=n_samples, axis=0)

    meta = {
        "method": "frequency_filter_1d",
        "filter_type": filter_key,
        "sample_rate_hz": float(resolved_sample_rate_hz),
        "nyquist_mhz": nyquist_mhz,
        "effective_params": effective,
    }
    if warnings:
        meta["runtime_warnings"] = warnings
    return result.astype(np.float32, copy=False), meta


def _resolve_sample_rate_hz(
    *,
    sample_rate_hz: float | None,
    sample_rate_mhz: float | None,
    time_step_s: float | None,
    sample_interval_ns: float | None,
) -> float:
    for value, multiplier in (
        (sample_rate_hz, 1.0),
        (sample_rate_mhz, 1.0e6),
    ):
        try:
            numeric = to_float(value, default=0.0)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0:
            return numeric * multiplier

    try:
        step_s = to_float(time_step_s, default=0.0)
    except (TypeError, ValueError):
        step_s = 0.0
    if step_s > 0.0:
        return 1.0 / step_s

    try:
        step_ns = to_float(sample_interval_ns, default=0.0)
    except (TypeError, ValueError):
        step_ns = 0.0
    if step_ns > 0.0:
        return 1.0 / (step_ns * 1.0e-9)

    return 0.0


def _build_filter_mask(
    freqs_hz: np.ndarray,
    filter_type: str,
    *,
    low_freq_mhz: float,
    high_freq_mhz: float,
    notch_freq_mhz: float,
    notch_width_mhz: float,
    notch_depth: float,
    taper_ratio: float,
    nyquist_mhz: float,
    warnings: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    nyquist_hz = max(0.0, float(nyquist_mhz) * 1.0e6)
    low_hz = _mhz_to_hz(low_freq_mhz)
    high_hz = _mhz_to_hz(high_freq_mhz)
    notch_hz = _mhz_to_hz(notch_freq_mhz)
    notch_width_hz = _mhz_to_hz(notch_width_mhz)
    taper_ratio = float(np.clip(taper_ratio, 0.0, 0.5))

    if filter_type == "lowpass":
        cutoff = _clamp_cutoff(high_hz, 0.0, nyquist_hz)
        if cutoff <= 0.0:
            _append_skip_warning(warnings, "lowpass cutoff is outside the valid band")
            return np.ones_like(freqs_hz, dtype=np.float64), {"skipped": True}
        mask = _lowpass_mask(freqs_hz, cutoff, _transition_width(cutoff, nyquist_hz, taper_ratio))
        return mask, {"high_freq_mhz": cutoff / 1.0e6, "skipped": False}

    if filter_type == "highpass":
        cutoff = _clamp_cutoff(low_hz, 0.0, nyquist_hz)
        if cutoff >= nyquist_hz:
            _append_skip_warning(warnings, "highpass cutoff is outside the valid band")
            return np.ones_like(freqs_hz, dtype=np.float64), {"skipped": True}
        mask = _highpass_mask(freqs_hz, cutoff, _transition_width(cutoff, nyquist_hz, taper_ratio))
        return mask, {"low_freq_mhz": cutoff / 1.0e6, "skipped": False}

    if filter_type == "notch":
        width = max(0.0, notch_width_hz)
        center = _clamp_cutoff(notch_hz, 0.0, nyquist_hz)
        if center <= 0.0 or center >= nyquist_hz or width <= 0.0:
            _append_skip_warning(warnings, "notch band is outside the valid band")
            return np.ones_like(freqs_hz, dtype=np.float64), {"skipped": True}
        half = min(width / 2.0, max(center, nyquist_hz - center))
        band_low = max(0.0, center - half)
        band_high = min(nyquist_hz, center + half)
        band = _bandpass_mask(freqs_hz, band_low, band_high, _transition_width(width, nyquist_hz, taper_ratio))
        depth = float(np.clip(notch_depth, 0.0, 1.0))
        mask = 1.0 - depth * band
        return mask, {
            "notch_freq_mhz": center / 1.0e6,
            "notch_width_mhz": (band_high - band_low) / 1.0e6,
            "notch_depth": depth,
            "skipped": False,
        }

    low = _clamp_cutoff(low_hz, 0.0, nyquist_hz)
    high = _clamp_cutoff(high_hz, 0.0, nyquist_hz)
    if high <= 0.0:
        _append_skip_warning(warnings, "bandpass high cutoff is outside the valid band")
        return np.ones_like(freqs_hz, dtype=np.float64), {"skipped": True}
    if low >= high:
        adjusted_low = max(0.0, high * 0.15)
        warnings.append(
            build_runtime_warning(
                "frequency_filter_band_adjusted",
                "带通下限不低于上限，已按有效上限收缩下限。",
                method_id="frequency_filter_1d",
                requested_low_freq_mhz=low_freq_mhz,
                requested_high_freq_mhz=high_freq_mhz,
                effective_low_freq_mhz=adjusted_low / 1.0e6,
                effective_high_freq_mhz=high / 1.0e6,
            )
        )
        low = adjusted_low
    transition = _transition_width(max(high - low, high), nyquist_hz, taper_ratio)
    mask = _bandpass_mask(freqs_hz, low, high, transition)
    return mask, {
        "low_freq_mhz": low / 1.0e6,
        "high_freq_mhz": high / 1.0e6,
        "skipped": False,
    }


def _mhz_to_hz(value: float) -> float:
    numeric = to_float(value, default=0.0)
    return numeric * 1.0e6


def _clamp_cutoff(value_hz: float, minimum_hz: float, maximum_hz: float) -> float:
    return float(max(minimum_hz, min(maximum_hz, float(value_hz))))


def _transition_width(reference_hz: float, nyquist_hz: float, taper_ratio: float) -> float:
    if taper_ratio <= 0.0 or nyquist_hz <= 0.0:
        return 0.0
    return float(max(nyquist_hz * 0.005, abs(reference_hz) * taper_ratio))


def _smooth_step(freqs_hz: np.ndarray, start_hz: float, end_hz: float) -> np.ndarray:
    if end_hz <= start_hz:
        return (freqs_hz >= end_hz).astype(np.float64)
    x = np.clip((freqs_hz - start_hz) / (end_hz - start_hz), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _highpass_mask(freqs_hz: np.ndarray, cutoff_hz: float, transition_hz: float) -> np.ndarray:
    if cutoff_hz <= 0.0:
        return np.ones_like(freqs_hz, dtype=np.float64)
    return _smooth_step(freqs_hz, cutoff_hz - transition_hz, cutoff_hz + transition_hz)


def _lowpass_mask(freqs_hz: np.ndarray, cutoff_hz: float, transition_hz: float) -> np.ndarray:
    if cutoff_hz <= 0.0:
        return np.zeros_like(freqs_hz, dtype=np.float64)
    return 1.0 - _smooth_step(freqs_hz, cutoff_hz - transition_hz, cutoff_hz + transition_hz)


def _bandpass_mask(
    freqs_hz: np.ndarray, low_hz: float, high_hz: float, transition_hz: float
) -> np.ndarray:
    return _highpass_mask(freqs_hz, low_hz, transition_hz) * _lowpass_mask(
        freqs_hz, high_hz, transition_hz
    )


def _append_skip_warning(warnings: list[dict[str, Any]], reason: str) -> None:
    warnings.append(
        build_runtime_warning(
            "frequency_filter_skipped",
            "频域滤波有效频带为空，已保持输入不变。",
            method_id="frequency_filter_1d",
            reason=reason,
        )
    )
