"""Native zero-phase frequency filtering along the A-scan axis."""
from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.infrastructure.processing.algorithms.common import (
    as_float,
    ensure_matrix,
    normalize_output,
    warning,
)


def method_frequency_filter(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    sample_rate = resolve_sample_rate(params)
    filter_type = str(params.get("filter_type") or "bandpass").strip().lower()
    if filter_type not in {"bandpass", "lowpass", "highpass", "notch"}:
        raise ValueError(f"Unsupported filter_type: {filter_type}")
    if sample_rate <= 0.0:
        warnings.append(warning("frequency_sampling_missing", "缺少采样率或时间窗，频域滤波已跳过。", "frequency_filter_1d"))
        return normalize_output(
            "frequency_filter_1d",
            arr.copy(),
            {"method": "frequency_filter_1d", "filter_type": filter_type, "skipped": True},
            warnings,
        )
    frequencies = np.fft.rfftfreq(arr.shape[0], d=1.0 / sample_rate)
    mask, effective, mask_warnings = build_mask(frequencies, sample_rate / 2.0, filter_type, params)
    warnings.extend(mask_warnings)
    if effective.get("skipped"):
        result = arr.copy()
    else:
        spectrum = np.fft.rfft(arr, axis=0)
        spectrum *= mask[:, None]
        result = np.fft.irfft(spectrum, n=arr.shape[0], axis=0)
    meta = {
        "method": "frequency_filter_1d",
        "filter_type": filter_type,
        "sample_rate_hz": sample_rate,
        "nyquist_mhz": sample_rate / 2.0e6,
        "effective_params": effective,
    }
    return normalize_output("frequency_filter_1d", result, meta, warnings)


def resolve_sample_rate(params: dict[str, Any]) -> float:
    sample_rate = as_float(params.get("sample_rate_hz"), 0.0)
    if sample_rate > 0.0:
        return sample_rate
    sample_rate_mhz = as_float(params.get("sample_rate_mhz"), 0.0)
    if sample_rate_mhz > 0.0:
        return sample_rate_mhz * 1.0e6
    step_s = as_float(params.get("time_step_s"), 0.0)
    if step_s > 0.0:
        return 1.0 / step_s
    step_ns = as_float(params.get("sample_interval_ns"), 0.0)
    return 1.0 / (step_ns * 1.0e-9) if step_ns > 0.0 else 0.0


def build_mask(
    frequencies: np.ndarray,
    nyquist_hz: float,
    filter_type: str,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    taper = float(np.clip(as_float(params.get("taper_ratio"), 0.08), 0.0, 0.5))
    low = clamp_hz(as_float(params.get("low_freq_mhz"), 10.0) * 1.0e6, nyquist_hz)
    high = clamp_hz(as_float(params.get("high_freq_mhz"), 800.0) * 1.0e6, nyquist_hz)
    if filter_type == "lowpass":
        return lowpass_case(frequencies, high, nyquist_hz, taper, warnings)
    if filter_type == "highpass":
        return highpass_case(frequencies, low, nyquist_hz, taper, warnings)
    if filter_type == "notch":
        center = clamp_hz(as_float(params.get("notch_freq_mhz"), 50.0) * 1.0e6, nyquist_hz)
        width = max(0.0, as_float(params.get("notch_width_mhz"), 5.0) * 1.0e6)
        depth = float(np.clip(as_float(params.get("notch_depth"), 1.0), 0.0, 1.0))
        return notch_case(frequencies, center, width, depth, nyquist_hz, taper, warnings)
    return bandpass_case(frequencies, low, high, nyquist_hz, taper, warnings)


def lowpass_case(freqs: np.ndarray, cutoff: float, nyquist: float, taper: float, warnings: list[dict[str, Any]]):
    if cutoff <= 0.0:
        return skipped(freqs, warnings, "lowpass cutoff is outside the valid band")
    mask = lowpass_mask(freqs, cutoff, transition(cutoff, nyquist, taper))
    return mask, {"high_freq_mhz": cutoff / 1.0e6, "skipped": False}, warnings


def highpass_case(freqs: np.ndarray, cutoff: float, nyquist: float, taper: float, warnings: list[dict[str, Any]]):
    if cutoff >= nyquist:
        return skipped(freqs, warnings, "highpass cutoff is outside the valid band")
    mask = highpass_mask(freqs, cutoff, transition(cutoff, nyquist, taper))
    return mask, {"low_freq_mhz": cutoff / 1.0e6, "skipped": False}, warnings


def notch_case(
    freqs: np.ndarray,
    center: float,
    width: float,
    depth: float,
    nyquist: float,
    taper: float,
    warnings: list[dict[str, Any]],
):
    if center <= 0.0 or center >= nyquist or width <= 0.0:
        return skipped(freqs, warnings, "notch band is outside the valid band")
    half = min(width / 2.0, max(center, nyquist - center))
    low, high = max(0.0, center - half), min(nyquist, center + half)
    band = bandpass_mask(freqs, low, high, transition(width, nyquist, taper))
    meta = {"notch_freq_mhz": center / 1.0e6, "notch_width_mhz": (high - low) / 1.0e6, "notch_depth": depth, "skipped": False}
    return 1.0 - depth * band, meta, warnings


def bandpass_case(
    freqs: np.ndarray,
    low: float,
    high: float,
    nyquist: float,
    taper: float,
    warnings: list[dict[str, Any]],
):
    if high <= 0.0:
        return skipped(freqs, warnings, "bandpass high cutoff is outside the valid band")
    if low >= high:
        adjusted = max(0.0, high * 0.15)
        warnings.append(warning("frequency_filter_band_adjusted", "带通下限不低于上限，已自动收缩。", "frequency_filter_1d", effective_low_freq_mhz=adjusted / 1.0e6))
        low = adjusted
    width = transition(max(high - low, high), nyquist, taper)
    return bandpass_mask(freqs, low, high, width), {"low_freq_mhz": low / 1.0e6, "high_freq_mhz": high / 1.0e6, "skipped": False}, warnings


def skipped(freqs: np.ndarray, warnings: list[dict[str, Any]], reason: str):
    warnings.append(warning("frequency_filter_skipped", "频域滤波有效频带为空，已保持输入不变。", "frequency_filter_1d", reason=reason))
    return np.ones_like(freqs, dtype=np.float64), {"skipped": True}, warnings


def clamp_hz(value: float, nyquist: float) -> float:
    return float(max(0.0, min(float(nyquist), float(value))))


def transition(reference: float, nyquist: float, taper: float) -> float:
    if taper <= 0.0 or nyquist <= 0.0:
        return 0.0
    return float(max(nyquist * 0.005, abs(reference) * taper))


def smooth_step(freqs: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        return (freqs >= end).astype(np.float64)
    x = np.clip((freqs - start) / (end - start), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def highpass_mask(freqs: np.ndarray, cutoff: float, width: float) -> np.ndarray:
    if cutoff <= 0.0:
        return np.ones_like(freqs, dtype=np.float64)
    return smooth_step(freqs, cutoff - width, cutoff + width)


def lowpass_mask(freqs: np.ndarray, cutoff: float, width: float) -> np.ndarray:
    if cutoff <= 0.0:
        return np.zeros_like(freqs, dtype=np.float64)
    return 1.0 - smooth_step(freqs, cutoff - width, cutoff + width)


def bandpass_mask(freqs: np.ndarray, low: float, high: float, width: float) -> np.ndarray:
    return highpass_mask(freqs, low, width) * lowpass_mask(freqs, high, width)
