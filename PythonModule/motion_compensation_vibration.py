#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic vibration-safe suppression for UAV-GPR motion workflows.

V1 不依赖 RPM，也不做旋翼频率识别。该实现只做保守的周期条带抑制：
- 沿道方向对指定频带做自适应衰减
- 用显著性行保护目标/双曲线样异常
- 叠加极弱的 CCBS-inspired 平均背景抑制
- 如有 angular_rate_* 或显式 trajectory_jitter_* 元数据，可仅用于确定抑制强度并返回平滑后的 metadata 副本
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d

from core.quality_metrics import build_saliency_map


EPS = 1.0e-6
GUIDANCE_JITTER_KEYS = (
    "trajectory_jitter_m",
    "trajectory_jitter_score",
    "lateral_jitter_m",
)


def _normalize_window(window: int, size: int) -> int:
    """Return a valid odd smoothing window for the given vector size."""
    if size <= 1:
        return 1
    resolved = max(3, min(int(window), size))
    if resolved % 2 == 0:
        resolved = max(1, resolved - 1)
    return max(1, resolved)


def _extract_numeric_trace_field(
    trace_metadata: dict[str, Any],
    key: str,
    trace_count: int,
) -> np.ndarray | None:
    """Extract a finite 1D trace-aligned metadata field if available."""
    if key not in trace_metadata:
        return None
    values = np.asarray(trace_metadata[key], dtype=np.float64)
    if values.ndim != 1 or values.size < trace_count:
        return None
    trimmed = np.array(values[:trace_count], copy=True)
    if not np.all(np.isfinite(trimmed)):
        return None
    return trimmed


def _high_frequency_score(values: np.ndarray, smooth_window: int) -> float:
    """Estimate how much high-frequency jitter is present in a 1D trace series."""
    if values.size <= 3:
        return 0.0
    window = _normalize_window(smooth_window, values.size)
    smooth = uniform_filter1d(values, size=window, mode="nearest")
    residual = values - smooth
    baseline = max(float(np.std(values)), EPS)
    score = float(np.mean(np.abs(residual)) / baseline)
    return float(np.clip(score, 0.0, 1.0))


def _resolve_guidance(
    trace_metadata: dict[str, Any] | None,
    trace_count: int,
    smooth_window: int,
) -> tuple[str, dict[str, np.ndarray], float, list[str]]:
    """Resolve optional metadata guidance and smoothed metadata copies."""
    if trace_metadata is None:
        return "radar_only_fallback", {}, 0.0, []

    smoothed_updates: dict[str, np.ndarray] = {}
    scores: list[float] = []
    used_keys: list[str] = []

    angular_keys = sorted(
        key for key in trace_metadata.keys() if str(key).startswith("angular_rate_")
    )
    for key in angular_keys:
        values = _extract_numeric_trace_field(trace_metadata, key, trace_count)
        if values is None:
            continue
        window = _normalize_window(smooth_window, values.size)
        smoothed = uniform_filter1d(values, size=window, mode="nearest")
        smoothed_updates[key] = smoothed.astype(np.float64)
        scores.append(_high_frequency_score(values, smooth_window=window))
        used_keys.append(key)

    if used_keys:
        return (
            "angular_rate_guided",
            smoothed_updates,
            float(np.mean(scores)) if scores else 0.0,
            used_keys,
        )

    for key in GUIDANCE_JITTER_KEYS:
        values = _extract_numeric_trace_field(trace_metadata, key, trace_count)
        if values is None:
            continue
        window = _normalize_window(smooth_window, values.size)
        smoothed_updates[key] = uniform_filter1d(values, size=window, mode="nearest").astype(
            np.float64
        )
        scores.append(_high_frequency_score(values, smooth_window=window))
        used_keys.append(key)

    if used_keys:
        return (
            "trajectory_jitter_guided",
            smoothed_updates,
            float(np.mean(scores)) if scores else 0.0,
            used_keys,
        )

    return "radar_only_fallback", {}, 0.0, []


def method_motion_compensation_vibration(
    data: np.ndarray,
    trace_metadata: dict[str, Any] | None = None,
    trace_band: tuple[float, float] = (0.05, 0.18),
    smooth_window: int = 9,
    preserve_row_percentile: float = 94.0,
    preserve_mix: float = 0.35,
    background_mix: float = 0.02,
    max_restore_gain: float = 1.25,
    **kwargs,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Conservative vibration/background suppression without RPM dependence."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("振动补偿需要二维 B-scan 数据")

    samples, traces = arr.shape
    output = np.array(arr, copy=True)
    meta: dict[str, Any] = {
        "method": "motion_compensation_vibration",
        "rpm_required": False,
        "trace_count": int(traces),
        "sample_count": int(samples),
        "deterministic": True,
        "trace_band": [float(trace_band[0]), float(trace_band[1])],
        "preserve_row_percentile": float(preserve_row_percentile),
        "preserve_mix": float(preserve_mix),
        "background_mix": float(background_mix),
        "max_restore_gain": float(max_restore_gain),
        "provenance": {
            "workflow": "saliency_guarded_lateral_band_suppression_v1",
            "background_model": "ccbs_inspired_mean_trace_guard",
            "rpm_notch_filtering": False,
        },
    }

    if traces <= 2 or samples <= 2:
        meta["skipped"] = True
        meta["reason"] = "数据尺寸过小，跳过振动补偿"
        return output, meta

    guidance_source, metadata_updates, jitter_score, guidance_keys = _resolve_guidance(
        trace_metadata,
        trace_count=traces,
        smooth_window=smooth_window,
    )
    meta["guidance_source"] = guidance_source
    meta["fallback_used"] = guidance_source == "radar_only_fallback"
    meta["guidance_keys"] = guidance_keys
    meta["metadata_jitter_score"] = float(jitter_score)
    if metadata_updates:
        meta["trace_metadata_updates"] = metadata_updates

    spectral_strength = float(np.clip(0.92 + 0.04 * jitter_score, 0.92, 0.96))
    meta["spectral_strength"] = spectral_strength

    arr64 = np.asarray(arr, dtype=np.float64)
    saliency = build_saliency_map(arr64)
    row_saliency = np.mean(saliency, axis=1)
    row_saliency_max = max(float(np.max(row_saliency)), EPS)
    row_saliency = row_saliency / row_saliency_max
    row_weight = np.clip((1.0 - row_saliency) ** 1.5, 0.05, 1.0)

    spectrum = np.fft.rfft(arr64, axis=1)
    freqs = np.fft.rfftfreq(traces, d=1.0)
    low = max(0.0, float(trace_band[0]))
    high = min(float(freqs[-1]), float(trace_band[1]))
    if high <= low:
        high = min(float(freqs[-1]), low + 0.04)
    band_mask = (freqs >= low) & (freqs <= high)
    if band_mask.size:
        band_mask[0] = False
    spectrum[:, band_mask] *= 1.0 - spectral_strength * row_weight[:, np.newaxis]
    filtered = np.fft.irfft(spectrum, n=traces, axis=1)

    preserve_threshold = float(np.percentile(row_saliency, preserve_row_percentile))
    preserve_rows = (row_saliency >= preserve_threshold).astype(np.float64)
    preserve_rows = uniform_filter1d(
        preserve_rows,
        size=_normalize_window(smooth_window, preserve_rows.size),
        mode="nearest",
    )
    preserve_mask = np.clip(preserve_rows[:, np.newaxis], 0.0, 1.0)

    blended = (1.0 - preserve_mix * preserve_mask) * filtered + (
        preserve_mix * preserve_mask
    ) * arr64

    # 极弱均值背景抑制，思路参考 CCBS 中“参考波/平均背景”背景估计，但这里仅用均值背景且对显著行做保护。
    mean_background = np.mean(arr64, axis=1, keepdims=True)
    blended = blended - background_mix * (1.0 - preserve_mask) * mean_background

    restore_ratio = float(
        np.percentile(np.abs(arr64), 95.0)
        / max(np.percentile(np.abs(blended), 95.0), EPS)
    )
    restore_gain = float(np.clip(restore_ratio, 1.0, max_restore_gain))
    output = np.asarray(blended * restore_gain, dtype=np.float32)

    meta.update(
        {
            "restore_gain": restore_gain,
            "preserved_row_count": int(np.count_nonzero(preserve_rows > 0.1)),
            "preserve_threshold": preserve_threshold,
        }
    )
    return output, meta
