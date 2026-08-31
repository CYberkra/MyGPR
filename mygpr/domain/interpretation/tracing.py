"""Deterministic, presentation-free basal-interface tracing primitives."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .models import InterfaceTraceConfig


def _envelope(matrix: np.ndarray) -> np.ndarray:
    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("B-scan matrix must be two-dimensional")
    try:
        from scipy.signal import hilbert

        envelope = np.abs(hilbert(data, axis=0))
    except ImportError:
        envelope = np.abs(data)
    envelope[~np.isfinite(envelope)] = 0.0
    scale = np.nanpercentile(envelope, 95.0, axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return envelope / scale[None, :]


def _moving_median(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size < 3:
        return values.copy()
    output = values.copy()
    for index in range(values.size):
        lo = max(0, index - radius)
        hi = min(values.size, index + radius + 1)
        output[index] = float(np.nanmedian(values[lo:hi]))
    return output


def trace_interface(
    matrix: np.ndarray,
    anchors: Iterable[tuple[int, float]],
    *,
    start_trace: int | None = None,
    end_trace: int | None = None,
    config: InterfaceTraceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or InterfaceTraceConfig()
    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim != 2 or not data.size:
        raise ValueError("B-scan data are empty")
    sample_count, trace_count = data.shape
    unique: dict[int, float] = {}
    for trace, sample in anchors:
        safe_trace = int(np.clip(int(trace), 0, max(trace_count - 1, 0)))
        unique[safe_trace] = float(sample)
    if not unique:
        raise ValueError("At least one control point is required")
    anchor_traces = np.asarray(sorted(unique), dtype=np.int64)
    anchor_samples = np.asarray([unique[int(trace)] for trace in anchor_traces], dtype=np.float64)
    lo_trace = int(anchor_traces.min() if start_trace is None else np.clip(start_trace, 0, trace_count - 1))
    hi_trace = int(anchor_traces.max() if end_trace is None else np.clip(end_trace, 0, trace_count - 1))
    if hi_trace < lo_trace:
        lo_trace, hi_trace = hi_trace, lo_trace
    traces = np.arange(lo_trace, hi_trace + 1, dtype=np.int64)
    min_sample = int(np.clip(cfg.min_sample, 0, sample_count - 1))
    max_sample = sample_count - 1 if cfg.max_sample is None else int(np.clip(cfg.max_sample, min_sample, sample_count - 1))
    anchor_samples = np.clip(anchor_samples, min_sample, max_sample)
    prior = np.interp(traces, anchor_traces, anchor_samples, left=anchor_samples[0], right=anchor_samples[-1])
    envelope = _envelope(data)
    result = np.empty(traces.size, dtype=np.float64)
    anchor_lookup = {int(trace): float(sample) for trace, sample in zip(anchor_traces, anchor_samples)}
    previous = float(prior[0])
    for index, trace in enumerate(traces):
        if int(trace) in anchor_lookup:
            chosen = anchor_lookup[int(trace)]
        else:
            expected = float(prior[index])
            center = previous if index else expected
            lo = max(min_sample, int(round(min(center, expected) - cfg.search_half_window)))
            hi = min(max_sample, int(round(max(center, expected) + cfg.search_half_window)))
            candidates = np.arange(lo, hi + 1, dtype=np.int64)
            score = envelope[candidates, int(trace)]
            score = score - cfg.anchor_weight * np.abs(candidates - expected)
            score = score - cfg.continuity_weight * np.abs(candidates - previous)
            if cfg.max_step_samples > 0:
                allowed = np.abs(candidates - previous) <= cfg.max_step_samples
                if allowed.any():
                    score = np.where(allowed, score, -np.inf)
            chosen = float(candidates[int(np.nanargmax(score))])
        result[index] = chosen
        previous = chosen
    result = _moving_median(result, int(cfg.smooth_radius))
    for trace, sample in anchor_lookup.items():
        if lo_trace <= trace <= hi_trace:
            result[trace - lo_trace] = sample
    return traces, np.clip(result, min_sample, max_sample).astype(np.float32)


def decimate_trace_path(
    traces: np.ndarray,
    samples: np.ndarray,
    *,
    max_points: int = 160,
    mandatory_traces: Iterable[int] = (),
) -> list[tuple[int, float]]:
    traces = np.asarray(traces, dtype=np.int64)
    samples = np.asarray(samples, dtype=np.float64)
    if traces.size != samples.size or traces.size == 0:
        return []
    stride = max(1, int(np.ceil(traces.size / max(max_points, 2))))
    selected = set(int(value) for value in traces[::stride])
    selected.update((int(traces[0]), int(traces[-1])))
    selected.update(int(value) for value in mandatory_traces if int(traces[0]) <= int(value) <= int(traces[-1]))
    lookup = {int(trace): float(sample) for trace, sample in zip(traces, samples)}
    return [(trace, lookup[trace]) for trace in sorted(selected) if trace in lookup]


__all__ = ["decimate_trace_path", "trace_interface"]
