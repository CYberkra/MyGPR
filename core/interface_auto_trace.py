#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative control-point constrained interface auto tracing.

This module intentionally provides a deterministic signal-processing assistant,
not a learned-model prediction.  It follows local envelope energy while
penalising slope and deviation from user anchors.  Results are returned in raw
trace/sample coordinates and must remain subject to human review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class InterfaceAutoTraceConfig:
    search_half_window: int = 18
    max_step_samples: int = 8
    smooth_radius: int = 2
    anchor_weight: float = 0.08
    continuity_weight: float = 0.12
    min_sample: int = 0
    max_sample: int | None = None


def _envelope(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("B-scan matrix must be two-dimensional")
    try:
        from scipy.signal import hilbert

        env = np.abs(hilbert(arr, axis=0))
    except Exception:
        # Absolute amplitude remains deterministic and dependency-free.
        env = np.abs(arr)
    env[~np.isfinite(env)] = 0.0
    # Robust per-trace normalisation prevents a few high-energy traces from
    # dominating the whole path.
    scale = np.nanpercentile(env, 95.0, axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return env / scale[None, :]


def _moving_median(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size < 3:
        return values.copy()
    out = values.copy()
    for index in range(values.size):
        lo = max(0, index - radius)
        hi = min(values.size, index + radius + 1)
        out[index] = float(np.nanmedian(values[lo:hi]))
    return out


def _anchor_arrays(anchors: Iterable[tuple[int, float]], trace_count: int) -> tuple[np.ndarray, np.ndarray]:
    unique: dict[int, float] = {}
    for trace, sample in anchors:
        t = int(np.clip(int(trace), 0, max(trace_count - 1, 0)))
        unique[t] = float(sample)
    if not unique:
        raise ValueError("At least one control point is required")
    traces = np.asarray(sorted(unique), dtype=np.int64)
    samples = np.asarray([unique[int(trace)] for trace in traces], dtype=np.float64)
    return traces, samples


def trace_interface(
    matrix: np.ndarray,
    anchors: Iterable[tuple[int, float]],
    *,
    start_trace: int | None = None,
    end_trace: int | None = None,
    config: InterfaceAutoTraceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace an interface between/around user anchors.

    Returns ``(trace_indices, sample_indices)``.  User anchor samples are
    restored exactly after smoothing.
    """
    cfg = config or InterfaceAutoTraceConfig()
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or not arr.size:
        raise ValueError("B-scan data are empty")
    sample_count, trace_count = arr.shape
    anchor_traces, anchor_samples = _anchor_arrays(anchors, trace_count)
    lo_trace = int(anchor_traces.min() if start_trace is None else np.clip(start_trace, 0, trace_count - 1))
    hi_trace = int(anchor_traces.max() if end_trace is None else np.clip(end_trace, 0, trace_count - 1))
    if hi_trace < lo_trace:
        lo_trace, hi_trace = hi_trace, lo_trace
    traces = np.arange(lo_trace, hi_trace + 1, dtype=np.int64)
    if traces.size == 0:
        raise ValueError("Selected trace interval is empty")

    min_sample = int(np.clip(cfg.min_sample, 0, sample_count - 1))
    max_sample = sample_count - 1 if cfg.max_sample is None else int(np.clip(cfg.max_sample, min_sample, sample_count - 1))
    anchor_samples = np.clip(anchor_samples, min_sample, max_sample)
    prior = np.interp(traces, anchor_traces, anchor_samples, left=anchor_samples[0], right=anchor_samples[-1])
    env = _envelope(arr)
    result = np.empty(traces.size, dtype=np.float64)

    anchor_lookup = {int(t): float(s) for t, s in zip(anchor_traces, anchor_samples)}
    previous = float(prior[0])
    for index, trace in enumerate(traces):
        if int(trace) in anchor_lookup:
            sample = anchor_lookup[int(trace)]
            result[index] = sample
            previous = sample
            continue
        expected = float(prior[index])
        center = previous if index else expected
        lo = max(min_sample, int(round(min(center, expected) - cfg.search_half_window)))
        hi = min(max_sample, int(round(max(center, expected) + cfg.search_half_window)))
        if hi < lo:
            lo = hi = int(np.clip(round(expected), min_sample, max_sample))
        candidates = np.arange(lo, hi + 1, dtype=np.int64)
        energy = env[candidates, int(trace)]
        score = energy
        score = score - cfg.anchor_weight * np.abs(candidates - expected)
        score = score - cfg.continuity_weight * np.abs(candidates - previous)
        if cfg.max_step_samples > 0:
            allowed = np.abs(candidates - previous) <= cfg.max_step_samples
            if allowed.any():
                score = np.where(allowed, score, -np.inf)
        best = int(candidates[int(np.nanargmax(score))])
        result[index] = float(best)
        previous = float(best)

    result = _moving_median(result, max(0, int(cfg.smooth_radius)))
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
    """Convert a dense path to a stable, editable key-point representation."""
    traces = np.asarray(traces, dtype=np.int64)
    samples = np.asarray(samples, dtype=np.float64)
    if traces.size != samples.size or traces.size == 0:
        return []
    stride = max(1, int(np.ceil(traces.size / max(max_points, 2))))
    selected = set(int(value) for value in traces[::stride])
    selected.add(int(traces[0])); selected.add(int(traces[-1]))
    selected.update(int(value) for value in mandatory_traces if int(traces[0]) <= int(value) <= int(traces[-1]))
    lookup = {int(t): float(s) for t, s in zip(traces, samples)}
    return [(trace, lookup[trace]) for trace in sorted(selected) if trace in lookup]


__all__ = ["InterfaceAutoTraceConfig", "trace_interface", "decimate_trace_path"]
