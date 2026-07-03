#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time-range helpers for background suppression methods."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Any

import numpy as np

from core.scalar_utils import first_scalar, to_float, to_int


@dataclass(frozen=True)
class TimeRangeSelection:
    """Resolved sample-index range for a time-limited processing step."""

    start_idx: int
    end_idx: int
    source: str

    @property
    def is_full_range(self) -> bool:
        return self.start_idx == 0 and self.source == "full"


def resolve_time_range_selection(
    data_shape: tuple[int, int],
    *,
    time_start_idx: Any = None,
    time_end_idx: Any = None,
    time_start_ns: Any = None,
    time_end_ns: Any = None,
    time_window_ns: Any = None,
) -> TimeRangeSelection:
    """Resolve optional sample or ns bounds to clamped row indices.

    ``time_end_*`` values of 0/None mean "to the bottom". This preserves the
    legacy default where no explicit range means the full B-scan is processed.
    """

    samples = max(1, int(data_shape[0]))

    if _has_value(time_start_idx) or _has_positive_value(time_end_idx):
        start = to_int(time_start_idx, default=0)
        end = to_int(time_end_idx, default=samples)
        if end <= 0:
            end = samples
        return _clamp_selection(start, end, samples, source="samples")

    if _has_positive_value(time_start_ns) or _has_positive_value(time_end_ns):
        start_ns = max(0.0, to_float(time_start_ns, default=0.0))
        end_ns = to_float(time_end_ns, default=0.0)
        total_ns = to_float(time_window_ns, default=float(samples))
        if total_ns <= 0.0:
            total_ns = float(samples)
        scale = float(samples) / total_ns
        start = int(floor(start_ns * scale))
        end = int(ceil(end_ns * scale)) if end_ns > 0.0 else samples
        return _clamp_selection(start, end, samples, source="ns")

    return TimeRangeSelection(0, samples, "full")


def apply_time_range_to_result(
    original: np.ndarray,
    processed: np.ndarray,
    selection: TimeRangeSelection,
    *,
    edge_taper_samples: int = 0,
) -> np.ndarray:
    """Copy a processed result into the selected time range only."""

    arr = np.asarray(original, dtype=np.float32)
    proc = np.asarray(processed, dtype=np.float32)
    if arr.shape != proc.shape:
        raise ValueError(f"处理前后数据 shape 不一致: {arr.shape} != {proc.shape}")

    if selection.start_idx == 0 and selection.end_idx >= arr.shape[0]:
        return proc.astype(np.float32, copy=False)

    out = np.array(arr, copy=True)
    start = selection.start_idx
    end = selection.end_idx
    window_len = max(1, end - start)
    taper = max(0, min(int(edge_taper_samples), window_len // 2))

    if taper <= 0:
        out[start:end, :] = proc[start:end, :]
        return out.astype(np.float32, copy=False)

    weights = np.ones((window_len, 1), dtype=np.float32)
    ramp_up = np.linspace(0.0, 1.0, taper + 2, dtype=np.float32)[1:-1]
    ramp_down = np.linspace(1.0, 0.0, taper + 2, dtype=np.float32)[1:-1]
    weights[:taper, 0] = ramp_up
    weights[-taper:, 0] = ramp_down

    selected_original = arr[start:end, :]
    selected_processed = proc[start:end, :]
    out[start:end, :] = selected_original * (1.0 - weights) + selected_processed * weights
    return out.astype(np.float32, copy=False)


def _clamp_selection(
    start: int, end: int, samples: int, *, source: str
) -> TimeRangeSelection:
    start_idx = max(0, min(int(start), samples - 1))
    end_idx = max(start_idx + 1, min(int(end), samples))
    if start_idx == 0 and end_idx == samples:
        return TimeRangeSelection(0, samples, "full")
    return TimeRangeSelection(start_idx, end_idx, source)


def _has_value(value: Any) -> bool:
    scalar = first_scalar(value)
    if scalar is None:
        return False
    if isinstance(scalar, str):
        return scalar.strip() != ""
    return True


def _has_positive_value(value: Any) -> bool:
    return to_float(value, default=0.0) > 0.0
