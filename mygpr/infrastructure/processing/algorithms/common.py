"""Numerical helpers shared by native GPR processing algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Any

import numpy as np


def as_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def warning(code: str, message: str, method_id: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "method_id": method_id, **details}


def ensure_matrix(data: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"processing data must be a non-empty 2D matrix, got {arr.shape!r}")
    if np.isfinite(arr).all():
        return arr, []
    finite = np.isfinite(arr)
    fill = float(np.mean(arr[finite])) if finite.any() else 0.0
    clean = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    return clean, [warning("data_sanitized", "输入数据包含 NaN/Inf，已使用均值填充。", "", fill_value=fill)]


def normalize_output(
    method_id: str,
    data: Any,
    metadata: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(data)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"processing result must be a non-empty 2D matrix, got {arr.shape!r}")
    all_warnings = [dict(item) for item in (warnings or [])]
    if not np.isfinite(arr).all():
        finite = np.isfinite(arr)
        fill = float(np.mean(arr[finite])) if finite.any() else 0.0
        arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
        all_warnings.append(
            warning("data_sanitized", "输出结果包含 NaN/Inf，已使用均值填充。", method_id, fill_value=fill)
        )
    meta = {"method_id": method_id, **dict(metadata or {})}
    if all_warnings:
        for item in all_warnings:
            item["method_id"] = item.get("method_id") or method_id
        meta["runtime_warnings"] = all_warnings
    return arr.astype(np.float32, copy=False), meta


def window_sums_axis0(arr: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    padded = np.empty((arr.shape[0] + 1, arr.shape[1]), dtype=np.float64)
    padded[0] = 0.0
    np.cumsum(arr, axis=0, out=padded[1:])
    return padded[ends] - padded[starts]


def window_sums_axis1(arr: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    padded = np.empty((arr.shape[0], arr.shape[1] + 1), dtype=np.float64)
    padded[:, 0] = 0.0
    np.cumsum(arr, axis=1, out=padded[:, 1:])
    return padded[:, ends] - padded[:, starts]


def gprpy_dewow(data: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    samples = arr.shape[0]
    window = max(1, as_int(window, 1))
    if window >= samples:
        return (arr - np.mean(arr, axis=0, keepdims=True)).astype(np.float32)
    half = int(np.ceil(window / 2.0))
    out = np.empty_like(arr)
    leading = np.mean(arr[: half + 1], axis=0)
    out[: half + 1] = arr[: half + 1] - leading
    indices = np.arange(half, samples - half, dtype=np.int32)
    if indices.size:
        means = window_sums_axis0(arr, indices - half, indices + half + 1) / (2 * half + 1)
        out[indices] = arr[indices] - means
    trailing = np.mean(arr[samples - half : samples + 1], axis=0)
    out[samples - half : samples + 1] = arr[samples - half : samples + 1] - trailing
    return out.astype(np.float32, copy=False)


def gprpy_remove_mean_trace(data: np.ndarray, ntraces: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    traces = arr.shape[1]
    ntraces = max(1, as_int(ntraces, 1))
    if ntraces >= traces:
        return (arr - np.mean(arr, axis=1, keepdims=True)).astype(np.float32)
    half = int(np.ceil(ntraces / 2.0))
    out = np.empty_like(arr)
    leading = np.mean(arr[:, : half + 1], axis=1, keepdims=True)
    out[:, : half + 1] = arr[:, : half + 1] - leading
    indices = np.arange(half, traces - half, dtype=np.int32)
    if indices.size:
        means = window_sums_axis1(arr, indices - half, indices + half + 1) / (2 * half + 1)
        out[:, indices] = arr[:, indices] - means
    trailing = np.mean(arr[:, traces - half : traces + 1], axis=1, keepdims=True)
    out[:, traces - half : traces + 1] = arr[:, traces - half : traces + 1] - trailing
    return out.astype(np.float32, copy=False)


def local_l2_energy(data: np.ndarray, window: int, eps: float = 1.0e-8) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    samples = arr.shape[0]
    window = max(1, as_int(window, 1))
    if window > samples:
        energy = np.maximum(np.linalg.norm(arr, axis=0, keepdims=True), eps)
        return np.repeat(energy, samples, axis=0)
    half = int(np.ceil(window / 2.0))
    squared = arr * arr
    out = np.empty_like(arr)
    leading = np.maximum(np.linalg.norm(arr[: half + 1], axis=0), eps)
    out[: half + 1] = leading
    indices = np.arange(half, samples - half, dtype=np.int32)
    if indices.size:
        out[indices] = np.maximum(
            np.sqrt(window_sums_axis0(squared, indices - half, indices + half + 1)), eps
        )
    trailing = np.maximum(np.linalg.norm(arr[samples - half : samples + 1], axis=0), eps)
    out[samples - half : samples + 1] = trailing
    return out


@dataclass(frozen=True, slots=True)
class TimeSelection:
    start: int
    end: int
    source: str


def resolve_time_selection(shape: tuple[int, int], params: dict[str, Any]) -> TimeSelection:
    samples = max(1, int(shape[0]))
    if params.get("time_start_idx") not in (None, "") or as_float(params.get("time_end_idx"), 0.0) > 0:
        start = as_int(params.get("time_start_idx"), 0)
        end = as_int(params.get("time_end_idx"), samples)
        return clamp_selection(start, samples if end <= 0 else end, samples, "samples")
    if as_float(params.get("time_start_ns"), 0.0) > 0 or as_float(params.get("time_end_ns"), 0.0) > 0:
        total_ns = max(as_float(params.get("time_window_ns"), float(samples)), 1.0e-12)
        scale = samples / total_ns
        start = floor(max(0.0, as_float(params.get("time_start_ns"), 0.0)) * scale)
        end_ns = as_float(params.get("time_end_ns"), 0.0)
        end = ceil(end_ns * scale) if end_ns > 0 else samples
        return clamp_selection(start, end, samples, "ns")
    return TimeSelection(0, samples, "full")


def clamp_selection(start: int, end: int, samples: int, source: str) -> TimeSelection:
    start_i = max(0, min(int(start), samples - 1))
    end_i = max(start_i + 1, min(int(end), samples))
    return TimeSelection(start_i, end_i, "full" if start_i == 0 and end_i == samples else source)


def apply_time_selection(original: np.ndarray, processed: np.ndarray, selection: TimeSelection, taper: int) -> np.ndarray:
    if selection.start == 0 and selection.end >= original.shape[0]:
        return np.asarray(processed, dtype=np.float32)
    out = np.asarray(original, dtype=np.float32).copy()
    start, end = selection.start, selection.end
    taper_i = max(0, min(int(taper), (end - start) // 2))
    if taper_i == 0:
        out[start:end] = processed[start:end]
        return out
    weights = np.ones((end - start, 1), dtype=np.float32)
    weights[:taper_i, 0] = np.linspace(0.0, 1.0, taper_i + 2, dtype=np.float32)[1:-1]
    weights[-taper_i:, 0] = np.linspace(1.0, 0.0, taper_i + 2, dtype=np.float32)[1:-1]
    out[start:end] = original[start:end] * (1.0 - weights) + processed[start:end] * weights
    return out
