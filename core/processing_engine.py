#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified ndarray processing engine for GUI, workflow, and batch execution."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.methods_registry import PROCESSING_METHODS
from core.runtime_warnings import build_runtime_warning, merge_runtime_warnings


class ProcessingEngineError(RuntimeError):
    """Raised when a processing method cannot be executed."""


def run_processing_method(
    data: np.ndarray,
    method_id: str,
    params: Optional[Dict[str, Any]] = None,
    cancel_checker=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run a single processing method on a 2D ndarray.

    Returns a fresh result array plus metadata.
    """
    method_info = PROCESSING_METHODS.get(method_id)
    if not method_info:
        raise ProcessingEngineError(f"未知方法: {method_id}")

    input_data = _ensure_2d_array(data)
    runtime_params = {
        k: v for k, v in (params or {}).items() if not str(k).startswith("_")
    }
    if cancel_checker is not None:
        runtime_params.setdefault("cancel_checker", cancel_checker)

    func = method_info.get("func")
    if callable(func):
        result = func(np.array(input_data, copy=True), **runtime_params)
        return _normalize_result(method_id, result)

    return _run_legacy_adapter(method_id, input_data, runtime_params)


def prepare_runtime_params(
    method_id: str,
    params: dict[str, Any] | None,
    header_info: dict[str, Any] | None,
    trace_metadata: dict[str, np.ndarray] | None,
    data_shape: tuple[int, int],
) -> dict[str, Any]:
    """Inject runtime-only parameters needed by some methods.

    This is not an auto-tuning system. It only supplies hidden runtime context,
    such as real time-step information for zero-time correction.
    """
    runtime_params = dict(params or {})
    samples = max(1, int(data_shape[0]))

    if method_id == "set_zero_time" and "time_step_s" not in runtime_params:
        total_time_ns = None
        if header_info:
            total_time_ns = header_info.get("total_time_ns")
        if total_time_ns and float(total_time_ns) > 0:
            runtime_params["time_step_s"] = float(total_time_ns) * 1e-9 / samples

    if method_id == "kirchhoff_migration":
        traces = max(1, int(data_shape[1]))
        info = header_info or {}
        if "header_info" not in runtime_params and info:
            runtime_params["header_info"] = clone_header_info(info)
        if "trace_metadata" not in runtime_params and trace_metadata:
            runtime_params["trace_metadata"] = clone_trace_metadata(trace_metadata)
        if "time_window_ns" not in runtime_params:
            total_time_ns = info.get("total_time_ns")
            runtime_params["time_window_ns"] = (
                float(total_time_ns)
                if total_time_ns and float(total_time_ns) > 0
                else float(samples)
            )
        if "length_m" not in runtime_params:
            if (
                info.get("track_length_m") is not None
                and float(info.get("track_length_m", 0.0)) > 0
            ):
                runtime_params["length_m"] = float(info["track_length_m"])
            elif info.get("trace_interval_m") is not None:
                runtime_params["length_m"] = float(
                    info.get("trace_interval_m", 0.0)
                ) * max(traces - 1, 1)

    return runtime_params


def merge_result_header_info(
    header_info: dict[str, Any] | None,
    result_meta: dict[str, Any] | None,
    data_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Merge method-returned header updates into runtime header info."""
    merged = clone_header_info(header_info)
    updates = (result_meta or {}).get("header_info_updates")
    if isinstance(updates, dict):
        for key, value in updates.items():
            merged[key] = (
                np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            )
    if data_shape is not None:
        merged["a_scan_length"] = int(data_shape[0])
        merged["num_traces"] = int(data_shape[1])
    return merged


def clone_header_info(header_info: dict[str, Any] | None) -> dict[str, Any]:
    """Clone header info while preserving ndarray values."""
    if not header_info:
        return {}
    cloned: Dict[str, Any] = {}
    for key, value in header_info.items():
        cloned[key] = (
            np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        )
    return cloned


def clone_trace_metadata(
    trace_metadata: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Clone per-trace metadata arrays for runtime use."""
    if not trace_metadata:
        return {}
    return {key: np.array(value, copy=True) for key, value in trace_metadata.items()}


def _normalize_result(
    method_id: str, result: Any, warnings: list[dict[str, Any]] | None = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"method_id": method_id}
    data = result

    if isinstance(result, tuple):
        data = result[0]
        if len(result) > 1 and isinstance(result[1], dict):
            meta.update(result[1])

    runtime_warnings = merge_runtime_warnings(warnings, meta.get("runtime_warnings"))

    output = _ensure_2d_array(data)
    if not np.isfinite(output).all():
        finite = np.isfinite(output)
        fill_value = float(np.mean(output[finite])) if finite.any() else 0.0
        output = np.nan_to_num(
            output, nan=fill_value, posinf=fill_value, neginf=fill_value
        )
        runtime_warnings.append(
            build_runtime_warning(
                "data_sanitized",
                "输出结果包含 NaN/Inf，已使用均值填充。",
                method_id=method_id,
                fill_value=fill_value,
            )
        )

    if runtime_warnings:
        meta["runtime_warnings"] = runtime_warnings

    return output.astype(np.float32, copy=False), meta


def _ensure_2d_array(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ProcessingEngineError(f"Must pass 2-d input. shape={arr.shape}")
    if arr.size == 0:
        raise ProcessingEngineError("输入数据为空")
    return np.array(arr, copy=True)


def _run_legacy_adapter(
    method_id: str, data: np.ndarray, params: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if method_id == "compensatingGain":
        return _normalize_result(method_id, _apply_compensating_gain(data, **params))
    if method_id == "agcGain":
        output, warnings = _apply_agc_gain(data, **params)
        return _normalize_result(method_id, output, warnings=warnings)
    if method_id == "subtracting_average_2D":
        output, warnings = _apply_subtracting_average_2d(data, **params)
        return _normalize_result(method_id, output, warnings=warnings)
    if method_id == "running_average_2D":
        output, warnings = _apply_running_average_2d(data, **params)
        return _normalize_result(method_id, output, warnings=warnings)
    if method_id == "dewow":
        from PythonModule.dewow import method_dewow

        return _normalize_result(method_id, method_dewow(data, **params))
    if method_id == "set_zero_time":
        from PythonModule.set_zero_time import method_set_zero_time

        return _normalize_result(method_id, method_set_zero_time(data, **params))

    raise ProcessingEngineError(f"未实现的处理方法: {method_id}")


def _apply_compensating_gain(
    data: np.ndarray, gain_min: float = 1.0, gain_max: float = 6.0, **kwargs
) -> np.ndarray:
    gain_curve_db = np.linspace(float(gain_min), float(gain_max), data.shape[0])
    gain_curve = 10.0 ** (gain_curve_db / 20.0)
    return data * gain_curve[:, np.newaxis]


def _apply_agc_gain(
    data: np.ndarray, window: int = 11, **kwargs
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from scipy.ndimage import uniform_filter1d

    requested_window = int(window)
    window = max(1, min(requested_window, data.shape[0]))
    eps = 1e-8
    warnings = []
    if window != requested_window:
        warnings.append(
            build_runtime_warning(
                "parameter_clamped",
                "AGC 窗口超过采样长度，已自动截断。",
                method_id="agcGain",
                parameter="window",
                requested=requested_window,
                effective=window,
            )
        )
    if window >= data.shape[0]:
        energy = np.maximum(np.linalg.norm(data, axis=0, keepdims=True), eps)
        warnings.append(
            build_runtime_warning(
                "global_gain_fallback",
                "AGC 窗口覆盖全时窗，已退化为全局能量归一化。",
                method_id="agcGain",
                effective_window=window,
            )
        )
    else:
        energy = np.sqrt(
            np.maximum(
                uniform_filter1d(data**2, size=window, axis=0, mode="nearest"),
                eps**2,
            )
        )
    return np.divide(data, energy), warnings


def _apply_subtracting_average_2d(
    data: np.ndarray, ntraces: int = 501, **kwargs
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from scipy.ndimage import uniform_filter1d

    requested_ntraces = int(ntraces)
    ntraces = max(1, requested_ntraces)
    warnings = []
    if ntraces >= data.shape[1]:
        background = np.mean(data, axis=1, keepdims=True)
        warnings.append(
            build_runtime_warning(
                "global_background_fallback",
                "背景窗口覆盖全部道数，已退化为全局平均背景。",
                method_id="subtracting_average_2D",
                parameter="ntraces",
                requested=requested_ntraces,
                effective=data.shape[1],
            )
        )
    else:
        background = uniform_filter1d(data, size=ntraces, axis=1, mode="nearest")
    return data - background, warnings


def _apply_running_average_2d(
    data: np.ndarray, ntraces: int = 9, **kwargs
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from scipy.ndimage import uniform_filter1d

    requested_ntraces = int(ntraces)
    ntraces = max(1, int(ntraces))
    warnings = []
    if ntraces <= 1:
        warnings.append(
            build_runtime_warning(
                "noop_window",
                "尖锐杂波抑制窗口为 1，输出等于输入。",
                method_id="running_average_2D",
                parameter="ntraces",
                requested=requested_ntraces,
            )
        )
        return np.array(data, copy=True), warnings
    if ntraces >= data.shape[1]:
        warnings.append(
            build_runtime_warning(
                "window_clamped",
                "尖锐杂波抑制窗口超过道数，已截断为当前道数。",
                method_id="running_average_2D",
                parameter="ntraces",
                requested=requested_ntraces,
                effective=data.shape[1],
            )
        )
        ntraces = data.shape[1]
    return uniform_filter1d(data, size=ntraces, axis=1, mode="nearest"), warnings
