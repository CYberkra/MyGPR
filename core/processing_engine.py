#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified ndarray processing engine for GUI, workflow, and batch execution."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.methods_registry import PROCESSING_METHODS
from core.background_time_range import (
    apply_time_range_to_result,
    resolve_time_range_selection,
)
from core.gprpy_compat import (
    apply_gprpy_agc_gain,
    apply_gprpy_rem_mean_trace,
    gprpy_local_window_l2_energy,
)
from mygpr.domain.processing.warnings import build_runtime_warning, merge_runtime_warnings
from mygpr.domain.common.scalars import to_float, to_int
from mygpr.domain.common.errors import MyGPRError


class ProcessingEngineError(MyGPRError):
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
    runtime_warnings = _build_input_sanitized_warnings(method_id, input_data)
    if runtime_warnings:
        finite = np.isfinite(input_data)
        fill_value = float(np.mean(input_data[finite])) if finite.any() else 0.0
        input_data = np.nan_to_num(
            input_data, nan=fill_value, posinf=fill_value, neginf=fill_value
        )
    runtime_params = _filter_runtime_params(method_id, params or {})
    if cancel_checker is not None:
        runtime_params.setdefault("cancel_checker", cancel_checker)

    func = method_info.get("func")
    if callable(func):
        result = func(np.array(input_data, copy=True), **runtime_params)
        return _normalize_result(method_id, result, warnings=runtime_warnings)

    return _run_legacy_adapter(
        method_id, input_data, runtime_params, warnings=runtime_warnings
    )


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

    if (
        method_id
        in {
            "set_zero_time",
            "time_cut",
            "agcGain",
            "frequency_filter_1d",
            "subtracting_average_2D",
            "median_background_2D",
        }
        and "time_step_s" not in runtime_params
    ):
        total_time_ns = None
        if header_info:
            total_time_ns = header_info.get("total_time_ns")
        total_time_value = to_float(total_time_ns, default=0.0)
        if total_time_value > 0:
            time_step_s = total_time_value * 1e-9 / samples
            runtime_params["time_step_s"] = time_step_s
            if method_id == "frequency_filter_1d":
                runtime_params.setdefault("sample_rate_hz", 1.0 / time_step_s)
            if method_id in {"subtracting_average_2D", "median_background_2D"}:
                runtime_params.setdefault("time_window_ns", total_time_value)

    if (
        method_id in {"subtracting_average_2D", "median_background_2D"}
        and "time_window_ns" not in runtime_params
        and "time_step_s" in runtime_params
    ):
        time_step_s = to_float(runtime_params["time_step_s"], default=0.0)
        if time_step_s > 0.0:
            runtime_params["time_window_ns"] = time_step_s * 1.0e9 * samples

    needs_motion_runtime = _requires_motion_runtime_context(method_id)

    if needs_motion_runtime or method_id in {
        "kirchhoff_migration",
        "trace_qc",
        "equidistant_trace_resample",
    }:
        _inject_runtime_metadata_context(
            runtime_params,
            header_info=header_info,
            trace_metadata=trace_metadata,
            samples=samples,
        )

    if method_id == "kirchhoff_migration":
        traces = max(1, int(data_shape[1]))
        info = header_info or {}
        # UI 会把 schema 默认值（0.0）一并传入，仅判 "不存在" 会导致
        # 真实测线长度永远无法注入，Kirchhoff 偏移在 GUI 中必然失败。
        if to_float(runtime_params.get("length_m"), default=0.0) <= 0.0:
            track_length_m = to_float(info.get("track_length_m"), default=0.0)
            if track_length_m > 0:
                runtime_params["length_m"] = track_length_m
            elif info.get("trace_interval_m") is not None:
                runtime_params["length_m"] = to_float(
                    info.get("trace_interval_m"),
                    default=0.0,
                ) * max(traces - 1, 1)

    return runtime_params


def _filter_runtime_params(method_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Drop hidden runtime flags unless the method explicitly supports them."""
    runtime_params: dict[str, Any] = {}
    for key, value in params.items():
        key_text = str(key)
        if key_text.startswith("_"):
            if method_id == "agcGain" and key_text == "_low_energy_guard":
                # agcGain is the only method that currently consumes _low_energy_guard
                runtime_params[key_text] = value
            continue
        runtime_params[key] = value
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


def merge_result_trace_metadata(
    trace_metadata: dict[str, np.ndarray] | None,
    result_meta: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Merge method-returned trace metadata into runtime trace metadata."""
    trace_metadata_out = (result_meta or {}).get("trace_metadata_out")
    if isinstance(trace_metadata_out, dict):
        return clone_trace_metadata(trace_metadata_out)

    merged = clone_trace_metadata(trace_metadata)
    updates = (result_meta or {}).get("trace_metadata_updates")
    if isinstance(updates, dict):
        for key, value in updates.items():
            merged[key] = np.array(value, copy=True)
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


def _requires_motion_runtime_context(method_id: str) -> bool:
    """Whether a method should receive cloned motion-runtime metadata.

    Runtime metadata is an execution capability, not an auto-tune classification.
    Vibration suppression is scored in the artifact family but can still consume
    IMU angular-rate/trajectory guidance, so it must receive the same metadata
    context as the other motion methods.
    """
    motion_methods = {
        "motion_compensation_height",
        "motion_compensation_speed",
        "trajectory_smoothing",
        "motion_compensation_attitude",
        "motion_compensation_vibration",
        "motion_compensation_v2",
    }
    if str(method_id) in motion_methods:
        return True
    method_info = PROCESSING_METHODS.get(method_id, {})
    stage = method_info.get("auto_tune_stage") or method_info.get("auto_tune_family")
    return str(stage or "") == "motion_comp"


def _inject_runtime_metadata_context(
    runtime_params: dict[str, Any],
    *,
    header_info: dict[str, Any] | None,
    trace_metadata: dict[str, np.ndarray] | None,
    samples: int,
) -> None:
    """Inject cloned runtime metadata for methods that need motion context."""
    info = header_info or {}
    if "header_info" not in runtime_params and info:
        runtime_params["header_info"] = clone_header_info(info)
    if "trace_metadata" not in runtime_params and trace_metadata:
        runtime_params["trace_metadata"] = clone_trace_metadata(trace_metadata)
    # UI 会传入 schema 默认值 0.0（占位），视为未设置时用真实时窗覆盖，
    # 否则 Kirchhoff/RTM 等需要真实时窗的方法在 GUI 中必失败。
    if to_float(runtime_params.get("time_window_ns"), default=0.0) <= 0.0:
        total_time_ns = info.get("total_time_ns")
        total_time_value = to_float(total_time_ns, default=0.0)
        runtime_params["time_window_ns"] = (
            total_time_value if total_time_value > 0.0 else float(samples)
        )


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


def _build_input_sanitized_warnings(
    method_id: str, input_data: np.ndarray
) -> list[dict[str, Any]]:
    """Build warnings for non-finite input before a method can hide them."""
    if np.isfinite(input_data).all():
        return []
    finite = np.isfinite(input_data)
    fill_value = float(np.mean(input_data[finite])) if finite.any() else 0.0
    return [
        build_runtime_warning(
            "data_sanitized",
            "输入数据包含 NaN/Inf，已使用均值填充后再处理。",
            method_id=method_id,
            fill_value=fill_value,
            stage="input",
        )
    ]


def _ensure_2d_array(data: Any) -> np.ndarray:
    arr = np.array(data, dtype=np.float64, copy=True)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ProcessingEngineError(f"Must pass 2-d input. shape={arr.shape}")
    if arr.size == 0:
        raise ProcessingEngineError("输入数据为空")
    return arr


def _run_legacy_adapter(
    method_id: str,
    data: np.ndarray,
    params: Dict[str, Any],
    warnings: list[dict[str, Any]] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if method_id == "compensatingGain":
        return _normalize_result(
            method_id, _apply_compensating_gain(data, **params), warnings=warnings
        )
    if method_id == "agcGain":
        output, method_warnings = _apply_agc_gain(data, **params)
        return _normalize_result(
            method_id,
            (
                output,
                {
                    "window": to_int(params.get("window", 11), default=11),
                    "method": "agcGain",
                },
            ),
            warnings=merge_runtime_warnings(warnings, method_warnings),
        )
    if method_id == "subtracting_average_2D":
        output, method_warnings = _apply_subtracting_average_2d(data, **params)
        selection = resolve_time_range_selection(
            data.shape,
            time_start_idx=params.get("time_start_idx"),
            time_end_idx=params.get("time_end_idx"),
            time_start_ns=params.get("time_start_ns"),
            time_end_ns=params.get("time_end_ns"),
            time_window_ns=params.get("time_window_ns"),
        )
        return _normalize_result(
            method_id,
            (
                output,
                {
                    "ntraces": int(params.get("ntraces", 501)),
                    "time_start_idx": int(selection.start_idx),
                    "time_end_idx": int(selection.end_idx),
                    "time_range_source": selection.source,
                    "method": "subtracting_average_2D",
                },
            ),
            warnings=merge_runtime_warnings(warnings, method_warnings),
        )
    if method_id == "running_average_2D":
        output, method_warnings = _apply_running_average_2d(data, **params)
        return _normalize_result(
            method_id,
            output,
            warnings=merge_runtime_warnings(warnings, method_warnings),
        )
    if method_id == "dewow":
        from PythonModule.dewow import method_dewow

        return _normalize_result(
            method_id, method_dewow(data, **params), warnings=warnings
        )
    if method_id == "set_zero_time":
        from PythonModule.set_zero_time import method_set_zero_time

        return _normalize_result(
            method_id, method_set_zero_time(data, **params), warnings=warnings
        )

    raise ProcessingEngineError(f"不支持的处理方法: {method_id}")


def _apply_compensating_gain(
    data: np.ndarray, gain_min: float = 1.0, gain_max: float = 6.0, **kwargs
) -> np.ndarray:
    min_db = to_float(gain_min, default=1.0)
    max_db = to_float(gain_max, default=6.0)
    gain_curve_db = np.linspace(min_db, max_db, data.shape[0])
    gain_curve = 10.0 ** (gain_curve_db / 20.0)
    return data * gain_curve[:, np.newaxis]


AGC_EPS = 1.0e-8
AGC_RMS_FLOOR_RATIO = 0.06
AGC_RMS_FLOOR_QUANTILE = 0.35
AGC_LOW_ENERGY_WARNING_FRACTION = 0.05
AGC_MIN_WINDOW_NS = 0.5


def _apply_agc_gain(
    data: np.ndarray,
    window: int = 11,
    _low_energy_guard: bool = False,
    **kwargs,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    requested_window = to_int(window, default=11)
    window = max(1, requested_window)
    warnings = []
    if requested_window < 1 or requested_window > data.shape[0]:
        warnings.append(
            build_runtime_warning(
                "parameter_clamped",
                "AGC 窗口超过采样长度，已自动截断。",
                method_id="agcGain",
                parameter="window",
                requested=requested_window,
                effective=max(1, min(requested_window, data.shape[0])),
            )
        )
    use_low_energy_guard = bool(_low_energy_guard)
    time_step_s = kwargs.get("time_step_s")
    if use_low_energy_guard and time_step_s is not None:
        step_s = to_float(time_step_s, default=0.0)
        window_ns = step_s * 1.0e9 * float(window) if step_s > 0.0 else None
        if window_ns is not None and 0.0 < window_ns < AGC_MIN_WINDOW_NS:
            warnings.append(
                build_runtime_warning(
                    "agc_window_too_short",
                    "AGC 窗口时间宽度过短，可能放大波形周期和低能量噪声。",
                    method_id="agcGain",
                    parameter="window",
                    effective=window,
                    window_ns=window_ns,
                    recommended_min_window_ns=AGC_MIN_WINDOW_NS,
                )
            )
    if window > data.shape[0]:
        energy = np.maximum(np.linalg.norm(data, axis=0, keepdims=True), AGC_EPS)
        warnings.append(
            build_runtime_warning(
                "global_gain_fallback",
                "AGC 窗口覆盖全时窗，已退化为全局能量归一化。",
                method_id="agcGain",
                effective_window=window,
            )
        )
    else:
        if use_low_energy_guard:
            local_energy = _gprpy_local_window_energy(data, window)
            energy_floor = _agc_energy_floor(data, local_energy)
            floor_mask = local_energy < energy_floor
            floor_fraction = float(np.mean(floor_mask)) if floor_mask.size else 0.0
            if floor_fraction >= AGC_LOW_ENERGY_WARNING_FRACTION:
                warnings.append(
                    build_runtime_warning(
                        "agc_low_energy_gain_guard",
                        "AGC 检测到低能量区域，已限制局部增益以减少噪声/空白区伪影。",
                        method_id="agcGain",
                        parameter="window",
                        effective=window,
                        low_energy_fraction=floor_fraction,
                        energy_floor=float(energy_floor),
                        max_gain=float(1.0 / max(energy_floor, AGC_EPS)),
                    )
                )
            energy = np.maximum(local_energy, energy_floor)
        else:
            return apply_gprpy_agc_gain(data, window), warnings
    return np.divide(data, energy), warnings


def _agc_energy_floor(data: np.ndarray, local_rms: np.ndarray) -> float:
    arr = np.asarray(data, dtype=np.float64)
    rms = np.asarray(local_rms, dtype=np.float64)
    finite_rms = rms[np.isfinite(rms) & (rms > 0.0)]
    global_rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0
    robust_rms = (
        float(np.quantile(finite_rms, AGC_RMS_FLOOR_QUANTILE))
        if finite_rms.size
        else global_rms
    )
    return float(
        max(
            AGC_EPS,
            global_rms * AGC_RMS_FLOOR_RATIO,
            robust_rms * AGC_RMS_FLOOR_RATIO,
        )
    )


def _apply_subtracting_average_2d(
    data: np.ndarray,
    ntraces: int = 501,
    time_start_idx: Any = None,
    time_end_idx: Any = None,
    time_start_ns: Any = None,
    time_end_ns: Any = None,
    time_window_ns: Any = None,
    edge_taper_samples: int = 0,
    **kwargs,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    requested_ntraces = int(ntraces)
    ntraces = max(1, requested_ntraces)
    warnings = []
    if ntraces >= data.shape[1]:
        background = np.mean(data, axis=1, keepdims=True)
        full_result = data - background
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
        full_result = apply_gprpy_rem_mean_trace(data, ntraces)

    selection = resolve_time_range_selection(
        data.shape,
        time_start_idx=time_start_idx,
        time_end_idx=time_end_idx,
        time_start_ns=time_start_ns,
        time_end_ns=time_end_ns,
        time_window_ns=time_window_ns,
    )
    result = apply_time_range_to_result(
        data,
        full_result,
        selection,
        edge_taper_samples=edge_taper_samples,
    )
    return result, warnings


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


def _gprpy_local_window_energy(data: np.ndarray, window: int) -> np.ndarray:
    """Return the GPRPy-style L2 norm over each moving window."""
    return gprpy_local_window_l2_energy(data, window, eps=AGC_EPS)
