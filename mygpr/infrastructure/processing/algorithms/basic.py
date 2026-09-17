"""Native baseline correction, gain and trace-domain algorithms."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter

from mygpr.infrastructure.processing.algorithms.common import (
    apply_time_selection,
    as_float,
    as_int,
    ensure_matrix,
    gprpy_dewow,
    gprpy_remove_mean_trace,
    local_l2_energy,
    normalize_output,
    resolve_time_selection,
    warning,
)

AGC_EPS = 1.0e-8
AGC_RMS_FLOOR_RATIO = 0.06
AGC_RMS_FLOOR_QUANTILE = 0.35


def method_compensating_gain(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    gain_min = as_float(params.get("gain_min"), 1.0)
    gain_max = as_float(params.get("gain_max"), 6.0)
    curve_db = np.linspace(gain_min, gain_max, arr.shape[0])
    result = arr * (10.0 ** (curve_db / 20.0))[:, None]
    meta = {"method": "compensatingGain", "gain_min": gain_min, "gain_max": gain_max}
    return normalize_output("compensatingGain", result, meta, warnings)


def method_dewow_native(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    raw_window = params.get("window", 23)
    try:
        numeric_window = float(raw_window)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dewow window must be numeric") from exc
    if not np.isfinite(numeric_window):
        raise ValueError("dewow window must be finite")
    requested = int(round(numeric_window))
    window = max(1, requested)
    if requested < 1 or requested > arr.shape[0]:
        warnings.append(
            warning("parameter_clamped", "去漂移窗口已按采样长度约束。", "dewow", requested=requested)
        )
    result = gprpy_dewow(arr, window)
    return normalize_output("dewow", result, {"method": "dewow", "window": window}, warnings)


def method_zero_time(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    new_zero = as_float(params.get("new_zero_time"), 5.0)
    if not np.isfinite(new_zero) or new_zero <= 0.0:
        meta = {
            "method": "set_zero_time",
            "new_zero_time": new_zero,
            "shift_samples": 0,
            "time_step_s": 0.0,
        }
        return normalize_output("set_zero_time", arr, meta, warnings)
    step_s = as_float(params.get("time_step_s"), 0.0)
    if step_s <= 0.0:
        header = dict(params.get("_header_info") or params.get("header_info") or {})
        total_time_ns = as_float(header.get("total_time_ns"), 0.0)
        if total_time_ns <= 0.0:
            total_time_ns = as_float(header.get("time_window_ns"), 0.0)
        if total_time_ns > 0.0:
            step_s = total_time_ns * 1.0e-9 / max(1, arr.shape[0])
    if step_s <= 0.0:
        raise ValueError(
            "set_zero_time 缺少时间基准：无法把 new_zero_time 映射到采样点。"
            "请提供 time_step_s 参数，或传入含 total_time_ns/time_window_ns 的 "
            "header_info（旧版按 48ns 采样间隔猜测步长的静默回退已移除）。"
        )
    shift = int(max(0.0, new_zero) / max(step_s * 1.0e9, 1.0e-12))
    shift = max(0, min(shift, arr.shape[0] - 1))
    result = np.zeros(arr.shape, dtype=np.float32)
    if shift == 0:
        result[:] = arr
    else:
        result[:-shift] = arr[shift:]
    meta = {"method": "set_zero_time", "new_zero_time": new_zero, "shift_samples": shift, "time_step_s": step_s}
    return normalize_output("set_zero_time", result, meta, warnings)


def _agc_floor(arr: np.ndarray, local_energy: np.ndarray) -> float:
    finite = local_energy[np.isfinite(local_energy) & (local_energy > 0.0)]
    global_rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0
    robust = float(np.quantile(finite, AGC_RMS_FLOOR_QUANTILE)) if finite.size else global_rms
    return max(AGC_EPS, global_rms * AGC_RMS_FLOOR_RATIO, robust * AGC_RMS_FLOOR_RATIO)


def method_agc(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    requested = as_int(params.get("window"), 11)
    window = max(1, requested)
    if requested < 1 or requested > arr.shape[0]:
        warnings.append(warning("parameter_clamped", "AGC 窗口已按采样长度约束。", "agcGain", requested=requested))
    energy = local_l2_energy(arr, window, eps=AGC_EPS)
    if bool(params.get("_low_energy_guard", False)):
        floor = _agc_floor(arr, energy)
        fraction = float(np.mean(energy < floor)) if energy.size else 0.0
        energy = np.maximum(energy, floor)
        if fraction >= 0.05:
            warnings.append(
                warning("agc_low_energy_gain_guard", "AGC 已限制低能量区域增益。", "agcGain", low_energy_fraction=fraction, energy_floor=floor)
            )
    result = np.divide(arr, energy)
    return normalize_output("agcGain", result, {"method": "agcGain", "window": window}, warnings)


def method_sec_gain_native(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    gain_min = as_float(params.get("gain_min"), 1.0)
    gain_max = as_float(params.get("gain_max"), 6.0)
    power = max(as_float(params.get("power"), 1.0), 1.0e-6)
    t = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float64) ** power
    curve = (gain_min + (gain_max - gain_min) * t).astype(np.float32)
    result = arr * curve[:, None]
    meta = {"method": "sec_gain", "gain_min": gain_min, "gain_max": gain_max, "power": power, "gain_curve": curve}
    return normalize_output("sec_gain", result, meta, warnings)


def method_remove_background(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    requested = as_int(params.get("ntraces"), 501)
    ntraces = max(1, requested)
    if ntraces >= arr.shape[1]:
        warnings.append(
            warning("global_background_fallback", "背景窗口覆盖全部道数，已使用全局平均背景。", "subtracting_average_2D", requested=requested, effective=arr.shape[1])
        )
    processed = gprpy_remove_mean_trace(arr, ntraces)
    selection = resolve_time_selection(arr.shape, params)
    result = apply_time_selection(arr, processed, selection, as_int(params.get("edge_taper_samples"), 0))
    meta = {"method": "subtracting_average_2D", "ntraces": ntraces, "time_start_idx": selection.start, "time_end_idx": selection.end, "time_range_source": selection.source}
    return normalize_output("subtracting_average_2D", result, meta, warnings)


def method_running_average(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    requested = as_int(params.get("ntraces"), 9)
    window = max(1, requested)
    if window <= 1:
        warnings.append(warning("noop_window", "尖锐杂波抑制窗口为 1，输出等于输入。", "running_average_2D"))
        result = arr.copy()
    else:
        if window >= arr.shape[1]:
            warnings.append(warning("window_clamped", "窗口超过道数，已截断。", "running_average_2D", requested=requested, effective=arr.shape[1]))
            window = arr.shape[1]
        result = uniform_filter1d(arr, size=window, axis=1, mode="nearest")
    return normalize_output("running_average_2D", result, {"method": "running_average_2D", "ntraces": window}, warnings)


def method_sliding_background(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    window = max(1, as_int(params.get("window_size"), 10))
    axis = as_int(params.get("axis"), 1)
    if axis not in (0, 1):
        raise ValueError("sliding_average axis must be 0 or 1")
    background = uniform_filter1d(arr, size=window, axis=axis, mode="nearest")
    return normalize_output("sliding_avg", arr - background, {"method": "sliding_avg", "window_size": window, "axis": axis}, warnings)


def _odd_window(value: Any, upper: int, default: int) -> int:
    window = min(max(1, as_int(value, default)), max(1, int(upper)))
    return max(1, window - 1) if window % 2 == 0 else window


def method_trace_median(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    requested = as_int(params.get("window_traces"), 5)
    window = _odd_window(requested, arr.shape[1], 5)
    if window < 3 or arr.shape[1] < 3:
        warnings.append(warning("trace_median_window_too_small", "道数或窗口过小，滤波已跳过。", "trace_median_filter"))
        result = arr.copy()
    else:
        result = median_filter(arr, size=(1, window), mode="nearest")
        if bool(params.get("preserve_mean", False)):
            result += float(np.mean(arr)) - float(np.mean(result))
    meta = {"method": "trace_median_filter", "requested_window_traces": requested, "effective_window_traces": window, "preserve_mean": bool(params.get("preserve_mean", False))}
    return normalize_output("trace_median_filter", result, meta, warnings)


def method_trace_savgol(data: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    window = _odd_window(params.get("window_traces"), arr.shape[1], 7)
    order = max(0, as_int(params.get("polyorder"), 2))
    if order >= window:
        order = max(0, window - 1)
        warnings.append(warning("trace_savgol_polyorder_adjusted", "polyorder 已按窗口调整。", "trace_savgol_filter", effective=order))
    mode = str(params.get("mode") or "interp").strip().lower()
    if mode not in {"interp", "nearest", "mirror", "constant", "wrap"}:
        mode = "interp"
        warnings.append(warning("trace_savgol_mode_adjusted", "不支持的 mode，已改用 interp。", "trace_savgol_filter"))
    if window < 3 or arr.shape[1] < 3 or order < 1:
        result = arr.copy()
        warnings.append(warning("trace_savgol_window_too_small", "道数、窗口或阶数不足，滤波已跳过。", "trace_savgol_filter"))
    else:
        result = savgol_filter(arr, window_length=window, polyorder=order, deriv=0, axis=1, mode=mode)
    meta = {"method": "trace_savgol_filter", "effective_window_traces": window, "polyorder": order, "derivative": 0, "mode": mode}
    return normalize_output("trace_savgol_filter", result, meta, warnings)
