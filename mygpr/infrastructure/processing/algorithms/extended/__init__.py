"""Native implementations for the remaining public processing methods."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .amplitude import method_amplitude_scale
from .ccbs import method_ccbs
from .deconvolution import method_mixed_phase_deconvolution
from .depth import method_time_to_depth
from .energy_gain import method_energy_decay_gain
from .hilbert import method_hilbert_envelope
from .inverse_q import method_inverse_q
from .median_background import method_median_background_2d
from .time_cut import method_time_cut
from .trace_cut import method_trace_cut
from .trace_qc import method_trace_qc
from .wavelet import method_wavelet_2d

ExtendedFunction = Callable[..., tuple[np.ndarray, dict[str, Any]]]


def _runtime_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(params or {})
    header = dict(runtime.pop("_header_info", runtime.pop("header_info", {})) or {})
    trace = dict(runtime.pop("_trace_metadata", runtime.pop("trace_metadata", {})) or {})
    runtime.pop("_execution_context", None)
    runtime.setdefault("header_info", header)
    runtime.setdefault("trace_metadata", trace)
    if "time_window_ns" not in runtime:
        total = header.get("total_time_ns") or header.get("time_window_ns")
        if total not in (None, "", 0, 0.0):
            runtime["time_window_ns"] = total
    return runtime


def _execute(function: ExtendedFunction, data: Any, params: dict[str, Any]):
    return function(data, **_runtime_kwargs(params))


def native_time_cut(data: Any, params: dict[str, Any]):
    return _execute(method_time_cut, data, params)


def native_trace_cut(data: Any, params: dict[str, Any]):
    return _execute(method_trace_cut, data, params)


def native_trace_qc(data: Any, params: dict[str, Any]):
    return _execute(method_trace_qc, data, params)



def native_energy_decay_gain(data: Any, params: dict[str, Any]):
    return _execute(method_energy_decay_gain, data, params)


def native_amplitude_scale(data: Any, params: dict[str, Any]):
    return _execute(method_amplitude_scale, data, params)


def native_median_background(data: Any, params: dict[str, Any]):
    return _execute(method_median_background_2d, data, params)


def native_wavelet_2d(data: Any, params: dict[str, Any]):
    return _execute(method_wavelet_2d, data, params)


def native_hilbert_envelope(data: Any, params: dict[str, Any]):
    return _execute(method_hilbert_envelope, data, params)


def native_ccbs(data: Any, params: dict[str, Any]):
    return _execute(method_ccbs, data, params)


def native_time_to_depth(data: Any, params: dict[str, Any]):
    for key, default in (("dt", 0.1), ("v", 0.10), ("dz", 0.02)):
        value = float(params.get(key, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"时间-深度转换 {key} 必须为正数")
    output, metadata = _execute(method_time_to_depth, data, params)
    output = np.asarray(output, dtype=np.float32)
    resolved = dict(metadata)
    header = params.get("_header_info", params.get("header_info", {})) or {}
    source_axis = header.get("time_axis_ns")
    start = None
    if source_axis is not None:
        source_axis = np.asarray(source_axis, dtype=np.float64)
        if source_axis.shape != (np.shape(data)[0],) or not np.all(np.isfinite(source_axis)):
            raise ValueError("时间-深度转换输入时间轴无效")
        start = float(source_axis[0])
    elif header.get("time_cut_offset_ns") is not None:
        start = float(header["time_cut_offset_ns"])
    if start is not None and not np.isfinite(start):
        raise ValueError("时间-深度转换输入时间起点无效")
    velocity = float(params.get("v", 0.10))
    relative_depth = np.linspace(0, float(resolved.get("z_max", 0.0)), output.shape[0])
    output_time = None if start is None or velocity <= 0 else start + relative_depth * 2.0 / velocity
    output_depth = None if output_time is None else output_time * velocity / 2.0
    resolved.setdefault(
        "header_info_updates",
        {
            "a_scan_length": int(output.shape[0]),
            "num_traces": int(output.shape[1]),
            "is_depth": True,
            "is_elevation": False,
            "depth_max_m": float(output_depth[-1]) if output_depth is not None else float(resolved.get("z_max", 0.0)),
            "total_time_ns": 0.0,
            "time_axis_ns": output_time,
            "depth_axis_m": output_depth,
            "time_step_s": ((float(output_time[-1] - output_time[0]) / (len(output_time) - 1)) * 1e-9
                            if output_time is not None and len(output_time) > 1 else None),
        },
    )
    return output, resolved


def native_mixed_phase_deconvolution(data: Any, params: dict[str, Any]):
    return _execute(method_mixed_phase_deconvolution, data, params)


def native_inverse_q(data: Any, params: dict[str, Any]):
    return _execute(method_inverse_q, data, params)


__all__ = [name for name in globals() if name.startswith("native_")]
