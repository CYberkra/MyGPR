"""Native implementations for the remaining public processing methods."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .amplitude import method_amplitude_scale
from .ccbs import method_ccbs
from .depth import method_time_to_depth
from .energy_gain import method_energy_decay_gain
from .hilbert import method_hilbert_envelope
from .median_background import method_median_background_2d
from .time_cut import method_time_cut
from .trace_qc import method_trace_qc
from .wavelet import method_wavelet_2d, method_wavelet_svd

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


def native_wavelet_svd(data: Any, params: dict[str, Any]):
    return _execute(method_wavelet_svd, data, params)


def native_hilbert_envelope(data: Any, params: dict[str, Any]):
    return _execute(method_hilbert_envelope, data, params)


def native_ccbs(data: Any, params: dict[str, Any]):
    return _execute(method_ccbs, data, params)


def native_time_to_depth(data: Any, params: dict[str, Any]):
    output, metadata = _execute(method_time_to_depth, data, params)
    output = np.asarray(output, dtype=np.float32)
    resolved = dict(metadata)
    resolved.setdefault(
        "header_info_updates",
        {
            "a_scan_length": int(output.shape[0]),
            "num_traces": int(output.shape[1]),
            "is_depth": True,
            "is_elevation": False,
            "depth_max_m": float(resolved.get("z_max", 0.0)),
            "total_time_ns": 0.0,
        },
    )
    return output, resolved


__all__ = [name for name in globals() if name.startswith("native_")]
