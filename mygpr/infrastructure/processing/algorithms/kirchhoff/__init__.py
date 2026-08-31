"""Native Kirchhoff migration facade and parameter mapping."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mygpr.domain.processing.models import ResourceEstimate
from mygpr.infrastructure.processing.algorithms.kirchhoff.cpu import _run_kirchhoff_cpu
from mygpr.infrastructure.processing.algorithms.kirchhoff.gpu import _run_kirchhoff_gpu
from mygpr.infrastructure.processing.algorithms.kirchhoff.shared import _to_float64_2d
from mygpr.infrastructure.processing.gpu_policy import is_gpu_resource_error, resolve_gpu_backend

def load_cagpr_kir_parameter_file(file_path: str | Path) -> dict[str, object]:
    """Parse CaGPR Kirchhoff parameter text file into current method params."""
    key_map = {
        "freq": "freq",
        "M-depth": "depth",
        "v": "v",
        "weight": "weight",
        "num_cal": "num_cal",
        "len": "length_m",
        "T": "time_window_ns",
        "topo_cor": "topo_cor",
        "hei_cor": "hei_cor",
    }
    int_keys = {"num_cal", "topo_cor", "hei_cor"}
    path = Path(file_path)
    parsed: dict[str, object] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            src_key, raw_value = parts
            dst_key = key_map.get(src_key)
            if dst_key is None:
                continue
            value = raw_value.strip()
            if dst_key in int_keys:
                parsed[dst_key] = int(float(value))
            else:
                parsed[dst_key] = float(value)
    return parsed


def _release_cupy_memory_pools() -> None:
    """Best-effort cleanup before a CPU retry after a failed GPU execution."""
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
        return


def _estimate_gpu_workspace_bytes(
    input_shape: tuple[int, int],
    *,
    freq: float,
    depth: float,
    length_m: float,
    time_window_ns: float,
) -> int:
    """Estimate a conservative Kirchhoff GPU peak before any device allocation."""
    c = 3.0e8
    dx = c / (60.0 * max(float(freq), 1.0))
    dt_s = dx / (2.0 * c)
    nt = max(1, int(np.ceil(float(time_window_ns) / (dt_s * 1.0e9))))
    nz = max(1, int(np.ceil(float(depth) / dx)))
    nx_model = max(1, 2 * nz)
    traces = max(1, int(np.ceil(float(length_m) / dx)))
    input_elements = max(1, int(input_shape[0]) * int(input_shape[1]))
    model_elements = nx_model * nz
    resized_elements = nt * traces
    # Input, resized data, travel-time/model, shot/output, and CuPy FFT/TV
    # scratch are represented with a safety factor rather than optimistic reuse.
    return int(8 * (input_elements + 3 * resized_elements + 8 * model_elements) * 1.35)


def method_kirchhoff_migration(
    data,
    freq=5.0e7,
    depth=40.0,
    v=0.10,
    alpha=1.0,
    weight=0.5,
    num_cal=1,
    topo_cor=0,
    hei_cor=0,
    length_m=None,
    time_window_ns=None,
    backend="auto",
    **kwargs,
):
    """Run the CaGPR-style Kirchhoff imaging main chain in memory.

    Args:
        data: Input data array
        freq: Center frequency in Hz
        depth: Imaging depth in meters
        v: Wave velocity in m/ns
        alpha: Power gain factor
        weight: TV denoising weight
        num_cal: Parallel chunk size
        topo_cor: Topography correction (0=off, 1=post, 2=pre)
        hei_cor: Height correction (0=off, 1=post, 2=pre)
        length_m: Track length in meters (auto-detected if None)
        time_window_ns: Time window in ns (auto-detected if None)
        backend: "auto", "gpu", or "cpu". "auto" tries GPU first, falls back to CPU.
        **kwargs: Additional options including cancel_checker, header_info, trace_metadata

    Returns:
        tuple: (result_array, metadata_dict)
    """
    cancel_checker = kwargs.get("cancel_checker")
    header_info = kwargs.get("header_info") or {}
    trace_metadata = kwargs.get("trace_metadata") or {}
    arr = _to_float64_2d(data)

    freq = float(freq)
    depth = float(depth)
    velocity = float(v)
    alpha = float(alpha)
    weight = float(weight)
    num_cal = max(1, int(num_cal))
    topo_cor = int(topo_cor)
    hei_cor = int(hei_cor)
    if freq <= 0 or depth <= 0 or velocity <= 0:
        raise ValueError("freq、depth、v 必须为正数")
    if topo_cor not in (0, 1, 2) or hei_cor not in (0, 1, 2):
        raise ValueError("当前 Kirchhoff 仅支持 topo_cor/hei_cor 取值 0、1 或 2")

    if length_m is None:
        length_m = float(max(arr.shape[1] - 1, 1))
    if time_window_ns is None:
        time_window_ns = float(max(arr.shape[0], 1))

    length_m = float(length_m)
    time_window_ns = float(time_window_ns)
    if length_m <= 0 or time_window_ns <= 0:
        raise ValueError("length_m 和 time_window_ns 必须为正数")

    # Resolve backend (GPU or CPU) with a conservative workspace budget.
    required_gpu_bytes = _estimate_gpu_workspace_bytes(
        arr.shape, freq=freq, depth=depth, length_m=length_m, time_window_ns=time_window_ns
    )
    selection = resolve_gpu_backend(backend, required_bytes=required_gpu_bytes)
    resolved_backend = selection.backend
    fallback_reason = selection.fallback_reason or None

    # If GPU fails during processing, we'll retry with CPU
    execution_backend = resolved_backend
    gpu_error = None

    try:
        if resolved_backend == "gpu":
            result, metadata = _run_kirchhoff_gpu(
                arr,
                freq,
                depth,
                velocity,
                alpha,
                weight,
                num_cal,
                topo_cor,
                hei_cor,
                length_m,
                time_window_ns,
                cancel_checker,
                header_info,
                trace_metadata,
            )
            execution_backend = "gpu"
        else:
            result, metadata = _run_kirchhoff_cpu(
                arr,
                freq,
                depth,
                velocity,
                alpha,
                weight,
                num_cal,
                topo_cor,
                hei_cor,
                length_m,
                time_window_ns,
                cancel_checker,
                header_info,
                trace_metadata,
            )
            execution_backend = "cpu"

    except Exception as exc:
        cancelled = callable(cancel_checker) and bool(cancel_checker())
        explicit_gpu = str(backend).strip().lower() == "gpu"
        allow_runtime_fallback = bool(kwargs.get("allow_gpu_runtime_fallback", not explicit_gpu))
        if resolved_backend == "gpu" and not cancelled and allow_runtime_fallback and is_gpu_resource_error(exc):
            gpu_error = str(exc)
            execution_backend = "cpu"
            fallback_reason = f"GPU resource exhaustion: {gpu_error}"
            _release_cupy_memory_pools()
            result, metadata = _run_kirchhoff_cpu(
                arr,
                freq,
                depth,
                velocity,
                alpha,
                weight,
                num_cal,
                topo_cor,
                hei_cor,
                length_m,
                time_window_ns,
                cancel_checker,
                header_info,
                trace_metadata,
            )
        else:
            raise

    # Add backend info to metadata
    metadata["mapped_params"]["execution_backend"] = execution_backend
    metadata["mapped_params"]["requested_backend"] = backend
    metadata["mapped_params"]["gpu_required_bytes"] = int(required_gpu_bytes)
    metadata["mapped_params"]["gpu_device_name"] = selection.capability.device_name
    metadata["mapped_params"]["gpu_free_bytes"] = int(selection.capability.free_bytes)
    if fallback_reason:
        metadata["mapped_params"]["fallback_reason"] = fallback_reason
    if gpu_error:
        metadata["mapped_params"]["gpu_error"] = gpu_error

    return result, metadata



def estimate_kirchhoff_resources(
    shape: tuple[int, int],
    dtype: np.dtype | str,
    params: dict[str, Any],
    header_info: dict[str, Any],
) -> ResourceEstimate:
    """Estimate CPU/GPU workspaces from the actual imaging grid."""
    samples, traces = (max(1, int(shape[0])), max(1, int(shape[1])))
    freq = float(params.get("freq", 5.0e7))
    depth = float(params.get("depth", 40.0))
    length_m = float(
        params.get("length_m")
        or header_info.get("length_m")
        or header_info.get("track_length_m")
        or max(traces - 1, 1)
    )
    time_window_ns = float(
        params.get("time_window_ns")
        or header_info.get("total_time_ns")
        or header_info.get("time_window_ns")
        or samples
    )
    gpu_peak = _estimate_gpu_workspace_bytes(
        (samples, traces),
        freq=freq,
        depth=depth,
        length_m=length_m,
        time_window_ns=time_window_ns,
    )
    input_bytes = samples * traces * np.dtype(dtype).itemsize
    return ResourceEstimate(
        memory_bytes=max(int(input_bytes * 4), int(gpu_peak * 1.25)),
        temporary_disk_bytes=int(input_bytes * 2),
        relative_cost="very_high",
        supports_cancellation=True,
        supports_chunking=False,
        notes=(
            "native CaGPR-compatible Kirchhoff kernel",
            "loaded global execution with explicit imaging-grid budget",
            "GPU auto mode falls back to CPU when CUDA or usable memory is unavailable",
        ),
    )

def method_kirchhoff_migration_native(
    data: Any, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute Kirchhoff migration through the native processing contract."""
    runtime = dict(params or {})
    header_info = dict(runtime.pop("_header_info", runtime.pop("header_info", {})) or {})
    trace_metadata = dict(runtime.pop("_trace_metadata", runtime.pop("trace_metadata", {})) or {})
    runtime.pop("_execution_context", None)
    return method_kirchhoff_migration(
        data,
        header_info=header_info,
        trace_metadata=trace_metadata,
        **runtime,
    )


__all__ = [
    "load_cagpr_kir_parameter_file",
    "method_kirchhoff_migration",
    "method_kirchhoff_migration_native",
    "estimate_kirchhoff_resources",
]
