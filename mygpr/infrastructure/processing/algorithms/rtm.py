"""Experimental zero-offset exploding-reflector RTM baseline.

This is a scalar 2-D finite-difference reverse-time continuation for a B-scan
with colocated transmit/receive positions.  It is intentionally labelled as an
experimental baseline, not as a full shot-gather electromagnetic RTM solver.
The implementation is UI-independent, cancellable, resource-capped, and emits
an explicit modelling contract in its metadata.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from mygpr.domain.processing.models import ResourceEstimate
from mygpr.infrastructure.processing.algorithms.common import ensure_matrix, normalize_output, warning


class RTMResourceLimitError(RuntimeError):
    """Raised before allocation when the configured RTM budget is exceeded."""


def _cancel_checker(params: dict[str, Any]) -> Callable[[], bool]:
    checker = params.get("cancel_checker")
    if callable(checker):
        return checker
    context = params.get("_execution_context")
    if context is not None and hasattr(context, "is_cancelled"):
        return context.is_cancelled
    return lambda: False


def _header(params: dict[str, Any]) -> dict[str, Any]:
    return dict(params.get("_header_info") or params.get("header_info") or {})


def _positive_float(value: Any, default: float, name: str) -> float:
    resolved = default if value in (None, "") else float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return resolved


def _optional_positive_float(value: Any, default: float, name: str) -> float:
    if value in (None, "", 0, 0.0):
        return _positive_float(default, default, name)
    return _positive_float(value, default, name)


def _resolve_geometry(
    shape: tuple[int, int], params: dict[str, Any], header: dict[str, Any]
) -> dict[str, float | int]:
    samples, traces = (max(1, int(shape[0])), max(1, int(shape[1])))
    total_time_ns = _positive_float(
        params.get("time_window_ns")
        or header.get("total_time_ns")
        or header.get("time_window_ns"),
        float(samples),
        "time_window_ns",
    )
    dt_ns = _positive_float(params.get("dt_ns"), total_time_ns / samples, "dt_ns")
    length_m = _positive_float(
        params.get("length_m")
        or header.get("length_m")
        or header.get("track_length_m"),
        float(max(traces - 1, 1)),
        "length_m",
    )
    dx_m = _optional_positive_float(
        params.get("dx_m") or params.get("dx") or header.get("trace_interval_m"),
        length_m / max(traces - 1, 1),
        "dx_m",
    )
    velocity = _positive_float(params.get("v") or params.get("velocity_m_per_ns"), 0.10, "v")
    depth_default = velocity * total_time_ns * 0.5
    depth_m = _positive_float(params.get("depth_m") or params.get("depth"), depth_default, "depth_m")
    dz_m = _optional_positive_float(params.get("dz_m") or params.get("dz"), min(dx_m, depth_m), "dz_m")
    nz = max(2, int(math.floor(depth_m / dz_m)) + 1)
    # Exploding-reflector propagation is one-way, so half the physical velocity
    # maps the recorded two-way time to depth.
    propagation_velocity = velocity * 0.5
    cfl = propagation_velocity * dt_ns * math.sqrt(1.0 / dx_m**2 + 1.0 / dz_m**2)
    cfl_limit = float(np.clip(float(params.get("cfl_limit", 0.65)), 0.20, 0.70))
    substeps = max(1, int(math.ceil(cfl / cfl_limit)))
    return {
        "samples": samples,
        "traces": traces,
        "total_time_ns": total_time_ns,
        "dt_ns": dt_ns,
        "length_m": length_m,
        "dx_m": dx_m,
        "velocity_m_per_ns": velocity,
        "propagation_velocity_m_per_ns": propagation_velocity,
        "depth_m": depth_m,
        "dz_m": dz_m,
        "nz": nz,
        "cfl": cfl,
        "cfl_limit": cfl_limit,
        "substeps": substeps,
    }


def estimate_rtm_resources(
    shape: tuple[int, int], dtype: np.dtype | str, params: dict[str, Any], header_info: dict[str, Any]
) -> ResourceEstimate:
    geometry = _resolve_geometry(shape, params, header_info)
    nz = int(geometry["nz"])
    nx = int(geometry["traces"])
    samples = int(geometry["samples"])
    substeps = int(geometry["substeps"])
    grid_bytes = nz * nx * np.dtype(np.float32).itemsize
    input_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    updates = samples * substeps * nz * nx
    return ResourceEstimate(
        memory_bytes=int(input_bytes + grid_bytes * 7),
        temporary_disk_bytes=int(grid_bytes + input_bytes),
        relative_cost="very_high",
        supports_cancellation=True,
        supports_chunking=False,
        notes=(
            "experimental scalar zero-offset exploding-reflector RTM",
            f"finite-difference grid: {nz} x {nx}",
            f"reverse-time cell updates: {updates}",
            "loaded execution only; not a shot-gather electromagnetic RTM",
        ),
    )


def _sponge(nz: int, nx: int, width: int, strength: float) -> np.ndarray:
    width = max(0, min(int(width), max(0, min(nz, nx) // 3)))
    damping = np.ones((nz, nx), dtype=np.float32)
    if width == 0:
        return damping
    strength = float(np.clip(strength, 0.0, 12.0))
    for offset in range(width):
        normalized = (width - offset) / max(width, 1)
        factor = math.exp(-strength * normalized * normalized)
        damping[-offset - 1, :] *= factor
        damping[:, offset] *= factor
        damping[:, -offset - 1] *= factor
    # The acquisition surface remains an injection boundary rather than a
    # damped boundary.  Damping begins below the source row.
    return damping


def _laplacian(field: np.ndarray, dx_m: float, dz_m: float, out: np.ndarray) -> None:
    out.fill(0.0)
    out[1:-1, 1:-1] = (
        (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / dx_m**2
        + (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / dz_m**2
    )


def _validate_rtm_budget(geometry: dict[str, float | int], params: dict[str, Any]) -> tuple[int, int, int]:
    samples = int(geometry["samples"])
    traces = int(geometry["traces"])
    nz = int(geometry["nz"])
    updates = samples * int(geometry["substeps"]) * nz * traces
    max_grid_elements = max(1, int(params.get("max_grid_elements", 40_000_000)))
    max_cell_updates = max(1, int(params.get("max_cell_updates", 1_500_000_000)))
    if nz * traces > max_grid_elements:
        raise RTMResourceLimitError(f"RTM grid has {nz * traces} cells; limit is {max_grid_elements}")
    if updates > max_cell_updates:
        raise RTMResourceLimitError(f"RTM requires {updates} cell updates; limit is {max_cell_updates}")
    return updates, max_grid_elements, max_cell_updates


def _reverse_propagate(
    arr: np.ndarray, geometry: dict[str, float | int], params: dict[str, Any]
) -> tuple[np.ndarray, int]:
    samples = int(geometry["samples"])
    traces = int(geometry["traces"])
    nz = int(geometry["nz"])
    substeps = int(geometry["substeps"])
    checker = _cancel_checker(params)
    context = params.get("_execution_context")
    source_depth = max(1, min(nz - 2, int(params.get("source_depth_index", 1))))
    damping = _sponge(
        nz, traces,
        int(params.get("boundary_width", min(24, max(2, min(nz, traces) // 8)))),
        float(params.get("boundary_strength", 3.5)),
    )
    previous = np.zeros((nz, traces), dtype=np.float32)
    current = np.zeros_like(previous)
    following = np.zeros_like(previous)
    lap = np.zeros_like(previous)
    dt_sub_ns = float(geometry["dt_ns"]) / substeps
    coefficient = np.float32((float(geometry["propagation_velocity_m_per_ns"]) * dt_sub_ns) ** 2)
    injection_scale = np.float32(float(params.get("injection_scale", 1.0)))
    for reverse_index, time_index in enumerate(range(samples - 1, -1, -1), start=1):
        if checker():
            raise RuntimeError("processing cancelled during RTM reverse propagation")
        current[source_depth, :] += np.asarray(arr[time_index, :], dtype=np.float32) * injection_scale
        for _ in range(substeps):
            _laplacian(current, float(geometry["dx_m"]), float(geometry["dz_m"]), lap)
            np.multiply(current, 2.0, out=following)
            following -= previous
            following += coefficient * lap
            following *= damping
            previous, current, following = current, following, previous
        if context is not None and hasattr(context, "report_progress"):
            context.report_progress(reverse_index, samples, f"RTM reverse step {reverse_index}/{samples}")
    return np.asarray(current, dtype=np.float32), source_depth


def _rtm_metadata(
    geometry: dict[str, float | int], updates: int, max_grid: int, max_updates: int
) -> dict[str, Any]:
    nz = int(geometry["nz"])
    traces = int(geometry["traces"])
    dz_m = float(geometry["dz_m"])
    dx_m = float(geometry["dx_m"])
    return {
        "method": "rtm_migration",
        "migration_mode": "zero_offset_exploding_reflector_scalar_2d",
        "stability": {
            "input_cfl": float(geometry["cfl"]),
            "cfl_limit": float(geometry["cfl_limit"]),
            "substeps": int(geometry["substeps"]),
        },
        "resource_contract": {
            "grid_elements": nz * traces,
            "cell_updates": updates,
            "max_grid_elements": max_grid,
            "max_cell_updates": max_updates,
        },
        "mapped_params": dict(geometry),
        "header_info_updates": {
            "a_scan_length": nz, "num_traces": traces, "is_depth": True, "is_elevation": False,
            "depth_step_m": dz_m, "depth_max_m": dz_m * max(nz - 1, 0),
            "trace_interval_m": dx_m, "total_time_ns": 0.0,
            "display_hint": "signed_migration", "display_center_zero": True,
            "display_skip_preprocess": True, "display_title": "Experimental RTM migration profile",
            "display_xlabel": "水平距离 (m)", "display_ylabel": "深度 (m)",
            "display_colorbar_label": "反传波场幅度",
        },
    }


def method_rtm_migration_native(
    data: Any, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reverse continue a zero-offset B-scan to an exploding-reflector image."""
    arr, warnings = ensure_matrix(data)
    geometry = _resolve_geometry(arr.shape, params, _header(params))
    updates, max_grid, max_updates = _validate_rtm_budget(geometry, params)
    image, source_depth = _reverse_propagate(arr, geometry, params)
    if bool(params.get("remove_surface_row", True)):
        image[: source_depth + 1, :] = 0.0
    if bool(params.get("normalize", False)):
        scale = float(np.max(np.abs(image)))
        if scale > 0.0:
            image = image / scale
    warnings.append(
        warning(
            "experimental_rtm_baseline",
            "当前 RTM 为零偏移爆炸反射体标量波基线，不等同于多炮多检波电磁 RTM。",
            "rtm_migration",
        )
    )
    return normalize_output(
        "rtm_migration", image, _rtm_metadata(geometry, updates, max_grid, max_updates), warnings
    )


__all__ = [
    "RTMResourceLimitError",
    "estimate_rtm_resources",
    "method_rtm_migration_native",
]
