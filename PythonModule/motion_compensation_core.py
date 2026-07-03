#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared UAV-GPR motion compensation helpers.

This module holds the V2 physical assumptions used by both the unified
``motion_compensation_v2`` entry point and the user-visible atomic motion
nodes.  The helpers deliberately avoid GUI concepts; they operate on B-scan
arrays plus per-trace metadata and return explicit metadata updates, runtime
warnings, and quality flags.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.runtime_warnings import build_runtime_warning
from core.scalar_utils import to_float, to_optional_float
from core.trace_metadata_utils import (
    build_uniform_trace_distance_m,
    resample_bscan_columns_linear,
    resample_trace_metadata,
)


AIR_WAVE_SPEED_M_PER_NS = 0.299792458


def clone_metadata(trace_metadata: dict[str, Any] | None) -> dict[str, np.ndarray]:
    """Return a defensive ndarray copy of trace metadata."""
    if not trace_metadata:
        return {}
    return {key: np.array(value, copy=True) for key, value in trace_metadata.items()}


def field_1d(
    metadata: dict[str, np.ndarray],
    key: str,
    trace_count: int,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Read a metadata field as a 1D array aligned to trace count."""
    values = np.asarray(metadata[key], dtype=dtype)
    if values.ndim == 0:
        values = np.full(trace_count, values.item(), dtype=dtype)
    if values.ndim != 1:
        raise ValueError(f"trace_metadata['{key}'] must be 1D")
    if values.size < trace_count:
        raise ValueError(f"trace_metadata['{key}'] length is shorter than trace count")
    return values[:trace_count].astype(dtype, copy=True)


def numeric_field_or_none(
    metadata: dict[str, np.ndarray],
    key: str,
    trace_count: int,
) -> np.ndarray | None:
    """Read a numeric 1D field, returning None when unavailable or invalid."""
    if key not in metadata:
        return None
    try:
        return field_1d(metadata, key, trace_count, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def motion_warning(
    method_id: str,
    code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    """Build a structured warning with method provenance."""
    return build_runtime_warning(code, message, method_id=method_id, **details)


def append_quality_warning(
    warnings: list[dict[str, Any]],
    quality_flags: list[str],
    method_id: str,
    code: str,
    message: str,
    **details: Any,
) -> None:
    """Append a warning and the matching quality flag."""
    quality_flags.append(code)
    warnings.append(motion_warning(method_id, code, message, **details))


def compute_trace_distance(local_x_m: np.ndarray, local_y_m: np.ndarray) -> np.ndarray:
    """Compute cumulative along-track distance from local x/y coordinates."""
    if local_x_m.size == 0:
        return np.array([], dtype=np.float64)
    step = np.hypot(np.diff(local_x_m), np.diff(local_y_m))
    distance = np.empty(local_x_m.size, dtype=np.float64)
    distance[0] = 0.0
    distance[1:] = np.cumsum(step, dtype=np.float64)
    return distance


def compute_reference_height(
    height: np.ndarray,
    *,
    mode: str,
    manual_height_m: float,
    warnings: list[dict[str, Any]],
    method_id: str,
) -> float:
    """Resolve the reference AGL height using the V2 policy."""
    mode = str(mode or "mean")
    if mode == "min":
        return float(np.min(height))
    if mode == "manual":
        manual = to_float(manual_height_m, default=0.0)
        if manual > 0.0 and np.isfinite(manual):
            return manual
        warnings.append(
            motion_warning(
                method_id,
                "invalid_manual_height",
                "manual_height_m is invalid; falling back to mean AGL height.",
                requested=manual_height_m,
            )
        )
    return float(np.mean(height))


def select_height(
    metadata: dict[str, np.ndarray],
    trace_count: int,
    *,
    height_source: str = "auto",
    method_id: str,
    warnings: list[dict[str, Any]],
    quality_flags: list[str],
) -> tuple[np.ndarray | None, str | None]:
    """Select AGL height using V2 priority: height_agl_m before flight_height_m."""
    source = str(height_source or "auto")
    if source in {"auto", "height_agl_m"} and "height_agl_m" in metadata:
        return field_1d(metadata, "height_agl_m", trace_count, dtype=np.float64), "height_agl_m"

    if source not in {"auto", "height_agl_m"} and source in metadata:
        return field_1d(metadata, source, trace_count, dtype=np.float64), source

    if source == "auto" and "flight_height_m" in metadata:
        warnings.append(
            motion_warning(
                method_id,
                "height_source_fallback",
                "height_agl_m is missing; using legacy flight_height_m as AGL fallback.",
            )
        )
        quality_flags.append("height_from_legacy_flight_height")
        return field_1d(metadata, "flight_height_m", trace_count, dtype=np.float64), "flight_height_m"

    quality_flags.append("missing_height_agl")
    warnings.append(
        motion_warning(
            method_id,
            "missing_height_agl",
            "No valid AGL height field is available; height correction was skipped.",
            requested_source=source,
        )
    )
    return None, None


def apply_time_shift(data: np.ndarray, shift_samples: np.ndarray) -> np.ndarray:
    """Apply per-trace vertical interpolation shifts."""
    sample_indices = np.arange(data.shape[0], dtype=np.float64)
    shifted = np.array(data, copy=True)
    for trace_idx in range(data.shape[1]):
        shift = float(shift_samples[trace_idx])
        if abs(shift) < 1.0e-6:
            continue
        source_indices = np.clip(sample_indices - shift, 0, data.shape[0] - 1)
        shifted[:, trace_idx] = np.interp(
            sample_indices,
            source_indices,
            shifted[:, trace_idx],
        ).astype(np.float32)
    return shifted


def resolve_shift_sample_limit(
    *,
    max_shift_samples: float | None,
    max_shift_ns: float | None,
    sample_interval_ns: float,
    sample_count: int,
) -> tuple[float | None, str | None, dict[str, float]]:
    """Resolve V2 sample-shift clamp from sample and nanosecond limits."""
    candidates: list[tuple[str, float]] = []
    sample_limit = to_float(max_shift_samples, default=0.0)
    if sample_limit > 0.0 and np.isfinite(sample_limit):
        candidates.append(("max_shift_samples", sample_limit))

    ns_limit = to_float(max_shift_ns, default=0.0)
    if ns_limit > 0.0 and np.isfinite(ns_limit) and sample_interval_ns > 0.0:
        candidates.append(("max_shift_ns", ns_limit / sample_interval_ns))

    if not candidates:
        return None, None, {}

    requested_limit = min(value for _, value in candidates)
    source = "+".join(name for name, _ in candidates)
    data_cap = max(1.0, min(float(sample_count - 1), float(sample_count) * 0.35))
    effective = min(requested_limit, data_cap)
    if effective < requested_limit:
        source = f"{source}+data_fraction_cap"
    return float(effective), source, {
        "requested_shift_limit_samples": float(requested_limit),
        "data_fraction_cap_samples": float(data_cap),
    }


def resolve_time_window_ns(
    *,
    explicit_time_window_ns: Any = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    header_info: dict[str, Any] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> float | None:
    """Resolve time_window_ns without inventing a physical time window."""
    if explicit_time_window_ns is not None:
        value = to_optional_float(explicit_time_window_ns)
        if value is not None:
            return value
    if kwargs and kwargs.get("time_window_ns") is not None:
        value = to_optional_float(kwargs.get("time_window_ns"))
        if value is not None:
            return value
    if header_info:
        for key in ("total_time_ns", "time_window_ns"):
            if header_info.get(key) is not None:
                value = to_optional_float(header_info.get(key))
                if value is not None:
                    return value
    if trace_metadata and "time_window_ns" in trace_metadata:
        return to_optional_float(trace_metadata.get("time_window_ns"))
    return None


def rotate_xy(
    x_body_m: np.ndarray,
    y_body_m: np.ndarray,
    yaw_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate body-frame x/y offsets into local x/y."""
    local_x = x_body_m * np.cos(yaw_rad) - y_body_m * np.sin(yaw_rad)
    local_y = x_body_m * np.sin(yaw_rad) + y_body_m * np.cos(yaw_rad)
    return local_x, local_y


def build_attitude_updates(
    metadata: dict[str, np.ndarray],
    trace_count: int,
    *,
    height_m: np.ndarray | None,
    apc_offset_x_m: float,
    apc_offset_y_m: float,
    apc_offset_z_m: float,
    max_abs_tilt_deg: float,
    method_id: str,
    quality_flags: list[str],
    warnings: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Build APC/attitude footprint metadata updates."""
    required = ("local_x_m", "local_y_m", "roll_deg", "pitch_deg", "yaw_deg")
    if any(key not in metadata for key in required):
        return {}
    if (
        height_m is None
        and abs(float(apc_offset_x_m)) < 1.0e-12
        and abs(float(apc_offset_y_m)) < 1.0e-12
        and abs(float(apc_offset_z_m)) < 1.0e-12
    ):
        return {}

    local_x_m = field_1d(metadata, "local_x_m", trace_count, dtype=np.float64)
    local_y_m = field_1d(metadata, "local_y_m", trace_count, dtype=np.float64)
    roll_deg = field_1d(metadata, "roll_deg", trace_count, dtype=np.float64)
    pitch_deg = field_1d(metadata, "pitch_deg", trace_count, dtype=np.float64)
    yaw_deg = field_1d(metadata, "yaw_deg", trace_count, dtype=np.float64)
    if not (
        np.isfinite(local_x_m).all()
        and np.isfinite(local_y_m).all()
        and np.isfinite(roll_deg).all()
        and np.isfinite(pitch_deg).all()
        and np.isfinite(yaw_deg).all()
    ):
        append_quality_warning(
            warnings,
            quality_flags,
            method_id,
            "invalid_attitude_or_position",
            "Attitude/APC metadata contains non-finite values; footprint updates skipped.",
        )
        return {}

    tilt_limit = max(float(max_abs_tilt_deg), 0.1)
    roll_used_deg = np.clip(roll_deg, -tilt_limit, tilt_limit)
    pitch_used_deg = np.clip(pitch_deg, -tilt_limit, tilt_limit)
    clamped = (roll_used_deg != roll_deg) | (pitch_used_deg != pitch_deg)
    if np.any(clamped):
        append_quality_warning(
            warnings,
            quality_flags,
            method_id,
            "attitude_clamped",
            "Roll/pitch exceeded the configured tilt limit and were clamped.",
            clamped_trace_count=int(np.count_nonzero(clamped)),
            max_abs_tilt_deg=tilt_limit,
        )

    yaw_rad = np.deg2rad(yaw_deg)
    apc_x = np.full(trace_count, float(apc_offset_x_m), dtype=np.float64)
    apc_y = np.full(trace_count, float(apc_offset_y_m), dtype=np.float64)
    apc_local_x, apc_local_y = rotate_xy(apc_x, apc_y, yaw_rad)

    if height_m is None:
        projection_height = np.full(trace_count, float(apc_offset_z_m), dtype=np.float64)
    else:
        projection_height = height_m + float(apc_offset_z_m)
    if np.any(~np.isfinite(projection_height)) or np.any(projection_height < 0.0):
        append_quality_warning(
            warnings,
            quality_flags,
            method_id,
            "invalid_projection_height",
            "Projection height is invalid; footprint updates skipped.",
        )
        return {}

    pitch_body = projection_height * np.tan(np.deg2rad(pitch_used_deg))
    roll_body = projection_height * np.tan(np.deg2rad(roll_used_deg))
    footprint_dx, footprint_dy = rotate_xy(pitch_body, roll_body, yaw_rad)

    corrected_x = local_x_m + apc_local_x + footprint_dx
    corrected_y = local_y_m + apc_local_y + footprint_dy
    return {
        "local_x_m": corrected_x.astype(np.float64),
        "local_y_m": corrected_y.astype(np.float64),
        "footprint_x_m": corrected_x.astype(np.float64),
        "footprint_y_m": corrected_y.astype(np.float64),
        "trace_distance_m": compute_trace_distance(corrected_x, corrected_y),
    }


def metadata_for_output(
    metadata: dict[str, np.ndarray],
    updates: dict[str, np.ndarray],
    trace_count: int,
) -> dict[str, np.ndarray]:
    """Merge input metadata and same-length updates for output/resampling."""
    prepared: dict[str, np.ndarray] = {}
    for key, value in metadata.items():
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.size == 1:
            prepared[key] = np.array(arr, copy=True)
            continue
        if arr.ndim == 1 and arr.size >= trace_count:
            prepared[key] = np.array(arr[:trace_count], copy=True)
    for key, value in updates.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size == trace_count:
            prepared[key] = np.array(arr, copy=True)
    if "trace_index" not in prepared:
        prepared["trace_index"] = np.arange(trace_count, dtype=np.int32)
    return prepared


def resample_bscan_columns(
    data: np.ndarray,
    source_distance_m: np.ndarray,
    target_distance_m: np.ndarray,
) -> np.ndarray:
    """Linearly resample B-scan columns by along-track distance."""
    return resample_bscan_columns_linear(data, source_distance_m, target_distance_m)


def resample_equal_distance(
    data: np.ndarray,
    metadata: dict[str, np.ndarray],
    *,
    spacing_m: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Resample B-scan and metadata to equal trace spacing."""
    source_distance = np.asarray(metadata["trace_distance_m"], dtype=np.float64)
    target_distance = build_uniform_trace_distance_m(source_distance, spacing_m=spacing_m)
    corrected = resample_bscan_columns(data, source_distance, target_distance)
    trace_metadata_out = resample_trace_metadata(
        metadata,
        target_trace_distance_m=target_distance,
    )
    return corrected, trace_metadata_out, {
        "source_traces": int(source_distance.size),
        "target_traces": int(target_distance.size),
        "spacing_m": float(spacing_m),
        "distance_start_m": float(source_distance[0]),
        "distance_end_m": float(source_distance[-1]),
    }
