#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified UAV-GPR motion compensation V2.

The V2 method is the first stable contract for airborne CSV processing. It
keeps V1 modules available, but combines the physically coupled pieces that
must share the same per-trace metadata: AGL height correction, conservative
amplitude normalization, attitude/APC footprint metadata, and optional
equal-distance resampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from PythonModule.motion_compensation_core import (
    AIR_WAVE_SPEED_M_PER_NS,
    apply_time_shift,
    append_quality_warning,
    build_attitude_updates,
    clone_metadata,
    compute_reference_height,
    field_1d,
    metadata_for_output,
    motion_warning,
    numeric_field_or_none,
    resample_bscan_columns,
    resample_equal_distance,
    resolve_shift_sample_limit,
    resolve_time_window_ns,
    select_height,
)
from core.scalar_utils import to_float, to_optional_float


def _warning(code: str, message: str, **details: Any) -> dict[str, Any]:
    return motion_warning("motion_compensation_v2", code, message, **details)


def _append_quality_warning(
    warnings: list[dict[str, Any]],
    quality_flags: list[str],
    code: str,
    message: str,
    **details: Any,
) -> None:
    append_quality_warning(
        warnings,
        quality_flags,
        "motion_compensation_v2",
        code,
        message,
        **details,
    )


def _gap_stats(values: np.ndarray) -> tuple[float, float, float] | None:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size < 3:
        return None
    median = float(np.median(positive))
    maximum = float(np.max(positive))
    if median <= 0.0:
        return None
    return median, maximum, maximum / median


def _analyze_trace_quality(
    metadata: dict[str, np.ndarray],
    trace_count: int,
    input_quality: dict[str, Any],
    warnings: list[dict[str, Any]],
    quality_flags: list[str],
) -> None:
    timestamps = _numeric_field_or_none(metadata, "trace_timestamp_s", trace_count)
    distance = _numeric_field_or_none(metadata, "trace_distance_m", trace_count)
    input_quality["trace_timestamp_available"] = timestamps is not None
    input_quality["trace_distance_available"] = distance is not None

    timestamp_steps: np.ndarray | None = None
    distance_steps: np.ndarray | None = None

    if timestamps is not None:
        if not np.isfinite(timestamps).all():
            invalid_count = int(np.count_nonzero(~np.isfinite(timestamps)))
            input_quality["trace_timestamp_invalid_count"] = invalid_count
            _append_quality_warning(
                warnings,
                quality_flags,
                "invalid_trace_timestamp_s",
                "trace_timestamp_s contains non-finite values; sidecar timing quality is unreliable.",
                invalid_trace_count=invalid_count,
                total_trace_count=trace_count,
            )
        else:
            input_quality["trace_timestamp_min_s"] = float(np.min(timestamps))
            input_quality["trace_timestamp_max_s"] = float(np.max(timestamps))
            timestamp_steps = np.diff(timestamps)
            nonpositive = int(np.count_nonzero(timestamp_steps <= 0.0))
            input_quality["trace_timestamp_nonpositive_steps"] = nonpositive
            if nonpositive:
                _append_quality_warning(
                    warnings,
                    quality_flags,
                    "trace_timestamp_nonmonotonic",
                    "trace_timestamp_s is not strictly increasing; sidecar interpolation should be reviewed.",
                    nonpositive_step_count=nonpositive,
                    total_step_count=max(trace_count - 1, 0),
                )
            stats = _gap_stats(timestamp_steps)
            if stats is not None:
                median, maximum, ratio = stats
                input_quality["trace_timestamp_step_median_s"] = median
                input_quality["trace_timestamp_step_max_s"] = maximum
                input_quality["trace_timestamp_gap_ratio"] = ratio
                if ratio > 3.0:
                    _append_quality_warning(
                        warnings,
                        quality_flags,
                        "trace_timestamp_gap",
                        "trace timestamps contain a large sampling gap; motion compensation may bridge a flight interruption.",
                        median_step_s=median,
                        max_step_s=maximum,
                        gap_ratio=ratio,
                    )

    if distance is not None:
        if not np.isfinite(distance).all():
            invalid_count = int(np.count_nonzero(~np.isfinite(distance)))
            input_quality["trace_distance_invalid_count"] = invalid_count
            _append_quality_warning(
                warnings,
                quality_flags,
                "invalid_trace_distance_m",
                "trace_distance_m contains non-finite values; trajectory quality is unreliable.",
                invalid_trace_count=invalid_count,
                total_trace_count=trace_count,
            )
        else:
            input_quality["trace_distance_start_m"] = float(distance[0])
            input_quality["trace_distance_end_m"] = float(distance[-1])
            distance_steps = np.diff(distance)
            negative = int(np.count_nonzero(distance_steps < 0.0))
            input_quality["trace_distance_negative_steps"] = negative
            if negative:
                _append_quality_warning(
                    warnings,
                    quality_flags,
                    "trace_distance_nonmonotonic",
                    "trace_distance_m is not monotonic; equal-distance resampling and speed estimates should be reviewed.",
                    negative_step_count=negative,
                    total_step_count=max(trace_count - 1, 0),
                )
            stats = _gap_stats(distance_steps)
            if stats is not None:
                median, maximum, ratio = stats
                input_quality["trace_distance_step_median_m"] = median
                input_quality["trace_distance_step_max_m"] = maximum
                input_quality["trace_distance_gap_ratio"] = ratio
                if ratio > 3.0:
                    _append_quality_warning(
                        warnings,
                        quality_flags,
                        "trace_distance_gap",
                        "trace_distance_m contains a large spatial gap; resampling may smear across a flight break.",
                        median_step_m=median,
                        max_step_m=maximum,
                        gap_ratio=ratio,
                    )

    if timestamp_steps is not None and distance_steps is not None:
        mask = (
            np.isfinite(timestamp_steps)
            & np.isfinite(distance_steps)
            & (timestamp_steps > 0.0)
            & (distance_steps >= 0.0)
        )
        if np.count_nonzero(mask) >= 3:
            speed_mps = distance_steps[mask] / timestamp_steps[mask]
            stats = _gap_stats(speed_mps)
            if stats is not None:
                median, maximum, ratio = stats
                input_quality["trace_speed_median_mps"] = median
                input_quality["trace_speed_max_mps"] = maximum
                input_quality["trace_speed_outlier_ratio"] = ratio
                if ratio > 2.5:
                    _append_quality_warning(
                        warnings,
                        quality_flags,
                        "trace_speed_outlier",
                        "Trace spacing and timestamps imply a large speed outlier; trajectory synchronization should be reviewed.",
                        median_speed_mps=median,
                        max_speed_mps=maximum,
                        speed_ratio=ratio,
                    )


_clone_metadata = clone_metadata
_field_1d = field_1d
_numeric_field_or_none = numeric_field_or_none
_apply_time_shift = apply_time_shift
_metadata_for_output = metadata_for_output


def _compute_reference_height(
    height: np.ndarray,
    *,
    mode: str,
    manual_height_m: float,
    warnings: list[dict[str, Any]],
) -> float:
    return compute_reference_height(
        height,
        mode=mode,
        manual_height_m=manual_height_m,
        warnings=warnings,
        method_id="motion_compensation_v2",
    )


def _select_height(
    metadata: dict[str, np.ndarray],
    trace_count: int,
    *,
    height_source: str,
    warnings: list[dict[str, Any]],
    quality_flags: list[str],
) -> tuple[np.ndarray | None, str | None]:
    return select_height(
        metadata,
        trace_count,
        height_source=height_source,
        method_id="motion_compensation_v2",
        warnings=warnings,
        quality_flags=quality_flags,
    )


def _resolve_shift_sample_limit(
    *,
    max_shift_samples: float | None,
    max_shift_ns: float | None,
    sample_interval_ns: float,
    sample_count: int,
) -> tuple[float | None, str | None, dict[str, float]]:
    return resolve_shift_sample_limit(
        max_shift_samples=max_shift_samples,
        max_shift_ns=max_shift_ns,
        sample_interval_ns=sample_interval_ns,
        sample_count=sample_count,
    )


def _build_attitude_updates(
    metadata: dict[str, np.ndarray],
    trace_count: int,
    *,
    height_m: np.ndarray | None,
    apc_offset_x_m: float,
    apc_offset_y_m: float,
    apc_offset_z_m: float,
    max_abs_tilt_deg: float,
    quality_flags: list[str],
    warnings: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    return build_attitude_updates(
        metadata,
        trace_count,
        height_m=height_m,
        apc_offset_x_m=apc_offset_x_m,
        apc_offset_y_m=apc_offset_y_m,
        apc_offset_z_m=apc_offset_z_m,
        max_abs_tilt_deg=max_abs_tilt_deg,
        method_id="motion_compensation_v2",
        quality_flags=quality_flags,
        warnings=warnings,
    )


def _resample_bscan_columns(
    data: np.ndarray,
    source_distance_m: np.ndarray,
    target_distance_m: np.ndarray,
) -> np.ndarray:
    return resample_bscan_columns(data, source_distance_m, target_distance_m)


def method_motion_compensation_v2(
    data: np.ndarray,
    height_reference_mode: str = "mean",
    manual_height_m: float = 0.0,
    height_source: str = "auto",
    compensate_time_shift: bool = True,
    compensate_amplitude: bool = True,
    max_shift_samples: float | None = 0.0,
    max_shift_ns: float = 20.0,
    max_amplitude_scale: float = 2.0,
    resample_spacing_m: float = 0.0,
    interpolation_mode: str = "linear",
    apc_offset_x_m: float = 0.0,
    apc_offset_y_m: float = 0.0,
    apc_offset_z_m: float = 0.0,
    max_abs_tilt_deg: float = 20.0,
    air_wave_speed_m_per_ns: float = AIR_WAVE_SPEED_M_PER_NS,
    trace_metadata: dict[str, Any] | None = None,
    header_info: dict[str, Any] | None = None,
    time_window_ns: float | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run unified UAV-GPR motion compensation V2."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("motion_compensation_v2 requires a 2D B-scan array")
    if interpolation_mode != "linear":
        raise ValueError("motion_compensation_v2 currently supports linear interpolation only")

    samples, trace_count = arr.shape
    metadata = _clone_metadata(trace_metadata)
    warnings: list[dict[str, Any]] = []
    quality_flags: list[str] = []
    updates: dict[str, np.ndarray] = {}
    corrected = np.array(arr, copy=True)
    manual_height_value = to_float(manual_height_m, default=0.0)
    max_shift_samples_value = to_optional_float(max_shift_samples)
    max_shift_ns_value = to_float(max_shift_ns, default=0.0)
    max_amplitude_scale_value = to_float(max_amplitude_scale, default=2.0)
    resample_spacing_value = to_float(resample_spacing_m, default=0.0)
    air_wave_speed_value = to_float(
        air_wave_speed_m_per_ns,
        default=AIR_WAVE_SPEED_M_PER_NS,
    )
    apc_offset_x_value = to_float(apc_offset_x_m, default=0.0)
    apc_offset_y_value = to_float(apc_offset_y_m, default=0.0)
    apc_offset_z_value = to_float(apc_offset_z_m, default=0.0)
    max_abs_tilt_value = to_float(max_abs_tilt_deg, default=20.0)

    meta: dict[str, Any] = {
        "method": "motion_compensation_v2",
        "skipped": False,
        "source_traces": int(trace_count),
        "air_wave_speed_m_per_ns": air_wave_speed_value,
        "height_reference_mode": str(height_reference_mode),
        "max_shift_samples_requested": max_shift_samples_value,
        "max_shift_ns_requested": max_shift_ns_value,
        "height_correction_applied": False,
        "time_shift_correction_applied": False,
        "amplitude_correction_applied": False,
        "resampling_applied": False,
        "provenance": {
            "schema": "motion_compensation_v2",
            "height_priority": ["height_agl_m", "flight_height_m"],
            "air_path_velocity": "c0",
            "rtk_altitude_policy": "not_used_as_agl_without_height_agl_m",
        },
    }

    if not metadata:
        quality_flags.append("missing_trace_metadata")
        warnings.append(
            _warning(
                "missing_trace_metadata",
                "trace_metadata is missing; motion compensation V2 returned a data copy.",
            )
        )
        meta["input_quality"] = {
            "trace_count": int(trace_count),
            "height_source_requested": str(height_source),
            "height_source_used": None,
            "alignment_status_available": False,
            "height_confidence_available": False,
        }
        meta["quality_flags"] = quality_flags
        meta["runtime_warnings"] = warnings
        return corrected, meta

    try:
        height_m, height_source_used = _select_height(
            metadata,
            trace_count,
            height_source=height_source,
            warnings=warnings,
            quality_flags=quality_flags,
        )
    except ValueError as exc:
        height_m = None
        height_source_used = None
        quality_flags.append("invalid_height_agl")
        warnings.append(_warning("invalid_height_agl", str(exc)))

    valid_height = False
    if height_m is not None:
        valid_height = bool(
            height_m.ndim == 1
            and height_m.size == trace_count
            and np.isfinite(height_m).all()
            and np.all(height_m > 0.0)
        )
        if not valid_height:
            quality_flags.append("invalid_height_agl")
            warnings.append(
                _warning(
                    "invalid_height_agl",
                    "AGL height contains non-finite or non-positive values; height correction skipped.",
                )
            )

    alignment_status = np.asarray(metadata.get("alignment_status", []), dtype="<U16")
    height_confidence = _numeric_field_or_none(metadata, "height_confidence", trace_count)
    input_quality: dict[str, Any] = {
        "trace_count": int(trace_count),
        "height_source_requested": str(height_source),
        "height_source_used": str(height_source_used) if height_source_used else None,
        "alignment_status_available": bool(alignment_status.size),
        "height_confidence_available": height_confidence is not None,
    }
    _analyze_trace_quality(
        metadata,
        trace_count,
        input_quality,
        warnings,
        quality_flags,
    )

    if alignment_status.size:
        alignment_extrapolated = int(np.count_nonzero(alignment_status == "extrapolated"))
        alignment_resampled = int(np.count_nonzero(alignment_status == "resampled"))
        input_quality["alignment_extrapolated_traces"] = alignment_extrapolated
        input_quality["alignment_resampled_traces"] = alignment_resampled
        if alignment_extrapolated > 0:
            quality_flags.append("sidecar_extrapolated")
            warnings.append(
                _warning(
                    "sidecar_extrapolated",
                    "Some sidecar traces were extrapolated outside the available timestamp coverage.",
                    extrapolated_trace_count=alignment_extrapolated,
                    total_trace_count=trace_count,
                )
            )

    if height_confidence is not None:
        confidence_valid = height_confidence[np.isfinite(height_confidence)]
        if confidence_valid.size:
            low_count = int(np.count_nonzero(confidence_valid < 0.5))
            input_quality["height_confidence_min"] = float(np.min(confidence_valid))
            input_quality["height_confidence_mean"] = float(np.mean(confidence_valid))
            input_quality["height_confidence_low_traces"] = low_count
            if low_count > 0:
                quality_flags.append("low_height_confidence")
                warnings.append(
                    _warning(
                        "low_height_confidence",
                        "Height confidence is low for part of the line; motion compensation should be reviewed.",
                        low_confidence_trace_count=low_count,
                        total_trace_count=trace_count,
                    )
                )
        else:
            input_quality["height_confidence_min"] = None
            input_quality["height_confidence_mean"] = None
            input_quality["height_confidence_low_traces"] = 0

    meta["input_quality"] = input_quality

    if valid_height and air_wave_speed_value > 0.0:
        h_ref = _compute_reference_height(
            height_m,
            mode=height_reference_mode,
            manual_height_m=manual_height_value,
            warnings=warnings,
        )
        if h_ref <= 0.0 or not np.isfinite(h_ref):
            h_ref = float(np.mean(height_m))
        meta["height_source_used"] = str(height_source_used)
        meta["height_reference_m"] = float(h_ref)
        meta["height_summary"] = {
            "min_m": float(np.min(height_m)),
            "max_m": float(np.max(height_m)),
            "mean_m": float(np.mean(height_m)),
            "std_m": float(np.std(height_m)),
        }
        updates["height_agl_m"] = height_m.astype(np.float32)
        if "height_confidence" in metadata:
            updates["height_confidence"] = _field_1d(
                metadata, "height_confidence", trace_count, dtype=np.float32
            )
        else:
            updates["height_confidence"] = np.ones(trace_count, dtype=np.float32)
        if "height_source" in metadata:
            updates["height_source"] = _field_1d(
                metadata, "height_source", trace_count, dtype=str
            ).astype("<U32")
        elif height_source_used is not None:
            updates["height_source"] = np.full(
                trace_count, str(height_source_used), dtype="<U32"
            )

        if compensate_amplitude:
            max_scale = max(max_amplitude_scale_value, 1.0)
            amp_scale = (height_m / h_ref) ** 2
            amp_scale = np.clip(amp_scale, 1.0 / max_scale, max_scale)
            corrected = corrected * amp_scale[np.newaxis, :].astype(np.float32)
            updates["height_amplitude_scale"] = amp_scale.astype(np.float32)
            meta["amplitude_correction_applied"] = True

        if compensate_time_shift:
            resolved_time_window_value = resolve_time_window_ns(
                explicit_time_window_ns=time_window_ns,
                trace_metadata=metadata,
                header_info=header_info,
                kwargs=kwargs,
            )
            if resolved_time_window_value is None or resolved_time_window_value <= 0.0:
                quality_flags.append("missing_time_window_ns")
                warnings.append(
                    _warning(
                        "missing_time_window_ns",
                        "time_window_ns is missing; time-shift correction skipped.",
                    )
                )
            else:
                dt_ns = resolved_time_window_value / max(samples - 1, 1)
                time_shift_ns = 2.0 * (height_m - h_ref) / air_wave_speed_value
                time_shift_samples = time_shift_ns / dt_ns
                raw_shift_samples = time_shift_samples.copy()
                clamp, clamp_source, clamp_details = _resolve_shift_sample_limit(
                    max_shift_samples=max_shift_samples_value,
                    max_shift_ns=max_shift_ns_value,
                    sample_interval_ns=dt_ns,
                    sample_count=samples,
                )
                if clamp is not None:
                    time_shift_samples = np.clip(time_shift_samples, -clamp, clamp)
                    meta["max_shift_samples_effective"] = float(clamp)
                    meta["max_shift_limit_source"] = str(clamp_source)
                    meta.update(clamp_details)
                clamped_mask = ~np.isclose(raw_shift_samples, time_shift_samples)
                time_shift_clamped = bool(np.any(clamped_mask))
                if time_shift_clamped:
                    quality_flags.append("time_shift_clamped")
                    warnings.append(
                        _warning(
                            "time_shift_clamped",
                            "Height time-shift correction exceeded max_shift_samples and was clamped.",
                            clamped_trace_count=int(np.count_nonzero(clamped_mask)),
                            total_trace_count=trace_count,
                            max_shift_samples_effective=float(clamp or 0.0),
                            max_shift_limit_source=str(clamp_source),
                            raw_shift_samples_min=float(np.min(raw_shift_samples)),
                            raw_shift_samples_max=float(np.max(raw_shift_samples)),
                        )
                    )
                corrected = _apply_time_shift(corrected, time_shift_samples)
                updates["time_shift_ns"] = time_shift_ns.astype(np.float32)
                updates["time_shift_samples"] = time_shift_samples.astype(np.float32)
                meta["time_window_ns"] = resolved_time_window_value
                meta["sample_interval_ns"] = float(dt_ns)
                meta["time_shift_ns"] = time_shift_ns.astype(np.float32)
                meta["time_shift_samples"] = time_shift_samples.astype(np.float32)
                meta["raw_time_shift_samples_min"] = float(np.min(raw_shift_samples))
                meta["raw_time_shift_samples_max"] = float(np.max(raw_shift_samples))
                meta["time_shift_clamped"] = time_shift_clamped
                meta["time_shift_correction_applied"] = True

        meta["height_correction_applied"] = bool(
            meta["time_shift_correction_applied"]
            or meta["amplitude_correction_applied"]
        )
    else:
        meta["height_source_used"] = str(height_source_used) if height_source_used else None
        meta["height_summary"] = {}

    attitude_updates = _build_attitude_updates(
        metadata,
        trace_count,
        height_m=height_m if valid_height else None,
        apc_offset_x_m=apc_offset_x_value,
        apc_offset_y_m=apc_offset_y_value,
        apc_offset_z_m=apc_offset_z_value,
        max_abs_tilt_deg=max_abs_tilt_value,
        quality_flags=quality_flags,
        warnings=warnings,
    )
    updates.update(attitude_updates)

    if resample_spacing_value > 0.0:
        metadata_for_resampling = _metadata_for_output(metadata, updates, trace_count)
        source_distance = _numeric_field_or_none(
            metadata_for_resampling, "trace_distance_m", trace_count
        )
        if source_distance is None:
            quality_flags.append("missing_trace_distance_m")
            warnings.append(
                _warning(
                    "missing_trace_distance_m",
                    "trace_distance_m is missing; equal-distance resampling skipped.",
                )
            )
        elif np.any(np.diff(source_distance) < 0.0):
            quality_flags.append("nonmonotonic_trace_distance_m")
            warnings.append(
                _warning(
                    "nonmonotonic_trace_distance_m",
                    "trace_distance_m is not monotonic; equal-distance resampling skipped.",
                )
            )
        else:
            corrected, trace_metadata_out, resample_meta = resample_equal_distance(
                corrected,
                metadata_for_resampling,
                spacing_m=resample_spacing_value,
            )
            meta["trace_metadata_out"] = trace_metadata_out
            meta["target_traces"] = int(resample_meta["target_traces"])
            meta["resample_spacing_m"] = float(resample_meta["spacing_m"])
            meta["distance_start_m"] = float(resample_meta["distance_start_m"])
            meta["distance_end_m"] = float(resample_meta["distance_end_m"])
            meta["resampling_applied"] = True

    if not meta["resampling_applied"] and updates:
        meta["trace_metadata_updates"] = updates

    meta["quality_flags"] = sorted(set(quality_flags))
    if warnings:
        meta["runtime_warnings"] = warnings
    return corrected.astype(np.float32, copy=False), meta
