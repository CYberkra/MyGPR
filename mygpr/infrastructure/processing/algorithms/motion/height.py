#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-visible UAV-GPR height normalization atomic node.

The public method id remains ``motion_compensation_height``, but the physical
contract now follows motion compensation V2: AGL height is preferred, the air
path velocity defaults to c0, and all clamps/warnings are explicit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.infrastructure.processing.algorithms.motion.shared import (
    AIR_WAVE_SPEED_M_PER_NS,
    apply_time_shift,
    clone_metadata,
    compute_reference_height,
    motion_warning,
    resolve_shift_sample_limit,
    resolve_time_window_ns,
    select_height,
)
from mygpr.domain.common.scalars import to_float, to_optional_float


METHOD_ID = "motion_compensation_height"


def _skip(
    data: np.ndarray,
    *,
    reason: str,
    code: str,
    trace_count: int,
    warnings: list[dict[str, Any]] | None = None,
    quality_flags: list[str] | None = None,
    **extra: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    runtime_warnings = list(warnings or [])
    flags = list(quality_flags or [])
    if code not in flags:
        flags.append(code)
    runtime_warnings.append(motion_warning(METHOD_ID, code, reason, **extra))
    return np.array(data, copy=True), {
        "method": METHOD_ID,
        "skipped": True,
        "reason": reason,
        "source_traces": int(trace_count),
        "input_height_valid": False,
        "runtime_warnings": runtime_warnings,
        "quality_flags": sorted(set(flags)),
        **extra,
    }


def _reason_for_invalid_height(height: np.ndarray | None) -> str:
    if height is None:
        return "缺少 height_agl_m / flight_height_m 高度字段"
    if height.size == 0:
        return "高度数组为空"
    if not np.isfinite(height).all():
        if np.isnan(height).any():
            return "高度包含 NaN"
        return "高度包含 Inf"
    if np.any(height <= 0.0):
        return "高度包含零或负值"
    return "高度长度与道数不一致"


def method_motion_compensation_height(
    data: np.ndarray,
    reference_height_mode: str = "mean",
    manual_height: float = 0.0,
    height_source: str = "auto",
    compensate_amplitude: bool = True,
    compensate_time_shift: bool = True,
    air_wave_speed_m_per_ns: float = AIR_WAVE_SPEED_M_PER_NS,
    wave_speed_m_per_ns: float | None = None,
    max_shift_samples: float | None = 0.0,
    max_shift_ns: float = 20.0,
    max_amplitude_scale: float = 2.0,
    interpolation_mode: str = "linear",
    trace_metadata: dict[str, Any] | None = None,
    header_info: dict[str, Any] | None = None,
    time_window_ns: float | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize height effects using the shared UAV motion V2 core."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("motion_compensation_height requires a 2D B-scan array")
    if interpolation_mode != "linear":
        raise ValueError(
            f"interpolation_mode '{interpolation_mode}' 不受支持；当前仅支持 'linear'"
        )

    samples, trace_count = arr.shape
    metadata = clone_metadata(trace_metadata)
    warnings: list[dict[str, Any]] = []
    quality_flags: list[str] = []
    if not metadata:
        return _skip(
            arr,
            reason="缺少 height_agl_m / flight_height_m 高度字段",
            code="missing_height_agl",
            trace_count=trace_count,
        )

    try:
        height_m, height_source_used = select_height(
            metadata,
            trace_count,
            height_source=height_source,
            method_id=METHOD_ID,
            warnings=warnings,
            quality_flags=quality_flags,
        )
    except ValueError:
        metadata_count = 0
        for candidate in ("height_agl_m", "flight_height_m", str(height_source)):
            if candidate in metadata:
                metadata_count = int(np.asarray(metadata[candidate]).size)
                break
        reason = "高度数组为空" if metadata_count == 0 else "高度长度与道数不一致"
        return _skip(
            arr,
            reason=reason,
            code="height_length_mismatch",
            trace_count=trace_count,
            warnings=warnings,
            quality_flags=quality_flags,
            height_length_mismatch=True,
            metadata_trace_count=metadata_count,
            data_trace_count=trace_count,
        )

    if height_m is None:
        return _skip(
            arr,
            reason=_reason_for_invalid_height(height_m),
            code="missing_height_agl",
            trace_count=trace_count,
            warnings=warnings,
            quality_flags=quality_flags,
        )
    if height_m.size != trace_count:
        return _skip(
            arr,
            reason="高度长度与道数不一致",
            code="height_length_mismatch",
            trace_count=trace_count,
            warnings=warnings,
            quality_flags=quality_flags,
            height_length_mismatch=True,
            metadata_trace_count=int(height_m.size),
            data_trace_count=trace_count,
        )
    if height_m.size == 0 or not np.isfinite(height_m).all() or np.any(height_m <= 0.0):
        return _skip(
            arr,
            reason=_reason_for_invalid_height(height_m),
            code="invalid_height_agl",
            trace_count=trace_count,
            warnings=warnings,
            quality_flags=quality_flags,
        )

    speed_value = to_float(
        wave_speed_m_per_ns if wave_speed_m_per_ns is not None else air_wave_speed_m_per_ns,
        default=AIR_WAVE_SPEED_M_PER_NS,
    )
    if speed_value <= 0.0 or not np.isfinite(speed_value):
        warnings.append(
            motion_warning(
                METHOD_ID,
                "invalid_air_wave_speed",
                "air_wave_speed_m_per_ns is invalid; falling back to c0.",
                requested=speed_value,
            )
        )
        quality_flags.append("invalid_air_wave_speed")
        speed_value = AIR_WAVE_SPEED_M_PER_NS

    manual_height_value = to_float(manual_height, default=0.0)
    max_shift_samples_value = to_optional_float(max_shift_samples)
    max_shift_ns_value = to_float(max_shift_ns, default=0.0)
    max_amplitude_scale_value = max(to_float(max_amplitude_scale, default=2.0), 1.0)
    corrected = np.array(arr, copy=True)
    h_ref = compute_reference_height(
        height_m,
        mode=reference_height_mode,
        manual_height_m=manual_height_value,
        warnings=warnings,
        method_id=METHOD_ID,
    )
    if h_ref <= 0.0 or not np.isfinite(h_ref):
        h_ref = float(np.mean(height_m))

    meta: dict[str, Any] = {
        "method": METHOD_ID,
        "skipped": False,
        "source_traces": int(trace_count),
        "input_height_valid": True,
        "height_source_requested": str(height_source),
        "height_source_used": str(height_source_used),
        "reference_height_mode": str(reference_height_mode),
        "reference_height_m": float(h_ref),
        "height_reference_m": float(h_ref),
        "air_wave_speed_m_per_ns": float(speed_value),
        "wave_speed_m_per_ns": float(speed_value),
        "max_shift_samples": max_shift_samples_value,
        "max_shift_samples_requested": max_shift_samples_value,
        "max_shift_ns_requested": max_shift_ns_value,
        "max_amplitude_scale": float(max_amplitude_scale_value),
        "height_summary": {
            "min_m": float(np.min(height_m)),
            "max_m": float(np.max(height_m)),
            "mean_m": float(np.mean(height_m)),
            "std_m": float(np.std(height_m)),
        },
        "height_correction_applied": False,
        "amplitude_correction_applied": False,
        "time_shift_correction_applied": False,
        "shift_clamped": False,
        "time_shift_clamped": False,
        "trace_metadata_updates": {
            "height_agl_m": height_m.astype(np.float32),
            "height_source": np.full(trace_count, str(height_source_used), dtype="<U32"),
        },
        "provenance": {
            "schema": "motion_compensation_atomic_v2",
            "shared_core": "PythonModule.motion_compensation_core",
            "height_priority": ["height_agl_m", "flight_height_m"],
            "air_path_velocity": "c0",
        },
    }

    if compensate_amplitude:
        amp_scale = (height_m / h_ref) ** 2
        amp_scale = np.clip(
            amp_scale,
            1.0 / max_amplitude_scale_value,
            max_amplitude_scale_value,
        )
        corrected = corrected * amp_scale[np.newaxis, :].astype(np.float32)
        meta["trace_metadata_updates"]["height_amplitude_scale"] = amp_scale.astype(np.float32)
        meta["amplitude_correction_applied"] = True

    if compensate_time_shift:
        resolved_time_window = resolve_time_window_ns(
            explicit_time_window_ns=time_window_ns,
            trace_metadata=metadata,
            header_info=header_info,
            kwargs=kwargs,
        )
        if resolved_time_window is None or resolved_time_window <= 0.0:
            warnings.append(
                motion_warning(
                    METHOD_ID,
                    "missing_time_window_ns",
                    "time_window_ns is missing; time-shift correction skipped.",
                )
            )
            quality_flags.append("missing_time_window_ns")
        else:
            dt_ns = float(resolved_time_window) / max(samples - 1, 1)
            time_shift_ns = 2.0 * (height_m - h_ref) / speed_value
            time_shift_samples = time_shift_ns / dt_ns
            raw_shift_samples = time_shift_samples.copy()
            clamp, clamp_source, clamp_details = resolve_shift_sample_limit(
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
            else:
                meta["max_shift_samples_effective"] = float(np.max(np.abs(time_shift_samples)))
                meta["max_shift_limit_source"] = None

            clamped_mask = ~np.isclose(raw_shift_samples, time_shift_samples)
            if bool(np.any(clamped_mask)):
                quality_flags.append("time_shift_clamped")
                warnings.append(
                    motion_warning(
                        METHOD_ID,
                        "time_shift_clamped",
                        "Height time-shift correction exceeded configured limits and was clamped.",
                        clamped_trace_count=int(np.count_nonzero(clamped_mask)),
                        total_trace_count=trace_count,
                        max_shift_samples_effective=float(meta["max_shift_samples_effective"]),
                        max_shift_limit_source=str(clamp_source),
                    )
                )
                meta["shift_clamped"] = True
                meta["time_shift_clamped"] = True

            corrected = apply_time_shift(corrected, time_shift_samples)
            meta["time_window_ns"] = float(resolved_time_window)
            meta["sample_interval_ns"] = float(dt_ns)
            meta["time_shift_ns"] = time_shift_ns.astype(np.float32)
            meta["time_shift_samples"] = time_shift_samples.astype(np.float32)
            meta["raw_time_shift_samples_min"] = float(np.min(raw_shift_samples))
            meta["raw_time_shift_samples_max"] = float(np.max(raw_shift_samples))
            meta["max_shift_samples_applied"] = float(np.max(np.abs(time_shift_samples)))
            meta["trace_metadata_updates"]["time_shift_ns"] = time_shift_ns.astype(np.float32)
            meta["trace_metadata_updates"]["time_shift_samples"] = time_shift_samples.astype(np.float32)
            meta["time_shift_correction_applied"] = True

    meta["height_correction_applied"] = bool(
        meta["amplitude_correction_applied"] or meta["time_shift_correction_applied"]
    )
    if "max_shift_samples_applied" not in meta:
        meta["max_shift_samples_applied"] = 0.0
    meta["runtime_warnings"] = warnings
    meta["quality_flags"] = sorted(set(quality_flags))
    return corrected.astype(np.float32, copy=False), meta
