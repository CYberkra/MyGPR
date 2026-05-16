#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-aware parameter constraints for auto-tune candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log2
from typing import Any

from core.runtime_warnings import build_runtime_warning
from core.scalar_utils import to_float, to_float_or_none, to_int_or_none


@dataclass
class ParameterConstraintResult:
    """Requested and effective params after data-shape constraints."""

    requested_params: dict[str, Any]
    effective_params: dict[str, Any]
    warnings: list[dict[str, Any]]

    @property
    def adjusted(self) -> bool:
        """Whether any parameter changed."""
        return bool(self.warnings)


def constrain_auto_tune_params(
    method_key: str,
    params: dict[str, Any],
    data_shape: tuple[int, int],
    header_info: dict[str, Any] | None = None,
) -> ParameterConstraintResult:
    """Constrain auto-tune params to the current B-scan dimensions.

    Invalid non-numeric values are deliberately left unchanged so they still
    produce failed trials instead of being silently converted into valid ones.
    """
    requested = dict(params or {})
    effective = dict(requested)
    warnings: list[dict[str, Any]] = []
    n_samples, n_traces = _resolve_shape(data_shape)
    min_dim = max(1, min(n_samples, n_traces))

    if method_key == "set_zero_time":
        _clamp_float_param(
            method_key,
            effective,
            warnings,
            parameter="new_zero_time",
            minimum=0.0,
            maximum=_safe_zero_time_max_ns(n_samples, header_info or {}),
            unit="ns",
            reason="zero_time_safe_search_window",
        )
    elif method_key == "dewow":
        _clamp_int_param(
            method_key,
            effective,
            warnings,
            parameter="window",
            minimum=1,
            maximum=max(1, n_samples // 2),
            reason="sample_window_limit",
        )
    elif method_key == "agcGain":
        _clamp_int_param(
            method_key,
            effective,
            warnings,
            parameter="window",
            minimum=_safe_agc_window_min(n_samples, header_info or {}),
            maximum=n_samples,
            reason="sample_window_limit",
        )
    elif method_key in {
        "subtracting_average_2D",
        "median_background_2D",
        "running_average_2D",
    }:
        _clamp_int_param(
            method_key,
            effective,
            warnings,
            parameter="ntraces",
            minimum=1,
            maximum=n_traces,
            prefer_odd=method_key in {"median_background_2D", "running_average_2D"},
            reason="trace_window_limit",
        )
    elif method_key == "svd_bg":
        _clamp_int_param(
            method_key,
            effective,
            warnings,
            parameter="rank",
            minimum=1,
            maximum=max(1, min_dim - 1),
            reason="rank_limit",
        )
    elif method_key in {"svd_subspace", "wavelet_svd"}:
        _clamp_rank_interval(method_key, effective, warnings, rank_limit=min_dim)
        if method_key == "wavelet_svd":
            _clamp_int_param(
                method_key,
                effective,
                warnings,
                parameter="levels",
                minimum=1,
                maximum=_wavelet_level_limit(n_samples, n_traces),
                reason="wavelet_level_limit",
            )
    elif method_key == "hankel_svd":
        _clamp_int_param(
            method_key,
            effective,
            warnings,
            parameter="window_length",
            minimum=0,
            maximum=max(1, n_samples - 1),
            reason="sample_window_limit",
        )
        _clamp_int_param(
            method_key,
            effective,
            warnings,
            parameter="rank",
            minimum=0,
            maximum=max(1, min(10, n_samples - 1)),
            reason="rank_limit",
        )
    elif method_key == "frequency_filter_1d":
        _clamp_frequency_filter_params(
            method_key,
            effective,
            warnings,
            n_samples=n_samples,
            header_info=header_info or {},
        )
    elif method_key == "trajectory_smoothing":
        _clamp_int_param(
            method_key,
            effective,
            warnings,
            parameter="window_length",
            minimum=1,
            maximum=n_traces,
            prefer_odd=True,
            reason="trace_window_limit",
        )
        _clamp_polyorder(method_key, effective, warnings)
    elif method_key == "motion_compensation_vibration":
        _clamp_int_param(
            method_key,
            effective,
            warnings,
            parameter="smooth_window",
            minimum=1,
            maximum=n_traces,
            prefer_odd=True,
            reason="trace_window_limit",
        )

    return ParameterConstraintResult(
        requested_params=requested,
        effective_params=effective,
        warnings=warnings,
    )


def constrain_auto_tune_trials(
    method_key: str,
    trials: list[dict[str, Any]],
    data_shape: tuple[int, int],
    header_info: dict[str, Any] | None = None,
) -> list[ParameterConstraintResult]:
    """Constrain a list of trial params."""
    return [
        constrain_auto_tune_params(method_key, trial, data_shape, header_info)
        for trial in trials
    ]


def _resolve_shape(data_shape: tuple[int, int]) -> tuple[int, int]:
    if len(data_shape) != 2:
        return 1, 1
    n_samples = max(1, int(data_shape[0]))
    n_traces = max(1, int(data_shape[1]))
    return n_samples, n_traces


def _safe_zero_time_max_ns(n_samples: int, header_info: dict[str, Any]) -> float:
    total_time_ns = header_info.get("total_time_ns")
    total = to_float(total_time_ns, default=48.0)
    if total <= 0:
        total = 48.0
    max_shift_samples = max(0, min(int(n_samples) - 1, int(round(n_samples * 0.35))))
    return float(max_shift_samples) * total / max(1, int(n_samples))


def _safe_agc_window_min(n_samples: int, header_info: dict[str, Any]) -> int:
    samples = max(1, int(n_samples))
    min_by_fraction = int(round(samples * 0.02))
    min_by_time = 0
    total_time_ns = header_info.get("total_time_ns") if header_info else None
    total = to_float(total_time_ns, default=0.0)
    if total > 0.0:
        time_step_ns = total / samples
        min_by_time = int(round(0.5 / max(time_step_ns, 1.0e-9)))
    minimum = max(3, min_by_fraction, min_by_time)
    return max(1, min(samples, minimum))


def _wavelet_level_limit(n_samples: int, n_traces: int) -> int:
    smallest_dim = max(2, min(int(n_samples), int(n_traces)))
    return max(1, min(8, int(floor(log2(smallest_dim)))))


def _nyquist_mhz(n_samples: int, header_info: dict[str, Any]) -> float | None:
    sample_rate_hz = header_info.get("sample_rate_hz")
    sample_rate = to_float(sample_rate_hz, default=0.0)
    if sample_rate > 0.0:
        return float(sample_rate / 2.0e6)

    time_step_s = header_info.get("time_step_s")
    step_s = to_float(time_step_s, default=0.0)
    if step_s > 0.0:
        return float(1.0 / step_s / 2.0e6)

    total_time_ns = header_info.get("total_time_ns")
    total_ns = to_float(total_time_ns, default=0.0)
    if total_ns > 0.0:
        sample_rate = max(1, int(n_samples)) / (total_ns * 1.0e-9)
        return float(sample_rate / 2.0e6)
    return None


def _as_int(value: Any) -> int | None:
    return to_int_or_none(value)


def _as_float(value: Any) -> float | None:
    return to_float_or_none(value)


def _clamp_int_param(
    method_key: str,
    params: dict[str, Any],
    warnings: list[dict[str, Any]],
    *,
    parameter: str,
    minimum: int,
    maximum: int,
    reason: str,
    prefer_odd: bool = False,
) -> None:
    if parameter not in params:
        return
    requested = params.get(parameter)
    current = _as_int(requested)
    if current is None:
        return

    lower = int(minimum)
    upper = max(lower, int(maximum))
    effective = max(lower, min(upper, current))
    if prefer_odd and effective > lower and effective % 2 == 0:
        effective = effective - 1 if effective == upper else effective + 1
        effective = max(lower, min(upper, effective))
    if effective != current:
        params[parameter] = int(effective)
        warnings.append(
            _build_constraint_warning(
                method_key,
                parameter,
                requested=current,
                effective=effective,
                minimum=lower,
                maximum=upper,
                reason=reason,
            )
        )
    else:
        params[parameter] = int(effective)


def _clamp_float_param(
    method_key: str,
    params: dict[str, Any],
    warnings: list[dict[str, Any]],
    *,
    parameter: str,
    minimum: float,
    maximum: float,
    reason: str,
    unit: str | None = None,
) -> None:
    if parameter not in params:
        return
    requested = params.get(parameter)
    current = _as_float(requested)
    if current is None:
        return

    lower = float(minimum)
    upper = max(lower, float(maximum))
    effective = max(lower, min(upper, current))
    if effective != current:
        params[parameter] = float(effective)
        warnings.append(
            _build_constraint_warning(
                method_key,
                parameter,
                requested=current,
                effective=effective,
                minimum=lower,
                maximum=upper,
                reason=reason,
                unit=unit,
            )
        )
    else:
        params[parameter] = float(effective)


def _clamp_rank_interval(
    method_key: str,
    params: dict[str, Any],
    warnings: list[dict[str, Any]],
    *,
    rank_limit: int,
) -> None:
    limit = max(1, int(rank_limit))
    _clamp_int_param(
        method_key,
        params,
        warnings,
        parameter="rank_start",
        minimum=1,
        maximum=limit,
        reason="rank_limit",
    )
    rank_start = _as_int(params.get("rank_start")) or 1
    if "rank_end" not in params:
        return
    requested = params.get("rank_end")
    current = _as_int(requested)
    if current is None:
        return
    effective = max(rank_start, min(limit, current))
    if effective != current:
        params["rank_end"] = int(effective)
        warnings.append(
            _build_constraint_warning(
                method_key,
                "rank_end",
                requested=current,
                effective=effective,
                minimum=rank_start,
                maximum=limit,
                reason="rank_limit",
            )
        )
    else:
        params["rank_end"] = int(effective)


def _clamp_polyorder(
    method_key: str,
    params: dict[str, Any],
    warnings: list[dict[str, Any]],
) -> None:
    if "polyorder" not in params or "window_length" not in params:
        return
    window = _as_int(params.get("window_length"))
    if window is None:
        return
    requested = params.get("polyorder")
    current = _as_int(requested)
    if current is None:
        return
    maximum = max(0, window - 1)
    effective = max(0, min(maximum, current))
    if effective != current:
        params["polyorder"] = int(effective)
        warnings.append(
            _build_constraint_warning(
                method_key,
                "polyorder",
                requested=current,
                effective=effective,
                minimum=0,
                maximum=maximum,
                reason="savgol_polyorder_limit",
            )
        )
    else:
        params["polyorder"] = int(effective)


def _clamp_frequency_filter_params(
    method_key: str,
    params: dict[str, Any],
    warnings: list[dict[str, Any]],
    *,
    n_samples: int,
    header_info: dict[str, Any],
) -> None:
    nyquist = _nyquist_mhz(n_samples, header_info)
    if nyquist is None or nyquist <= 0.0:
        return

    filter_type = str(params.get("filter_type", "bandpass") or "bandpass").lower()
    if filter_type in {"bandpass", "highpass"}:
        _clamp_float_param(
            method_key,
            params,
            warnings,
            parameter="low_freq_mhz",
            minimum=0.0,
            maximum=nyquist,
            reason="frequency_nyquist_limit",
            unit="MHz",
        )
    if filter_type in {"bandpass", "lowpass"}:
        _clamp_float_param(
            method_key,
            params,
            warnings,
            parameter="high_freq_mhz",
            minimum=0.0,
            maximum=nyquist,
            reason="frequency_nyquist_limit",
            unit="MHz",
        )
    if filter_type == "bandpass":
        low = _as_float(params.get("low_freq_mhz")) or 0.0
        high = _as_float(params.get("high_freq_mhz")) or 0.0
        if high <= low:
            effective_low = max(0.0, high * 0.15)
            requested = low
            params["low_freq_mhz"] = float(effective_low)
            warnings.append(
                _build_constraint_warning(
                    method_key,
                    "low_freq_mhz",
                    requested=requested,
                    effective=effective_low,
                    minimum=0.0,
                    maximum=max(0.0, high),
                    reason="frequency_band_order",
                    unit="MHz",
                )
            )
    if filter_type == "notch":
        _clamp_float_param(
            method_key,
            params,
            warnings,
            parameter="notch_freq_mhz",
            minimum=0.0,
            maximum=nyquist,
            reason="frequency_nyquist_limit",
            unit="MHz",
        )
        _clamp_float_param(
            method_key,
            params,
            warnings,
            parameter="notch_width_mhz",
            minimum=0.0,
            maximum=nyquist,
            reason="frequency_nyquist_limit",
            unit="MHz",
        )
    _clamp_float_param(
        method_key,
        params,
        warnings,
        parameter="taper_ratio",
        minimum=0.0,
        maximum=0.5,
        reason="ratio_limit",
    )
    _clamp_float_param(
        method_key,
        params,
        warnings,
        parameter="notch_depth",
        minimum=0.0,
        maximum=1.0,
        reason="ratio_limit",
    )


def _build_constraint_warning(
    method_key: str,
    parameter: str,
    *,
    requested: Any,
    effective: Any,
    minimum: Any,
    maximum: Any,
    reason: str,
    unit: str | None = None,
) -> dict[str, Any]:
    details = {
        "method_id": method_key,
        "parameter": parameter,
        "requested": requested,
        "effective": effective,
        "minimum": minimum,
        "maximum": maximum,
        "reason": reason,
    }
    if unit:
        details["unit"] = unit
    return build_runtime_warning(
        "auto_tune_parameter_clamped",
        "自动选参候选参数已按当前数据尺度限制。",
        **details,
    )
