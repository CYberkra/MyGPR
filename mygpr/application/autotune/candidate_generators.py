#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Method-family candidate generators for automatic parameter tuning."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from mygpr.domain.autotune.data_context import frequency_band_from_context
from mygpr.domain.autotune.quality_metrics import detect_first_break_indices, median_first_break
from mygpr.domain.common.scalars import to_float, to_float_or_none, to_int_or_none
from mygpr.application.autotune.utils import (
    _dedupe_candidates,
    _trim_numeric_candidates,
    _trim_trial_candidates,
)
from mygpr.domain.autotune.models import AutoTuneContext


def _build_zero_time_candidates(
    data: np.ndarray,
    config: dict[str, Any],
    base_params: dict[str, Any],
    header_info: dict[str, Any],
    context: AutoTuneContext,
    stage: str = "coarse",
    budget: int = 8,
) -> list[dict[str, Any]]:
    n_samples = int(data.shape[0])
    time_step_ns = _resolve_time_step_ns(n_samples, header_info)
    search_ratio = float(config.get("search_ratio", 0.35))
    detectors = config.get("detectors", ["threshold", "peak", "first_break"])
    base_threshold = float(
        np.clip(0.02 + 0.02 * context.features.get("first_break_std", 0.0), 0.02, 0.14)
    )
    thresholds = _sanitize_float_candidates(
        list(config.get("thresholds", []))
        + [base_threshold * s for s in [0.75, 1.0, 1.25, 1.5]],
        minimum=0.001,
    )[: max(3, min(len(config.get("thresholds", [])) + 4, budget))]
    base_backup = max(1, int(round(context.features.get("first_break_std", 0.0) / 2.0)))
    backups = _sanitize_int_candidates(
        list(config.get("backup_samples", []))
        + [base_backup, base_backup + 2, base_backup + 4],
        n_samples,
        minimum=0,
        upper=max(1, n_samples - 1),
    )
    if stage == "coarse":
        detectors = list(detectors)[: min(len(detectors), 3)]
        thresholds = thresholds[: min(len(thresholds), 4)]
        backups = backups[: min(len(backups), 3)]
    else:
        thresholds = thresholds[: min(len(thresholds), 5)]
        backups = backups[: min(len(backups), 4)]

    trials: list[dict[str, Any]] = []
    seen: set[tuple[float, str, int, float]] = set()
    for detector, threshold, backup in itertools.product(
        detectors, thresholds, backups
    ):
        fb_idx = detect_first_break_indices(
            data,
            method=str(detector),
            threshold=float(threshold),
            search_ratio=search_ratio,
        )
        zero_idx = max(0, median_first_break(fb_idx) - int(backup))
        new_zero_time = float(zero_idx) * time_step_ns
        key = (
            round(new_zero_time, 6),
            str(detector),
            int(backup),
            round(float(threshold), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        trials.append(
            {
                "new_zero_time": new_zero_time,
                "_detector": str(detector),
                "_threshold": float(threshold),
                "_backup_samples": int(backup),
                "_zero_idx": int(zero_idx),
                "_first_break_std_before": float(np.std(fb_idx)),
            }
        )

    fallback = float(base_params.get("new_zero_time", 5.0))
    fallback_key = (round(fallback, 6), "manual", 0, 0.0)
    if fallback_key not in seen:
        trials.append(
            {
                "new_zero_time": fallback,
                "_detector": "manual",
                "_threshold": 0.0,
                "_backup_samples": 0,
                "_zero_idx": int(round(fallback / max(time_step_ns, 1.0e-6))),
            }
        )
    return _dedupe_candidates(trials)


def _resolve_time_step_ns(n_samples: int, header_info: dict[str, Any]) -> float:
    total_time_ns = header_info.get("total_time_ns") if header_info else None
    total_time_value = to_float(total_time_ns, default=0.0)
    if total_time_value > 0:
        return total_time_value / max(1, int(n_samples))
    return 48.0 / max(1, int(n_samples))


def _agc_window_min(n_samples: int, header_info: dict[str, Any]) -> int:
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


def _sanitize_int_candidates(
    values: list[Any],
    data_limit: int,
    minimum: int,
    upper: int,
) -> list[int]:
    cleaned: list[int] = []
    for value in values:
        current = to_int_or_none(value)
        if current is None:
            continue
        current = max(int(minimum), min(int(upper), current))
        if current not in cleaned:
            cleaned.append(current)
    if not cleaned:
        cleaned = [max(int(minimum), min(int(upper), max(1, data_limit // 8 or 1)))]
    return cleaned


def _adaptive_trace_windows(
    n_traces: int,
    configured_values: list[Any],
    base_value: Any | None,
    minimum: int,
    upper: int,
) -> list[int]:
    """Build trace-window candidates using configured values plus data-adaptive ratios."""
    ratio_values = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5, 0.8]
    adaptive = [max(minimum, int(round(n_traces * ratio))) for ratio in ratio_values]
    if base_value is not None:
        base_int = to_int_or_none(base_value)
        if base_int is not None:
            adaptive.extend(
                [max(minimum, base_int - 20), base_int, min(upper, base_int + 20)]
            )

    values = _sanitize_int_candidates(
        list(configured_values) + adaptive,
        n_traces,
        minimum=minimum,
        upper=upper,
    )
    return values


def _sanitize_float_candidates(values: list[Any], minimum: float) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        current = to_float_or_none(value)
        if current is None:
            continue
        current = max(float(minimum), current)
        if current not in cleaned:
            cleaned.append(current)
    return cleaned or [float(minimum)]


def _build_drift_windows(
    n_samples: int,
    context: AutoTuneContext,
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[int]:
    low_freq = float(context.features.get("low_freq_ratio", 0.1))
    base_window = int(round(n_samples * (0.05 + 0.30 * low_freq)))
    base_window = max(8, min(max(16, n_samples // 2), base_window))
    multipliers = (
        [0.55, 0.8, 1.0, 1.25, 1.6, 2.0]
        if stage == "coarse"
        else [0.7, 0.85, 1.0, 1.15, 1.3]
    )
    values = [int(round(base_window * scale)) for scale in multipliers]
    values = _sanitize_int_candidates(
        list(config.get("window", [])) + values,
        n_samples,
        minimum=8,
        upper=max(16, n_samples // 2),
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=base_window)
    ]


def _build_background_windows(
    n_traces: int,
    context: AutoTuneContext,
    config: dict[str, Any],
    base_value: Any | None,
    stage: str,
    budget: int | None = None,
) -> list[int]:
    corr_length = max(2, int(context.features.get("lateral_corr_length", 6)))
    base_window = max(5, min(n_traces, int(round(corr_length * 4.0))))
    multipliers = (
        [0.6, 1.0, 1.5, 2.0, 3.0, 4.0]
        if stage == "coarse"
        else [0.75, 0.9, 1.0, 1.1, 1.25]
    )
    adaptive = [int(round(base_window * scale)) for scale in multipliers]
    values = _sanitize_int_candidates(
        list(config.get("ntraces", []))
        + adaptive
        + ([base_value] if base_value is not None else []),
        n_traces,
        minimum=3,
        upper=max(3, n_traces),
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=base_window)
    ]


def _build_background_rank_candidates(
    data: np.ndarray,
    context: AutoTuneContext,
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[int]:
    rank_limit = max(1, min(data.shape) - 1)
    elbow = max(1, min(rank_limit, int(context.features.get("singular_elbow_rank", 2))))
    values = [
        1,
        max(1, elbow - 1),
        elbow,
        min(rank_limit, elbow + 1),
        min(rank_limit, elbow + 2),
    ]
    if stage == "coarse":
        values.append(min(rank_limit, elbow * 2))
    values = _sanitize_int_candidates(
        list(config.get("rank", [])) + values,
        rank_limit,
        minimum=1,
        upper=rank_limit,
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=elbow)
    ]


def _build_fk_filter_trials(
    base_params: dict[str, Any],
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    low_default = int(base_params.get("angle_low", 12))
    high_default = int(base_params.get("angle_high", 55))
    taper_default = int(base_params.get("taper_width", 4))

    if stage == "coarse":
        low_values = _sanitize_int_candidates(
            list(config.get("angle_low", [])) + [low_default],
            90,
            minimum=0,
            upper=80,
        )
        high_values = _sanitize_int_candidates(
            list(config.get("angle_high", [])) + [high_default],
            90,
            minimum=10,
            upper=90,
        )
        taper_values = _sanitize_int_candidates(
            list(config.get("taper_width", [])) + [taper_default],
            20,
            minimum=0,
            upper=20,
        )
    else:
        low_values = [low_default]
        high_values = [high_default]
        taper_values = [taper_default]

    trials = []
    for angle_low, angle_high, taper_width in itertools.product(
        low_values, high_values, taper_values
    ):
        if int(angle_high) - int(angle_low) >= 8:
            trials.append(
                {
                    "angle_low": int(angle_low),
                    "angle_high": int(angle_high),
                    "taper_width": int(taper_width),
                }
            )

    return _trim_trial_candidates(
        trials,
        budget=budget,
        center_params={
            "angle_low": low_default,
            "angle_high": high_default,
            "taper_width": taper_default,
        },
    )


def _build_frequency_filter_trials(
    data: np.ndarray,
    header_info: dict[str, Any],
    base_params: dict[str, Any],
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    filter_types = list(config.get("filter_type", [])) or [
        str(base_params.get("filter_type", "bandpass"))
    ]
    context_band = frequency_band_from_context(header_info)
    nyquist_mhz = _resolve_nyquist_mhz(data.shape[0], header_info)
    if nyquist_mhz is None or nyquist_mhz <= 0.0:
        high_values = list(config.get("high_freq_mhz", [])) or [
            float(base_params.get("high_freq_mhz", 800.0))
        ]
    else:
        ratios = list(config.get("high_freq_ratio", [])) or [0.75, 0.85, 0.95]
        high_values = [
            max(0.0, min(float(nyquist_mhz), float(nyquist_mhz) * float(ratio)))
            for ratio in ratios
        ]
        high_values.extend(config.get("high_freq_mhz", []))

    has_explicit_band = (
        "low_freq_mhz" in base_params or "high_freq_mhz" in base_params
    )
    if context_band is not None and not has_explicit_band:
        low_default, high_default = context_band
    else:
        low_default = float(base_params.get("low_freq_mhz", 10.0))
        high_default = float(
            base_params.get("high_freq_mhz", high_values[0] if high_values else 800.0)
        )
    taper_default = float(base_params.get("taper_ratio", 0.08))

    if stage == "coarse":
        low_values = _trim_numeric_candidates(
            _sanitize_float_candidates(
                list(config.get("low_freq_mhz", []))
                + [low_default, low_default * 0.5, low_default * 1.5]
                + ([context_band[0]] if context_band is not None else []),
                minimum=0.0,
            ),
            budget=max(2, min(4, int(budget or 6))),
            center=low_default,
        )
        high_values = _trim_numeric_candidates(
            _sanitize_float_candidates(
                high_values
                + [high_default]
                + ([context_band[1]] if context_band is not None else []),
                minimum=0.0,
            ),
            budget=max(2, min(4, int(budget or 6))),
            center=high_default,
        )
        taper_values = _trim_numeric_candidates(
            _sanitize_float_candidates(
                list(config.get("taper_ratio", []))
                + [taper_default, taper_default * 0.5, taper_default * 1.5],
                minimum=0.0,
            ),
            budget=max(1, min(3, int(budget or 6))),
            center=taper_default,
        )
    else:
        low_values = [low_default]
        high_values = [high_default]
        taper_values = [taper_default]

    trials = []
    for filter_type, low_freq_mhz, high_freq_mhz, taper_ratio in itertools.product(
        filter_types, low_values, high_values, taper_values
    ):
        if str(filter_type) == "bandpass" and float(high_freq_mhz) <= float(low_freq_mhz):
            continue
        trials.append(
            {
                "filter_type": str(filter_type),
                "low_freq_mhz": float(low_freq_mhz),
                "high_freq_mhz": float(high_freq_mhz),
                "taper_ratio": float(taper_ratio),
            }
        )

    return _trim_trial_candidates(
        trials,
        budget=budget,
        center_params={
            "low_freq_mhz": low_default,
            "high_freq_mhz": high_default,
            "taper_ratio": taper_default,
        },
    )


def _resolve_nyquist_mhz(n_samples: int, header_info: dict[str, Any]) -> float | None:
    sample_rate_hz = to_float(header_info.get("sample_rate_hz"), default=0.0)
    if sample_rate_hz > 0.0:
        return float(sample_rate_hz / 2.0e6)

    time_step_s = to_float(header_info.get("time_step_s"), default=0.0)
    if time_step_s > 0.0:
        return float(1.0 / time_step_s / 2.0e6)

    total_time_ns = to_float(header_info.get("total_time_ns"), default=0.0)
    if total_time_ns > 0.0:
        return float(max(1, int(n_samples)) / (total_time_ns * 1.0e-9) / 2.0e6)
    return None


def _build_subspace_rank_end_candidates(
    data: np.ndarray,
    context: AutoTuneContext,
    config: dict[str, Any],
    base_value: Any | None,
    stage: str,
    budget: int | None = None,
) -> list[int]:
    rank_limit = max(2, min(data.shape))
    elbow = max(2, min(rank_limit, int(context.features.get("singular_elbow_rank", 4))))
    base_rank = max(
        4, min(rank_limit, int(base_value) if base_value is not None else elbow * 3)
    )
    values = [
        max(4, elbow),
        max(6, elbow + 2),
        max(8, elbow * 2),
        base_rank,
        min(rank_limit, max(base_rank + 4, elbow * 3)),
        min(rank_limit, max(base_rank + 8, elbow * 4)),
    ]
    if stage == "fine":
        values.extend(
            [
                int(round(base_rank * 0.85)),
                int(round(base_rank * 0.95)),
                int(round(base_rank * 1.05)),
                int(round(base_rank * 1.15)),
            ]
        )
    values = _sanitize_int_candidates(
        list(config.get("rank_end", [])) + values,
        rank_limit,
        minimum=2,
        upper=rank_limit,
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=base_rank)
    ]


def _build_hankel_svd_trials(
    data: np.ndarray,
    config: dict[str, Any],
    base_params: dict[str, Any],
    stage: str,
    budget: int,
) -> list[dict[str, Any]]:
    """Build a tiny bounded Hankel candidate set around internal auto modes."""
    if stage != "coarse":
        return []

    n_samples = int(data.shape[0])
    fixed_window = _resolve_hankel_fixed_window(
        n_samples,
        base_params.get("window_length"),
        config.get("window_length", []),
    )
    fixed_rank = _resolve_hankel_fixed_rank(
        n_samples,
        base_params.get("rank"),
        config.get("rank", []),
    )
    candidates = _dedupe_candidates(
        [
            {"window_length": 0, "rank": 0},
            {"window_length": int(fixed_window), "rank": 0},
            {"window_length": 0, "rank": int(fixed_rank)},
            {"window_length": int(fixed_window), "rank": int(fixed_rank)},
        ]
    )
    return candidates[: max(1, min(int(budget), 4))]


def _resolve_hankel_fixed_window(
    n_samples: int,
    requested_value: Any,
    configured_values: Any,
) -> int:
    """Resolve a safe fixed Hankel window without the broad external grid."""
    upper = max(1, int(n_samples) - 1)
    fallback = max(1, min(upper, max(8, int(round(max(int(n_samples), 1) * 0.25)))))
    preferred_values: list[Any] = [requested_value]
    if isinstance(configured_values, list):
        preferred_values.extend(configured_values)
    else:
        preferred_values.append(configured_values)
    for value in preferred_values:
        if value is None:
            continue
        current = int(value)
        if current > 0:
            return max(1, min(current, upper))
    return fallback


def _resolve_hankel_fixed_rank(
    n_samples: int,
    requested_value: Any,
    configured_values: Any,
) -> int:
    """Resolve a safe fixed Hankel rank while reserving zero for internal auto-select."""
    upper = max(1, min(10, int(n_samples) - 1))
    fallback = max(1, min(upper, 5))
    preferred_values: list[Any] = [requested_value]
    if isinstance(configured_values, list):
        preferred_values.extend(configured_values)
    else:
        preferred_values.append(configured_values)
    for value in preferred_values:
        if value is None:
            continue
        current = int(value)
        if current > 0:
            return max(1, min(current, upper))
    return fallback


def _build_sec_gain_candidates(
    context: AutoTuneContext,
    config: dict[str, Any],
    gain_min: float,
    stage: str,
    budget: int | None = None,
) -> tuple[list[float], list[float]]:
    attenuation_ratio = float(context.features.get("attenuation_ratio", 1.8))
    base_gain_max = np.clip(
        2.2 + 1.6 * np.log1p(max(0.0, attenuation_ratio - 1.0)), 2.5, 12.0
    )
    base_power = np.clip(
        0.55 + 0.32 * np.log1p(max(0.0, attenuation_ratio - 1.0)), 0.5, 2.2
    )
    gain_scales = (
        [0.65, 0.85, 1.0, 1.2, 1.45]
        if stage == "coarse"
        else [0.85, 0.95, 1.0, 1.08, 1.18]
    )
    power_scales = (
        [0.7, 0.9, 1.0, 1.15, 1.35]
        if stage == "coarse"
        else [0.85, 0.95, 1.0, 1.08, 1.18]
    )
    dim_budget = max(
        3,
        min(
            5,
            int(np.ceil(np.sqrt(max(4, float(budget or 8) * 1.6)))),
        ),
    )
    gain_values = _trim_numeric_candidates(
        _sanitize_float_candidates(
            list(config.get("gain_max", [])) + [base_gain_max * s for s in gain_scales],
            minimum=gain_min,
        ),
        budget=dim_budget,
        center=base_gain_max,
    )
    power_values = _trim_numeric_candidates(
        _sanitize_float_candidates(
            list(config.get("power", [])) + [base_power * s for s in power_scales],
            minimum=0.2,
        ),
        budget=dim_budget,
        center=base_power,
    )
    return [float(value) for value in gain_values], [
        float(value) for value in power_values
    ]


def _build_agc_windows(
    n_samples: int,
    context: AutoTuneContext,
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[int]:
    attenuation_ratio = float(context.features.get("attenuation_ratio", 1.5))
    base_window = int(
        round(n_samples * np.clip(0.035 + 0.015 * attenuation_ratio, 0.03, 0.18))
    )
    min_window = _agc_window_min(n_samples, context.header_info)
    base_window = max(min_window, base_window)
    values = [
        int(round(base_window * scale))
        for scale in (
            [0.6, 0.85, 1.0, 1.25, 1.6]
            if stage == "coarse"
            else [0.8, 0.9, 1.0, 1.1, 1.25]
        )
    ]
    values = _sanitize_int_candidates(
        list(config.get("window", [])) + values,
        n_samples,
        minimum=min_window,
        upper=n_samples,
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=base_window)
    ]


def _build_compensating_gain_candidates(
    context: AutoTuneContext,
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> tuple[list[float], list[float]]:
    attenuation_ratio = float(context.features.get("attenuation_ratio", 1.6))
    base_max = np.clip(
        2.5 + 1.4 * np.log1p(max(0.0, attenuation_ratio - 1.0)), 2.0, 10.0
    )
    max_budget = 4 if stage == "coarse" else 3
    min_budget = 3
    max_values = _trim_numeric_candidates(
        _sanitize_float_candidates(
            list(config.get("gain_max", []))
            + [
                base_max * s
                for s in (
                    [0.7, 0.9, 1.0, 1.2, 1.4]
                    if stage == "coarse"
                    else [0.85, 0.95, 1.0, 1.1, 1.2]
                )
            ],
            minimum=0.2,
        ),
        budget=max_budget,
        center=base_max,
    )
    min_values = _trim_numeric_candidates(
        _sanitize_float_candidates(
            list(config.get("gain_min", [])) + [0.8, 1.0, 1.2], minimum=0.1
        ),
        budget=min_budget,
        center=1.0,
    )
    return [float(value) for value in min_values], [
        float(value) for value in max_values
    ]


def _build_impulse_windows(
    n_traces: int,
    context: AutoTuneContext,
    config: dict[str, Any],
    stage: str,
    budget: int | None = None,
) -> list[int]:
    spiky = float(context.features.get("spikiness", 0.0))
    hot = float(context.features.get("hot_pixel_ratio", 0.0))
    severity = spiky + 8.0 * hot
    base_window = (
        3 if severity < 0.5 else 5 if severity < 1.5 else 7 if severity < 3.0 else 9
    )
    values = (
        [base_window - 2, base_window, base_window + 2]
        if stage == "coarse"
        else [
            base_window - 2,
            base_window - 1,
            base_window,
            base_window + 1,
            base_window + 2,
        ]
    )
    values = _sanitize_int_candidates(
        list(config.get("ntraces", [])) + values,
        n_traces,
        minimum=3,
        upper=max(3, n_traces),
    )
    return [
        int(value)
        for value in _trim_numeric_candidates(values, budget=budget, center=base_window)
    ]
