#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-neighbourhood candidate refinement."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from mygpr.application.autotune.candidate_generators import (
    _agc_window_min,
    _resolve_time_step_ns,
    _sanitize_float_candidates,
    _sanitize_int_candidates,
)
from mygpr.application.autotune.context import _get_search_plan
from mygpr.application.autotune.utils import (
    _dedupe_candidates,
    _trim_numeric_candidates,
    _trim_trial_candidates,
)
from mygpr.domain.autotune.models import AutoTuneContext


def _refine_candidate_trials(
    method_key: str,
    data: np.ndarray,
    base_params: dict[str, Any],
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    context: AutoTuneContext,
    seed_trials: list[dict[str, Any]],
    method_info: dict[str, Any],
) -> list[dict[str, Any]]:
    if not seed_trials:
        return []
    method_info = dict(method_info)
    family = method_info["auto_tune_family"]
    plan = _get_search_plan(context.search_mode)
    refined: list[dict[str, Any]] = []
    n_samples, n_traces = int(data.shape[0]), int(data.shape[1])

    if method_key == "hankel_svd":
        return []

    for seed_rank, trial in enumerate(seed_trials, start=1):
        params = trial.get("params", {})
        if family == "background" and "ntraces" in params:
            center = int(params["ntraces"])
            values = _sanitize_int_candidates(
                [
                    int(round(center * 0.70)),
                    int(round(center * 0.85)),
                    center,
                    int(round(center * 1.15)),
                    int(round(center * 1.35)),
                    center - 4,
                    center + 4,
                ],
                n_traces,
                minimum=3,
                upper=max(3, n_traces),
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            for value in values:
                refined.append({"ntraces": int(value), "_seed_rank": seed_rank})
        elif family == "drift" and "window" in params:
            center = int(params["window"])
            values = _sanitize_int_candidates(
                [
                    int(round(center * 0.70)),
                    int(round(center * 0.85)),
                    center,
                    int(round(center * 1.15)),
                    int(round(center * 1.30)),
                ],
                n_samples,
                minimum=8,
                upper=max(16, n_samples // 2),
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            low_energy_guard = bool(params.get("_low_energy_guard", True))
            for value in values:
                refined.append(
                    {
                        "window": int(value),
                        "_low_energy_guard": low_energy_guard,
                        "_seed_rank": seed_rank,
                    }
                )
        elif family == "background" and method_key == "svd_bg" and "rank" in params:
            center = int(params["rank"])
            values = _sanitize_int_candidates(
                [center - 2, center - 1, center, center + 1, center + 2],
                max(1, min(data.shape) - 1),
                minimum=1,
                upper=max(1, min(data.shape) - 1),
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            for value in values:
                refined.append({"rank": int(value), "_seed_rank": seed_rank})
        elif family == "fk" and "angle_low" in params and "angle_high" in params:
            center_low = int(params["angle_low"])
            center_high = int(params["angle_high"])
            center_taper = int(params.get("taper_width", 4))
            low_values = _trim_numeric_candidates(
                _sanitize_int_candidates(
                    [
                        center_low - 4,
                        center_low - 2,
                        center_low,
                        center_low + 2,
                        center_low + 4,
                    ],
                    90,
                    minimum=0,
                    upper=80,
                ),
                budget=max(2, plan["fine_budget"] // 2),
                center=center_low,
            )
            high_values = _trim_numeric_candidates(
                _sanitize_int_candidates(
                    [
                        center_high - 6,
                        center_high - 3,
                        center_high,
                        center_high + 3,
                        center_high + 6,
                    ],
                    90,
                    minimum=10,
                    upper=90,
                ),
                budget=max(2, plan["fine_budget"] // 2),
                center=center_high,
            )
            taper_values = _trim_numeric_candidates(
                _sanitize_int_candidates(
                    [
                        center_taper - 2,
                        center_taper - 1,
                        center_taper,
                        center_taper + 1,
                        center_taper + 2,
                    ],
                    20,
                    minimum=0,
                    upper=20,
                ),
                budget=max(2, min(3, plan["fine_budget"])),
                center=center_taper,
            )
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
                            "_seed_rank": seed_rank,
                        }
                    )
            refined.extend(
                _trim_trial_candidates(
                    trials,
                    budget=plan["fine_budget"],
                    center_params={
                        "angle_low": center_low,
                        "angle_high": center_high,
                        "taper_width": center_taper,
                    },
                )
            )
        elif (
            family == "denoise"
            and method_key == "svd_subspace"
            and "rank_end" in params
        ):
            center = int(params["rank_end"])
            rank_start = int(params.get("rank_start", 1))
            rank_limit = max(rank_start, min(data.shape))
            values = _sanitize_int_candidates(
                [
                    int(round(center * 0.75)),
                    int(round(center * 0.90)),
                    center,
                    int(round(center * 1.10)),
                    int(round(center * 1.25)),
                ],
                rank_limit,
                minimum=rank_start,
                upper=rank_limit,
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            for value in values:
                if int(value) >= rank_start:
                    refined.append(
                        {
                            "rank_start": rank_start,
                            "rank_end": int(value),
                            "_seed_rank": seed_rank,
                        }
                    )
        elif (
            family == "denoise" and method_key == "wavelet_svd" and "rank_end" in params
        ):
            center = int(params["rank_end"])
            rank_start = int(params.get("rank_start", 1))
            rank_limit = max(rank_start, min(data.shape))
            rank_values = _sanitize_int_candidates(
                [
                    int(round(center * 0.80)),
                    int(round(center * 0.90)),
                    center,
                    int(round(center * 1.10)),
                    int(round(center * 1.20)),
                ],
                rank_limit,
                minimum=rank_start,
                upper=rank_limit,
            )
            rank_values = _trim_numeric_candidates(
                rank_values, budget=max(2, plan["fine_budget"] // 2), center=center
            )
            threshold_center = float(params.get("threshold", 0.05))
            threshold_values = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    [
                        threshold_center * 0.8,
                        threshold_center * 0.95,
                        threshold_center,
                        threshold_center * 1.1,
                        threshold_center * 1.25,
                    ],
                    minimum=0.01,
                ),
                budget=max(2, min(3, plan["fine_budget"])),
                center=threshold_center,
            )
            levels_center = int(params.get("levels", 2))
            levels_values = _trim_numeric_candidates(
                _sanitize_int_candidates(
                    [levels_center - 1, levels_center, levels_center + 1],
                    data.shape[0],
                    minimum=1,
                    upper=8,
                ),
                budget=max(1, min(3, plan["fine_budget"])),
                center=levels_center,
            )
            wavelet_name = str(params.get("wavelet", "db4"))
            for rank_end, threshold, levels in itertools.product(
                rank_values, threshold_values, levels_values
            ):
                if int(rank_end) >= rank_start:
                    refined.append(
                        {
                            "wavelet": wavelet_name,
                            "levels": int(levels),
                            "threshold": float(threshold),
                            "rank_start": rank_start,
                            "rank_end": int(rank_end),
                            "_seed_rank": seed_rank,
                        }
                    )
        elif family == "gain" and method_key == "sec_gain":
            center_gain = float(params.get("gain_max", 5.0))
            center_power = float(params.get("power", 1.0))
            gain_span = max(0.40, center_gain * 0.18)
            power_span = max(0.06, center_power * 0.15)
            dim_budget = max(
                3,
                min(4, int(np.ceil(np.sqrt(max(4, plan["fine_budget"] * 2))))),
            )
            gain_candidates = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    [
                        center_gain - gain_span,
                        center_gain - gain_span * 0.5,
                        center_gain,
                        center_gain + gain_span * 0.5,
                        center_gain + gain_span,
                    ],
                    minimum=float(
                        base_params.get("gain_min", params.get("gain_min", 1.0))
                    ),
                ),
                budget=dim_budget,
                center=center_gain,
            )
            power_candidates = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    [
                        center_power - power_span,
                        center_power - power_span * 0.5,
                        center_power,
                        center_power + power_span * 0.5,
                        center_power + power_span,
                    ],
                    minimum=0.2,
                ),
                budget=dim_budget,
                center=center_power,
            )
            for gain_value, power_value in itertools.product(
                gain_candidates, power_candidates
            ):
                refined.append(
                    {
                        "gain_min": float(
                            base_params.get("gain_min", params.get("gain_min", 1.0))
                        ),
                        "gain_max": max(1.0, float(gain_value)),
                        "power": max(0.2, float(power_value)),
                        "_seed_rank": seed_rank,
                    }
                )
        elif family == "gain" and method_key == "agcGain" and "window" in params:
            center = int(params["window"])
            min_window = _agc_window_min(n_samples, header_info)
            values = _sanitize_int_candidates(
                [
                    int(round(center * 0.80)),
                    int(round(center * 0.90)),
                    center,
                    int(round(center * 1.10)),
                    int(round(center * 1.25)),
                ],
                n_samples,
                minimum=min_window,
                upper=n_samples,
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            low_energy_guard = bool(params.get("_low_energy_guard", True))
            for value in values:
                refined.append(
                    {
                        "window": int(value),
                        "_low_energy_guard": low_energy_guard,
                        "_seed_rank": seed_rank,
                    }
                )
        elif family == "gain" and method_key == "compensatingGain":
            center_min = float(params.get("gain_min", 1.0))
            center_max = float(params.get("gain_max", 5.0))
            min_values = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    [center_min - 0.2, center_min, center_min + 0.2], minimum=0.1
                ),
                budget=3,
                center=center_min,
            )
            max_span = max(0.4, center_max * 0.12)
            max_values = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    [
                        center_max - max_span,
                        center_max - max_span * 0.5,
                        center_max,
                        center_max + max_span * 0.5,
                        center_max + max_span,
                    ],
                    minimum=0.2,
                ),
                budget=max(3, min(4, plan["fine_budget"])),
                center=center_max,
            )
            for gain_min, gain_max in itertools.product(min_values, max_values):
                if float(gain_max) > float(gain_min):
                    refined.append(
                        {
                            "gain_min": float(gain_min),
                            "gain_max": float(gain_max),
                            "_seed_rank": seed_rank,
                        }
                    )
        elif family == "zero_time":
            time_step = _resolve_time_step_ns(data.shape[0], header_info)
            center_idx = int(params.get("_zero_idx", 0))
            detector = str(params.get("_detector", "threshold"))
            threshold = float(params.get("_threshold", 0.05) or 0.05)
            backup = int(params.get("_backup_samples", 0))
            for delta in [0, -1, 1, -2, 2, -4, 4][: plan["fine_budget"]]:
                zero_idx = max(0, center_idx + delta)
                refined.append(
                    {
                        "new_zero_time": float(zero_idx) * time_step,
                        "_detector": detector,
                        "_threshold": max(0.001, threshold),
                        "_backup_samples": backup,
                        "_zero_idx": zero_idx,
                        "_seed_rank": seed_rank,
                    }
                )
        elif family == "impulse" and "ntraces" in params:
            center = int(params["ntraces"])
            values = _sanitize_int_candidates(
                [center - 2, center - 1, center, center + 1, center + 2],
                n_traces,
                minimum=3,
                upper=max(3, n_traces),
            )
            values = _trim_numeric_candidates(
                values, budget=plan["fine_budget"], center=center
            )
            for value in values:
                refined.append({"ntraces": int(value), "_seed_rank": seed_rank})

    return _dedupe_candidates(refined)
