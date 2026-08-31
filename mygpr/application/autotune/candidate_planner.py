#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coarse candidate planning for automatic parameter tuning."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from mygpr.application.autotune.candidate_generators import (
    _build_agc_windows,
    _build_background_rank_candidates,
    _build_background_windows,
    _build_compensating_gain_candidates,
    _build_drift_windows,
    _build_fk_filter_trials,
    _build_frequency_filter_trials,
    _build_hankel_svd_trials,
    _build_impulse_windows,
    _build_sec_gain_candidates,
    _build_subspace_rank_end_candidates,
    _build_zero_time_candidates,
    _sanitize_float_candidates,
    _sanitize_int_candidates,
)
from mygpr.application.autotune.context import _get_search_plan
from mygpr.application.autotune.utils import _dedupe_candidates, _trim_numeric_candidates
from mygpr.domain.autotune.models import AutoTuneContext


def _build_candidate_trials(
    method_key: str,
    data: np.ndarray,
    base_params: dict[str, Any],
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    context: AutoTuneContext,
    method_info: dict[str, Any],
    stage: str = "coarse",
    budget: int | None = None,
) -> list[dict[str, Any]]:
    method_info = dict(method_info)
    family = method_info["auto_tune_family"]
    config = method_info.get("auto_tune_candidates", {})
    plan = _get_search_plan(context.search_mode)
    stage_budget = int(
        budget or (plan["coarse_budget"] if stage == "coarse" else plan["fine_budget"])
    )

    if family == "zero_time":
        return _build_zero_time_candidates(
            data,
            config,
            base_params,
            header_info,
            context,
            stage=stage,
            budget=stage_budget,
        )
    if family == "drift":
        values = _build_drift_windows(
            data.shape[0], context, config, stage=stage, budget=stage_budget
        )
        return [{"window": value} for value in values]
    if family == "background":
        if method_key in {"subtracting_average_2D", "median_background_2D"}:
            values = _build_background_windows(
                data.shape[1],
                context,
                config,
                base_value=base_params.get("ntraces"),
                stage=stage,
                budget=stage_budget,
            )
            return [{"ntraces": value} for value in values]
        if method_key == "svd_bg":
            values = _build_background_rank_candidates(
                data, context, config, stage=stage, budget=stage_budget
            )
            return [{"rank": value} for value in values]
    if family == "fk":
        return _build_fk_filter_trials(
            base_params,
            config,
            stage=stage,
            budget=stage_budget,
        )
    if family == "frequency":
        return _build_frequency_filter_trials(
            data,
            header_info,
            base_params,
            config,
            stage=stage,
            budget=stage_budget,
        )
    if family == "denoise":
        if method_key == "hankel_svd":
            return _build_hankel_svd_trials(
                data,
                config,
                base_params,
                stage=stage,
                budget=stage_budget,
            )
        if method_key == "svd_subspace":
            rank_start_default = config.get("rank_start", [1])
            if isinstance(rank_start_default, list):
                rank_start_default = rank_start_default[0] if rank_start_default else 1
            rank_start = int(base_params.get("rank_start", rank_start_default))
            rank_end_values = _build_subspace_rank_end_candidates(
                data,
                context,
                config,
                base_value=base_params.get("rank_end"),
                stage=stage,
                budget=stage_budget,
            )
            trials = []
            for rank_end in rank_end_values:
                if int(rank_end) >= rank_start:
                    trials.append({"rank_start": rank_start, "rank_end": int(rank_end)})
            return _dedupe_candidates(trials)
        if method_key == "wavelet_svd":
            rank_start_default = config.get("rank_start", [1])
            if isinstance(rank_start_default, list):
                rank_start_default = rank_start_default[0] if rank_start_default else 1
            rank_start = int(base_params.get("rank_start", rank_start_default))
            rank_end_values = _build_subspace_rank_end_candidates(
                data,
                context,
                config,
                base_value=base_params.get("rank_end"),
                stage=stage,
                budget=max(2, stage_budget // 2),
            )
            threshold_default = float(base_params.get("threshold", 0.05))
            threshold_values = _trim_numeric_candidates(
                _sanitize_float_candidates(
                    list(config.get("threshold", []))
                    + [
                        threshold_default * 0.7,
                        threshold_default,
                        threshold_default * 1.3,
                    ],
                    minimum=0.01,
                ),
                budget=max(2, min(3, stage_budget)),
                center=threshold_default,
            )
            levels_default = int(base_params.get("levels", 2))
            levels_values = _trim_numeric_candidates(
                _sanitize_int_candidates(
                    list(config.get("levels", []))
                    + [levels_default - 1, levels_default, levels_default + 1],
                    data.shape[0],
                    minimum=1,
                    upper=8,
                ),
                budget=max(1, min(3, stage_budget)),
                center=levels_default,
            )
            wavelet_name = str(base_params.get("wavelet", "db4"))
            trials = []
            for rank_end, threshold, levels in itertools.product(
                rank_end_values, threshold_values, levels_values
            ):
                if int(rank_end) >= rank_start:
                    trials.append(
                        {
                            "wavelet": wavelet_name,
                            "levels": int(levels),
                            "threshold": float(threshold),
                            "rank_start": rank_start,
                            "rank_end": int(rank_end),
                        }
                    )
            return _dedupe_candidates(trials)
    if family == "gain":
        if method_key == "sec_gain":
            gain_min_default = config.get("gain_min", 1.0)
            if isinstance(gain_min_default, list):
                gain_min_default = gain_min_default[0] if gain_min_default else 1.0
            gain_min = float(base_params.get("gain_min", gain_min_default))
            gain_max_values, power_values = _build_sec_gain_candidates(
                context,
                config,
                gain_min=gain_min,
                stage=stage,
                budget=stage_budget,
            )
            trials = []
            for gain_max, power in itertools.product(gain_max_values, power_values):
                if gain_max > gain_min:
                    trials.append(
                        {"gain_min": gain_min, "gain_max": gain_max, "power": power}
                    )
            return _dedupe_candidates(trials)
        if method_key == "agcGain":
            values = _build_agc_windows(
                data.shape[0], context, config, stage=stage, budget=stage_budget
            )
            guard_values = config.get("_low_energy_guard", [True])
            if not isinstance(guard_values, list):
                guard_values = [guard_values]
            trials = []
            for value, guard in itertools.product(values, guard_values):
                trials.append(
                    {
                        "window": int(value),
                        "_low_energy_guard": bool(guard),
                    }
                )
            return _dedupe_candidates(trials)
        if method_key == "compensatingGain":
            gain_min_values, gain_max_values = _build_compensating_gain_candidates(
                context, config, stage=stage, budget=stage_budget
            )
            trials = []
            for gain_min, gain_max in itertools.product(
                gain_min_values, gain_max_values
            ):
                if gain_max > gain_min:
                    trials.append({"gain_min": gain_min, "gain_max": gain_max})
            return _dedupe_candidates(trials)
        if method_key == "energy_decay_gain":
            strengths = config.get("strength", [0.5, 0.8, 1.0, 1.2])
            if not isinstance(strengths, list):
                strengths = [strengths]
            smoothing_values = config.get("smoothing_samples", [15, 31, 61, 101])
            if not isinstance(smoothing_values, list):
                smoothing_values = [smoothing_values]
            max_gain_values = config.get("max_gain", [4.0, 6.0, 8.0, 12.0])
            if not isinstance(max_gain_values, list):
                max_gain_values = [max_gain_values]
            trials = []
            for strength, smoothing, max_gain in itertools.product(
                strengths,
                smoothing_values,
                max_gain_values,
            ):
                trials.append(
                    {
                        "strength": float(strength),
                        "smoothing_samples": int(smoothing),
                        "max_gain": float(max_gain),
                    }
                )
            return _dedupe_candidates(trials)
    if family == "impulse":
        values = _build_impulse_windows(
            data.shape[1], context, config, stage=stage, budget=stage_budget
        )
        return [{"ntraces": value} for value in values]

    # Generic Cartesian-product candidate builder for methods with explicit candidate lists
    if config:
        keys = list(config.keys())
        values_lists = [config[k] for k in keys]
        trials = []
        for combo in itertools.product(*values_lists):
            trial = dict(base_params)
            trial.update({k: v for k, v in zip(keys, combo)})
            trials.append(trial)
        return _dedupe_candidates(trials)

    return []
