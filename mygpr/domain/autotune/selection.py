#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Pareto and recommendation-profile selection rules."""
from __future__ import annotations

from typing import Any

from mygpr.domain.autotune.models import PROFILE_LABELS

_PUBLIC_RUNTIME_PARAMS = {"_low_energy_guard"}


def _public_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in params.items()
        if not str(key).startswith("_") or str(key) in _PUBLIC_RUNTIME_PARAMS
    }


def _penalty_sum(trial: dict[str, Any]) -> float:
    return float(sum(float(value) for value in (trial.get("penalties", {}) or {}).values()))


def _effective_metrics(trial: dict[str, Any]) -> dict[str, Any]:
    if trial.get("roi_used") and trial.get("roi_metrics"):
        return trial.get("roi_metrics", {}) or {}
    return trial.get("metrics", {}) or {}


def _compute_pareto_front(
    family: str, trials: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compute a simple Pareto front across primary objectives and penalties."""
    if not trials:
        return []

    objectives = [_trial_objectives(family, trial) for trial in trials]
    pareto_indices: list[int] = []
    for idx, current in enumerate(objectives):
        dominated = False
        for jdx, other in enumerate(objectives):
            if idx == jdx:
                continue
            if _dominates(other, current):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(idx)

    front = [trials[idx] for idx in pareto_indices]
    front.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return front


def _trial_objectives(family: str, trial: dict[str, Any]) -> tuple[float, ...]:
    metrics = _effective_metrics(trial)
    penalty = -_penalty_sum(trial)
    if family == "zero_time":
        return (
            -float(metrics.get("pre_zero_energy_ratio", 0.0)),
            -float(metrics.get("first_break_std", 0.0)),
            float(metrics.get("first_break_sharpness", 0.0)),
            penalty,
        )
    if family == "drift":
        band = float(metrics.get("target_band_fidelity", 0.0))
        return (
            float(metrics.get("baseline_drop", 0.0)),
            float(metrics.get("low_freq_drop", 0.0)),
            band,
            penalty,
        )
    if family == "background":
        return (
            -float(metrics.get("horizontal_coherence", 0.0)),
            float(metrics.get("local_saliency_preservation", 0.0)),
            float(metrics.get("edge_preservation", 0.0)),
            penalty,
        )
    if family == "fk":
        return (
            float(metrics.get("horizontal_coherence_drop", 0.0)),
            float(metrics.get("local_saliency_fidelity", 0.0)),
            float(metrics.get("edge_fidelity", 0.0)),
            float(metrics.get("target_band_fidelity", 0.0)),
            penalty,
        )
    if family == "denoise":
        return (
            float(metrics.get("hot_pixel_drop", 0.0)),
            float(metrics.get("spikiness_drop", 0.0)),
            float(metrics.get("local_saliency_fidelity", 0.0)),
            float(metrics.get("edge_fidelity", 0.0)),
            float(metrics.get("target_band_fidelity", 0.0)),
            penalty,
        )
    if family == "gain":
        deep_effective = float(
            metrics.get("deep_gain_effective", metrics.get("deep_gain_ratio", 0.0))
        )
        return (
            -float(metrics.get("depth_rms_cv", 0.0)),
            deep_effective,
            -float(metrics.get("clipping_ratio", 0.0)),
            -float(metrics.get("hot_pixel_ratio", 0.0)),
            penalty,
        )
    if family == "impulse":
        return (
            float(metrics.get("hot_pixel_drop", 0.0)),
            float(metrics.get("spikiness_drop", 0.0)),
            float(metrics.get("edge_fidelity", 0.0)),
            penalty,
        )
    return (float(trial.get("score", 0.0)), penalty)


def _dominates(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> bool:
    return all(l >= r for l, r in zip(lhs, rhs)) and any(
        l > r for l, r in zip(lhs, rhs)
    )


def _build_profiles(
    family: str,
    trials: list[dict[str, Any]],
    pareto_trials: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    pool = (
        pareto_trials
        or sorted(trials, key=lambda item: float(item.get("score", 0.0)), reverse=True)[
            : min(8, len(trials))
        ]
    )
    used: set[int] = set()
    profiles: dict[str, dict[str, Any]] = {}
    for key in ["conservative", "balanced", "aggressive"]:
        trial = _pick_profile_trial(family, key, pool, used)
        if trial is None and trials:
            trial = max(trials, key=lambda item: float(item.get("score", 0.0)))
        if trial is None:
            continue
        used.add(id(trial))
        profiles[key] = {
            "label": PROFILE_LABELS[key],
            "params": _public_params(trial.get("params", {})),
            "score": float(trial.get("score", 0.0)),
            "metrics": dict(_effective_metrics(trial)),
            "penalties": dict(trial.get("penalties", {})),
            "reason": _profile_reason(family, key, trial),
            "stage": str(trial.get("stage", "coarse")),
        }
    return profiles


def _pick_profile_trial(
    family: str,
    profile_key: str,
    pool: list[dict[str, Any]],
    used: set[int],
) -> dict[str, Any] | None:
    available = [trial for trial in pool if id(trial) not in used] or list(pool)
    if not available:
        return None
    return max(
        available, key=lambda trial: _profile_priority(family, profile_key, trial)
    )


def _profile_priority(family: str, profile_key: str, trial: dict[str, Any]) -> float:
    score = float(trial.get("score", 0.0))
    metrics = _effective_metrics(trial)
    penalty = _penalty_sum(trial)
    if profile_key == "balanced":
        return score - 0.35 * penalty

    if family == "background":
        coherence = float(metrics.get("horizontal_coherence", 0.0))
        saliency = float(metrics.get("local_saliency_preservation", 0.0))
        edge = float(metrics.get("edge_preservation", 0.0))
        if profile_key == "conservative":
            return 2.2 * saliency + 1.8 * edge - 4.0 * penalty - coherence
        return -3.0 * coherence + 1.2 * saliency + 0.8 * edge - 1.5 * penalty

    if family == "fk":
        coherence_gain = float(metrics.get("horizontal_coherence_drop", 0.0))
        saliency_fid = float(metrics.get("local_saliency_fidelity", 0.0))
        edge_fid = float(metrics.get("edge_fidelity", 0.0))
        band_fid = float(metrics.get("target_band_fidelity", 0.0))
        if profile_key == "conservative":
            return (
                1.8 * saliency_fid
                + 1.7 * edge_fid
                + 1.6 * band_fid
                + 1.2 * coherence_gain
                - 4.0 * penalty
            )
        return (
            2.5 * coherence_gain
            + 1.2 * saliency_fid
            + 1.0 * edge_fid
            + 1.4 * band_fid
            - 1.8 * penalty
        )

    if family == "denoise":
        hot_drop = float(metrics.get("hot_pixel_drop", 0.0))
        spiky_drop = float(metrics.get("spikiness_drop", 0.0))
        edge_fid = float(metrics.get("edge_fidelity", 0.0))
        saliency_fid = float(metrics.get("local_saliency_fidelity", 0.0))
        band_fid = float(metrics.get("target_band_fidelity", 0.0))
        if profile_key == "conservative":
            return (
                1.8 * saliency_fid
                + 1.7 * edge_fid
                + 1.2 * band_fid
                + 1.5 * hot_drop
                + 1.2 * spiky_drop
                - 4.0 * penalty
            )
        return (
            2.2 * hot_drop
            + 1.8 * spiky_drop
            + 1.2 * saliency_fid
            + 0.9 * edge_fid
            + 0.9 * band_fid
            - 2.0 * penalty
        )

    if family == "gain":
        deep_effective = float(
            metrics.get("deep_gain_effective", metrics.get("deep_gain_ratio", 0.0))
        )
        clip = float(metrics.get("clipping_ratio", 0.0))
        hot = float(metrics.get("hot_pixel_ratio", 0.0))
        if profile_key == "conservative":
            return 1.6 * deep_effective - 10.0 * clip - 7.0 * hot - 4.0 * penalty
        return 3.1 * deep_effective - 4.5 * clip - 3.0 * hot - 2.0 * penalty

    if family == "drift":
        band = float(metrics.get("target_band_fidelity", 0.0))
        low_drop = float(metrics.get("low_freq_drop", 0.0))
        baseline_drop = float(metrics.get("baseline_drop", 0.0))
        if profile_key == "conservative":
            return 1.8 * band + 1.5 * low_drop + 1.4 * baseline_drop - 3.5 * penalty
        return 2.2 * low_drop + 1.8 * baseline_drop + 1.2 * band - 1.4 * penalty

    if family == "zero_time":
        sharp = float(metrics.get("first_break_sharpness", 0.0))
        pre_zero = float(metrics.get("pre_zero_energy_ratio", 0.0))
        std = float(metrics.get("first_break_std", 0.0))
        if profile_key == "conservative":
            return -pre_zero - 0.8 * std - 4.0 * penalty
        return 1.8 * sharp - pre_zero - 1.2 * penalty

    if family == "impulse":
        hot_drop = float(metrics.get("hot_pixel_drop", 0.0))
        spiky_drop = float(metrics.get("spikiness_drop", 0.0))
        edge_fid = float(metrics.get("edge_fidelity", 0.0))
        if profile_key == "conservative":
            return 1.8 * edge_fid + 1.6 * hot_drop + 1.2 * spiky_drop - 4.0 * penalty
        return 2.4 * hot_drop + 1.8 * spiky_drop + 1.1 * edge_fid - 2.0 * penalty

    return score - penalty


def _profile_reason(family: str, profile_key: str, trial: dict[str, Any]) -> str:
    base = str(trial.get("reason", ""))
    prefix = {
        "conservative": "更保守，优先压低过处理风险。",
        "balanced": "更均衡，优先综合评分与稳定性。",
        "aggressive": "更增强，优先提升主目标效果。",
    }.get(profile_key, "")
    return f"{prefix} {base}".strip()
