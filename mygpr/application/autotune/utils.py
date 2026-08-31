#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared pure helpers for the auto-tune application slice."""
from __future__ import annotations

from typing import Any

import numpy as np

_PUBLIC_RUNTIME_PARAMS = {"_low_energy_guard"}


def _trial_signature(params: dict[str, Any]) -> str:
    return str(sorted(_public_params(params).items()))


def _trim_numeric_candidates(
    values: list[Any],
    budget: int | None,
    center: float | int | None = None,
) -> list[Any]:
    cleaned: list[Any] = []
    for value in values:
        if value not in cleaned:
            cleaned.append(value)
    if budget is None or budget <= 0 or len(cleaned) <= int(budget):
        return cleaned

    ordered = sorted(cleaned, key=float)
    budget = int(max(1, budget))
    if center is None:
        if budget >= len(ordered):
            return ordered
        positions = np.linspace(0, len(ordered) - 1, num=budget)
        selected = []
        for pos in positions:
            value = ordered[int(round(float(pos)))]
            if value not in selected:
                selected.append(value)
        return sorted(selected, key=float)

    center_value = float(center)
    selected: list[Any] = []
    closest = min(ordered, key=lambda item: abs(float(item) - center_value))
    selected.append(closest)
    if budget > 1 and ordered[0] not in selected:
        selected.append(ordered[0])
    if budget > 2 and ordered[-1] not in selected:
        selected.append(ordered[-1])

    remaining = [item for item in ordered if item not in selected]
    while len(selected) < budget and remaining:
        best = max(
            remaining,
            key=lambda item: (
                min(abs(float(item) - float(chosen)) for chosen in selected),
                -abs(float(item) - center_value),
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return sorted(selected, key=float)


def _safe_ratio(numerator: float, denominator: float, floor: float = 1.0e-6) -> float:
    return float(numerator) / max(abs(float(denominator)), float(floor))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _param_distance(lhs: dict[str, Any], rhs: dict[str, Any]) -> float:
    a = _public_params(lhs.get("params", {}))
    b = _public_params(rhs.get("params", {}))
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    parts: list[float] = []
    for key in keys:
        va = a.get(key)
        vb = b.get(key)
        if _is_number(va) and _is_number(vb):
            scale = max(abs(float(va)), abs(float(vb)), 1.0)
            parts.append(min(1.5, abs(float(va) - float(vb)) / scale))
        else:
            parts.append(0.0 if va == vb else 1.0)
    return float(np.mean(parts)) if parts else 0.0


def _min_param_distance(trial: dict[str, Any], seeds: list[dict[str, Any]]) -> float:
    if not seeds:
        return 1.0
    return float(min(_param_distance(trial, seed) for seed in seeds))


def _trim_trial_candidates(
    trials: list[dict[str, Any]],
    budget: int | None,
    center_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """按“中心值 + 两端 + 分散覆盖”裁剪 trial 候选。"""
    unique_trials = _dedupe_candidates(trials)
    if budget is None or budget <= 0 or len(unique_trials) <= int(budget):
        return unique_trials

    budget = int(max(1, budget))
    center_trial = min(
        unique_trials,
        key=lambda trial: _param_distance(
            {"params": trial}, {"params": dict(center_params)}
        ),
    )
    selected = [center_trial]
    remaining = [trial for trial in unique_trials if trial is not center_trial]

    while len(selected) < budget and remaining:
        candidate = max(
            remaining,
            key=lambda trial: _min_param_distance(
                {"params": trial}, [{"params": item} for item in selected]
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)

    return selected


def _public_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in params.items()
        if not str(k).startswith("_") or str(k) in _PUBLIC_RUNTIME_PARAMS
    }


def _penalty_sum(trial: dict[str, Any]) -> float:
    penalties = trial.get("penalties", {}) or {}
    return _penalty_sum_from_dict(penalties)


def _penalty_sum_from_dict(penalties: dict[str, Any]) -> float:
    return float(sum(float(value) for value in (penalties or {}).values()))


def _effective_metrics(trial: dict[str, Any]) -> dict[str, Any]:
    if trial.get("roi_used") and trial.get("roi_metrics"):
        return trial.get("roi_metrics", {}) or {}
    return trial.get("metrics", {}) or {}


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        signature = _trial_signature(item)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
    return unique


def _merge_trials(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for trial in group:
            signature = _trial_signature(trial.get("params", {}))
            current = merged.get(signature)
            if current is None:
                merged[signature] = trial
                continue
            current_valid = bool(current.get("valid", True))
            trial_valid = bool(trial.get("valid", True))
            if trial_valid and not current_valid:
                merged[signature] = trial
                continue
            if trial_valid == current_valid and float(trial.get("score", 0.0)) > float(
                current.get("score", 0.0)
            ):
                merged[signature] = trial
    return sorted(
        merged.values(),
        key=lambda item: (
            1 if item.get("valid", True) else 0,
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )
