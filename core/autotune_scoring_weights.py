#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Goal-aware metric weighting for MyGPR AutoTune runners.

The weights here are intentionally modest project heuristics, not field truth.
They are used by lightweight AutoTune runners to make UI goal choices auditable
and reproducible.
"""

from __future__ import annotations

from typing import Iterable

DEFAULT_SCORING_METRICS = ("roi_retention", "residual", "cnr", "shape")
SUPPORTED_SCORING_METRICS = (*DEFAULT_SCORING_METRICS, "rmse")

GOAL_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "均衡推荐": {
        "residual": 0.30,
        "roi_retention": 0.25,
        "cnr": 0.25,
        "shape": 0.20,
        "rmse": 0.30,
    },
    "局部异常增强": {
        "residual": 0.25,
        "roi_retention": 0.30,
        "cnr": 0.30,
        "shape": 0.15,
        "rmse": 0.25,
    },
    "连续界面保留": {
        "residual": 0.20,
        "roi_retention": 0.30,
        "cnr": 0.15,
        "shape": 0.35,
        "rmse": 0.30,
    },
    "裂隙/破碎带保留": {
        "residual": 0.20,
        "roi_retention": 0.35,
        "cnr": 0.20,
        "shape": 0.25,
        "rmse": 0.30,
    },
    "深部弱反射增强": {
        "residual": 0.15,
        "roi_retention": 0.35,
        "cnr": 0.30,
        "shape": 0.20,
        "rmse": 0.25,
    },
    "滑坡基覆界面 / 潜在滑移面": {
        "residual": 0.16,
        "roi_retention": 0.30,
        "cnr": 0.14,
        "shape": 0.40,
        "rmse": 0.30,
    },
    "含水软弱带": {
        "residual": 0.18,
        "roi_retention": 0.34,
        "cnr": 0.18,
        "shape": 0.30,
        "rmse": 0.28,
    },
}

GOAL_ALIASES: dict[str, str] = {
    "balanced": "均衡推荐",
    "balance": "均衡推荐",
    "default": "均衡推荐",
    "anomaly": "局部异常增强",
    "local_anomaly": "局部异常增强",
    "interface": "连续界面保留",
    "layer": "连续界面保留",
    "landslide_interface": "滑坡基覆界面 / 潜在滑移面",
    "sliding_surface": "滑坡基覆界面 / 潜在滑移面",
    "bedrock_interface": "滑坡基覆界面 / 潜在滑移面",
    "fracture": "裂隙/破碎带保留",
    "broken_zone": "裂隙/破碎带保留",
    "wet_weak_zone": "含水软弱带",
    "water_weak_zone": "含水软弱带",
    "weak_deep": "深部弱反射增强",
    "deep_weak_reflection": "深部弱反射增强",
}


def resolve_target_goal(target_goal: str | None) -> str:
    """Resolve UI labels and stable aliases to a known goal preset."""
    raw = str(target_goal or "均衡推荐").strip()
    if raw in GOAL_WEIGHT_PRESETS:
        return raw
    return GOAL_ALIASES.get(raw.lower(), "均衡推荐")


def resolve_scoring_weights(
    *,
    target_goal: str | None,
    scoring_metrics: Iterable[str] | None,
    target_response_available: bool,
) -> tuple[str, tuple[str, ...], dict[str, float]]:
    """Return ``(goal, metrics, normalized_weights)`` for an AutoTune run."""
    goal = resolve_target_goal(target_goal)
    metrics = tuple(dict.fromkeys(scoring_metrics or DEFAULT_SCORING_METRICS))
    metrics = tuple(metric for metric in metrics if metric in SUPPORTED_SCORING_METRICS)
    if not metrics:
        metrics = DEFAULT_SCORING_METRICS
    if not target_response_available:
        metrics = tuple(metric for metric in metrics if metric != "rmse") or DEFAULT_SCORING_METRICS

    preset = GOAL_WEIGHT_PRESETS.get(goal, GOAL_WEIGHT_PRESETS["均衡推荐"])
    raw_weights = {metric: float(preset.get(metric, 0.0)) for metric in metrics}
    total = sum(raw_weights.values())
    if total <= 0.0:
        raw_weights = {metric: 1.0 for metric in metrics}
        total = float(len(metrics))
    weights = {metric: weight / total for metric, weight in raw_weights.items()}
    return goal, metrics, weights
