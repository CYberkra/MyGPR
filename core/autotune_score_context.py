#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serializable scoring context objects for AutoTune v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

from core.autotune_goal_profiles import resolve_goal_profile


@dataclass(frozen=True)
class AutoTuneScoreContext:
    """Context recorded with each AutoTune recommendation."""

    target_goal: str
    roi_mode: str = "none"
    data_mode: str = "无参考标签"
    scoring_version: str = "autotune_scoring_v2"
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_score_context(
    *,
    target_goal: str | None,
    roi_mode: str | None = "none",
    target_response_available: bool = False,
) -> AutoTuneScoreContext:
    profile = resolve_goal_profile(target_goal)
    return AutoTuneScoreContext(
        target_goal=profile.label,
        roi_mode=str(roi_mode or "none"),
        data_mode="有参考响应" if target_response_available else "无参考标签",
        weights=dict(profile.weights),
    )


def merge_score_breakdown(*, terms: Mapping[str, float], weights: Mapping[str, float], score: float) -> dict:
    return {
        "score": float(score),
        "terms": {str(k): float(v) for k, v in dict(terms or {}).items()},
        "weights": {str(k): float(v) for k, v in dict(weights or {}).items()},
    }


__all__ = ["AutoTuneScoreContext", "build_score_context", "merge_score_breakdown"]
