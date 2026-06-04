#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Goal profiles for MyGPR AutoTune scoring v2.

A goal profile is the auditable contract between the UI's target selector and
backend scoring.  It does not describe geological truth; it only determines how
candidate processing recipes are weighted for the current recommendation task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from core.autotune_score_normalization import normalize_weights
from core.autotune_scoring_weights import resolve_target_goal


@dataclass(frozen=True)
class AutoTuneGoalProfile:
    """High-level scoring profile selected by the AutoTune target goal."""

    label: str
    weights: dict[str, float]
    background_preference: str
    gain_preference: str
    bandpass_preference: str
    description: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["weights"] = dict(self.weights)
        return data


# Metric keys intentionally use engineering terms that can be reported without
# exposing research-only jargon in the main UI.
_GOAL_PROFILES: dict[str, AutoTuneGoalProfile] = {
    "均衡推荐": AutoTuneGoalProfile(
        label="均衡推荐",
        weights=normalize_weights(
            {
                "background_suppression": 0.20,
                "response_preservation": 0.20,
                "continuity": 0.16,
                "contrast": 0.16,
                "deep_balance": 0.10,
                "artifact_control": 0.12,
                "stability": 0.06,
            }
        ),
        background_preference="balanced_real_method",
        gain_preference="moderate_agc",
        bandpass_preference="auto",
        description="默认均衡权重，兼顾背景抑制、响应保留和显示可辨识度。",
    ),
    "连续界面保留": AutoTuneGoalProfile(
        label="连续界面保留",
        weights=normalize_weights(
            {
                "background_suppression": 0.15,
                "response_preservation": 0.22,
                "continuity": 0.28,
                "contrast": 0.00,
                "deep_balance": 0.12,
                "artifact_control": 0.13,
                "stability": 0.10,
            }
        ),
        background_preference="gentle_median_or_low_rank",
        gain_preference="wide_mild_agc",
        bandpass_preference="conservative",
        description="优先保护连续层状/界面响应，避免强扣背景或强平滑。",
    ),
    "滑坡基覆界面 / 潜在滑移面": AutoTuneGoalProfile(
        label="滑坡基覆界面 / 潜在滑移面",
        weights=normalize_weights(
            {
                "background_suppression": 0.13,
                "response_preservation": 0.18,
                "continuity": 0.26,
                "contrast": 0.00,
                "deep_balance": 0.22,
                "artifact_control": 0.12,
                "stability": 0.09,
            }
        ),
        background_preference="gentle_median_or_low_rank",
        gain_preference="depth_balanced_agc",
        bandpass_preference="low_frequency_preserving",
        description="偏向保留深部、连续、弱反射界面，背景抑制采用温和策略。",
    ),
    "局部异常增强": AutoTuneGoalProfile(
        label="局部异常增强",
        weights=normalize_weights(
            {
                "background_suppression": 0.22,
                "response_preservation": 0.17,
                "continuity": 0.00,
                "contrast": 0.26,
                "deep_balance": 0.09,
                "artifact_control": 0.14,
                "stability": 0.12,
            }
        ),
        background_preference="stronger_mean_or_svd",
        gain_preference="moderate_contrast_gain",
        bandpass_preference="wider_bandpass",
        description="偏向局部高响应/双曲线类目标，但保留伪影控制。",
    ),
    "裂隙/破碎带保留": AutoTuneGoalProfile(
        label="裂隙/破碎带保留",
        weights=normalize_weights(
            {
                "background_suppression": 0.17,
                "response_preservation": 0.16,
                "continuity": 0.20,
                "texture_preservation": 0.24,
                "artifact_control": 0.13,
                "stability": 0.10,
            }
        ),
        background_preference="gentle_texture_preserving",
        gain_preference="mild_agc",
        bandpass_preference="texture_preserving",
        description="保留纹理和断续散射响应，避免过度平滑。",
    ),
    "含水软弱带": AutoTuneGoalProfile(
        label="含水软弱带",
        weights=normalize_weights(
            {
                "background_suppression": 0.14,
                "response_preservation": 0.24,
                "continuity": 0.20,
                "deep_balance": 0.18,
                "artifact_control": 0.14,
                "stability": 0.10,
            }
        ),
        background_preference="gentle_low_frequency_preserving",
        gain_preference="moderate_stable_gain",
        bandpass_preference="low_frequency_preserving",
        description="保留衰减、弱反射和带状连续响应，不鼓励强高通锐化。",
    ),
    "深部弱反射增强": AutoTuneGoalProfile(
        label="深部弱反射增强",
        weights=normalize_weights(
            {
                "background_suppression": 0.13,
                "response_preservation": 0.17,
                "continuity": 0.00,
                "contrast": 0.00,
                "deep_balance": 0.28,
                "gain_stability": 0.20,
                "artifact_control": 0.14,
                "stability": 0.08,
            }
        ),
        background_preference="gentle_median_or_low_rank",
        gain_preference="wide_stable_agc",
        bandpass_preference="deep_weak_reflection",
        description="优先保护深部弱响应，避免把深部噪声放大成假响应。",
    ),
}


def resolve_goal_profile(target_goal: str | None) -> AutoTuneGoalProfile:
    """Return the canonical scoring v2 profile for a UI label or alias."""
    label = resolve_target_goal(target_goal)
    return _GOAL_PROFILES.get(label, _GOAL_PROFILES["均衡推荐"])


def goal_profile_table() -> dict[str, dict]:
    """Return all profiles for reports/tests."""
    return {label: profile.to_dict() for label, profile in _GOAL_PROFILES.items()}


def profile_weights(target_goal: str | None) -> dict[str, float]:
    return dict(resolve_goal_profile(target_goal).weights)


__all__ = [
    "AutoTuneGoalProfile",
    "goal_profile_table",
    "profile_weights",
    "resolve_goal_profile",
]
