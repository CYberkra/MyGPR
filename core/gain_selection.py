#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable gain-method selection rules for MyGPR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


GAIN_METHOD_LABELS = {
    "sec_gain": "SEC 深度补偿",
    "agcGain": "AGC 自动增益",
    "compensatingGain": "线性/手动 TGC",
    "no_gain": "不施加增益",
}


@dataclass(frozen=True)
class GainCandidateDecision:
    """Decision record for one gain candidate selected from report branches."""

    method_key: str
    method_label: str
    branch: str
    score: float
    params: dict[str, Any]
    reason: str
    confidence: float
    risk_flags: list[str]
    score_terms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return asdict(self)


def score_gain_candidate(metrics: dict[str, float], method_key: str) -> float:
    """Score a gain result using target truth, artifact risk, and method priors."""
    terms = gain_score_terms(metrics, method_key)
    return float(sum(terms.values()))


def gain_score_terms(metrics: dict[str, float], method_key: str) -> dict[str, float]:
    """Return additive score terms for explaining a gain selection."""
    target_count = _metric(metrics, "target_count", 0.0)
    truth = _metric(metrics, "truth_score", 0.0)
    target_energy = _metric(metrics, "truth_target_energy_preservation", 1.0)
    target_band = _metric(metrics, "target_band_energy_ratio", 1.0)
    edge = _metric(metrics, "edge_preservation", 1.0)
    saliency_gain = _metric(metrics, "truth_target_saliency_gain", 1.0)
    contrast = _metric(metrics, "truth_target_contrast_after", 0.0)
    false_positive = _metric(metrics, "truth_false_positive_ratio", 1.0)
    background_reduction = _metric(metrics, "truth_background_energy_reduction", 0.0)
    amp_preserve = _metric(metrics, "relative_amplitude_preservation_score", 0.0)
    depth_balance = _metric(metrics, "depth_balance_score", 0.0)
    clip = _metric(metrics, "clipping_ratio_after", 0.0)
    hot = _metric(metrics, "hot_pixel_ratio_after", 0.0)

    terms = {
        "truth": 1.25 * truth,
        "amplitude_preservation": 0.55 * amp_preserve,
        "clipping_penalty": -8.0 * max(0.0, clip),
        "hot_pixel_penalty": -5.0 * max(0.0, hot),
        "method_prior": _method_prior(method_key, target_count),
    }

    if target_count <= 0:
        terms["truth"] = 1.4 * truth
        terms["amplitude_preservation"] = 0.5 * amp_preserve
        terms.update(
            {
                "background_reduction": 1.1 * float(np.clip(background_reduction, -1.0, 1.0)),
                "false_positive_penalty": -1.6 * max(0.0, false_positive - 1.0),
            }
        )
        return {key: float(value) for key, value in terms.items()}

    saliency_term = float(
        np.clip(np.log1p(max(0.0, saliency_gain)) / np.log(4.0), 0.0, 1.6)
    )
    contrast_term = float(
        np.clip(np.log1p(max(0.0, contrast)) / np.log(6.0), 0.0, 1.4)
    )
    target_vanish_penalty = (
        5.0 * max(0.0, 0.35 - target_energy)
        + 3.5 * max(0.0, 0.35 - target_band)
        + 2.5 * max(0.0, 0.35 - edge)
        + 1.4 * max(0.0, 0.90 - saliency_gain)
    )
    terms.update(
        {
            "target_saliency": 0.85 * saliency_term,
            "target_contrast": 0.45 * contrast_term,
            "depth_balance": 0.35 * depth_balance,
            "false_positive_penalty": -1.25 * max(0.0, false_positive - 0.85),
            "background_over_amplification_penalty": -0.40 * max(0.0, -background_reduction),
            "target_loss_penalty": -target_vanish_penalty,
        }
    )
    return {key: float(value) for key, value in terms.items()}


def gain_risk_flags(metrics: dict[str, float], method_key: str) -> list[str]:
    """Return risk flags that should trigger review of a gain decision."""
    target_count = _metric(metrics, "target_count", 0.0)
    target_energy = _metric(metrics, "truth_target_energy_preservation", 1.0)
    saliency_gain = _metric(metrics, "truth_target_saliency_gain", 1.0)
    false_positive = _metric(metrics, "truth_false_positive_ratio", 1.0)
    background_reduction = _metric(metrics, "truth_background_energy_reduction", 0.0)
    amp_preserve = _metric(metrics, "relative_amplitude_preservation_score", 1.0)
    clip = _metric(metrics, "clipping_ratio_after", 0.0)
    hot = _metric(metrics, "hot_pixel_ratio_after", 0.0)

    flags: list[str] = []
    if method_key == "agcGain":
        flags.append("relative_amplitude_not_interpretable")
    if target_count > 0 and target_energy < 0.35:
        flags.append("target_energy_loss")
    if target_count > 0 and saliency_gain < 0.90:
        flags.append("target_saliency_not_improved")
    if target_count <= 0 and false_positive > 1.25:
        flags.append("no_target_false_positive_amplification")
    elif false_positive > 1.10:
        flags.append("false_positive_amplification")
    if background_reduction < -0.10:
        flags.append("background_energy_amplified")
    if amp_preserve < 0.45:
        flags.append("low_relative_amplitude_preservation")
    if clip > 0.001:
        flags.append("clipping_risk")
    if hot > 0.005:
        flags.append("hot_pixel_risk")
    return _dedupe(flags)


def choose_gain_candidate(candidates: list[dict[str, Any]]) -> GainCandidateDecision:
    """Choose the best gain branch and attach confidence/risk metadata."""
    if not candidates:
        raise ValueError("gain selection requires at least one candidate")

    normalized = [_normalize_candidate(candidate) for candidate in candidates]
    target_count = max(_metric(item.get("metrics", {}), "target_count", 0.0) for item in normalized)
    if target_count > 0.0:
        gain_candidates = [
            item for item in normalized if item.get("method_key") != "no_gain"
        ]
        if gain_candidates:
            normalized = gain_candidates

    ranked = sorted(normalized, key=lambda item: float(item["score"]), reverse=True)
    best = ranked[0]
    second_score = float(ranked[1]["score"]) if len(ranked) > 1 else float("-inf")
    margin = float(best["score"]) - second_score if np.isfinite(second_score) else 1.0
    confidence = _confidence_from_margin(margin, float(best["score"]))
    risk_flags = list(best.get("risk_flags") or [])
    if len(ranked) > 1 and margin < 0.08:
        risk_flags.append("near_tie_gain_choice")
    if confidence < 0.45:
        risk_flags.append("low_gain_selection_confidence")

    return GainCandidateDecision(
        method_key=str(best["method_key"]),
        method_label=str(best["method_label"]),
        branch=str(best["branch"]),
        score=float(best["score"]),
        params=dict(best.get("params", {})),
        reason=str(best.get("reason") or ""),
        confidence=float(confidence),
        risk_flags=_dedupe(risk_flags),
        score_terms=dict(best.get("score_terms") or {}),
    )


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    method_key = str(candidate.get("method_key") or "")
    if not method_key:
        raise ValueError("gain candidate missing method_key")
    metrics = dict(candidate.get("metrics") or {})
    score = candidate.get("score")
    if score is None:
        score = score_gain_candidate(metrics, method_key)
    score_terms = candidate.get("score_terms")
    if score_terms is None:
        score_terms = gain_score_terms(metrics, method_key)
    risk_flags = list(candidate.get("risk_flags") or gain_risk_flags(metrics, method_key))
    return {
        "method_key": method_key,
        "method_label": str(
            candidate.get("method_label") or GAIN_METHOD_LABELS.get(method_key, method_key)
        ),
        "branch": str(candidate.get("branch") or ""),
        "score": float(score),
        "params": dict(candidate.get("params") or {}),
        "reason": str(candidate.get("reason") or ""),
        "metrics": metrics,
        "risk_flags": risk_flags,
        "score_terms": {str(key): float(value) for key, value in dict(score_terms).items()},
    }


def _method_prior(method_key: str, target_count: float) -> float:
    return {
        "sec_gain": 0.18,
        "compensatingGain": 0.08,
        "agcGain": -0.10,
        "no_gain": 0.04 if target_count <= 0 else -0.12,
    }.get(method_key, 0.0)


def _confidence_from_margin(margin: float, best_score: float) -> float:
    if not np.isfinite(margin):
        return 0.0
    scale = max(0.2, abs(float(best_score)) * 0.18)
    return float(np.clip(margin / (scale + 1.0e-9), 0.0, 1.0))


def _metric(metrics: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def _dedupe(flags: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for flag in flags:
        item = str(flag)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
