#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric terms and weighted scoring for MyGPR AutoTune v2.

The functions here produce bounded proxy scores.  For synthetic paired data they
can include a target-response similarity term; for real data they remain
no-prior processing heuristics and must not be interpreted as ground truth.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from core.autotune_goal_profiles import resolve_goal_profile
from core.autotune_score_normalization import clamp01, logistic01, ratio_similarity, weighted_sum


def _energy(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    return float(np.mean(vals * vals)) if vals.size else 0.0


def _shape_similarity(a_values: np.ndarray, b_values: np.ndarray) -> float:
    a = np.asarray(a_values, dtype=np.float64).ravel()
    b = np.asarray(b_values, dtype=np.float64).ravel()
    if a.size < 2 or b.size != a.size:
        return 0.0
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return clamp01(float(np.dot(a, b) / denom))


def _depth_balance(original: np.ndarray, candidate: np.ndarray) -> float:
    samples = int(original.shape[0])
    if samples < 3:
        return 0.5
    top_slice = slice(0, max(1, samples // 3))
    bottom_slice = slice(max(0, samples * 2 // 3), samples)
    raw_ratio = (_energy(original[bottom_slice, :]) + 1e-12) / (_energy(original[top_slice, :]) + 1e-12)
    cand_ratio = (_energy(candidate[bottom_slice, :]) + 1e-12) / (_energy(candidate[top_slice, :]) + 1e-12)
    return ratio_similarity(cand_ratio / raw_ratio, tolerance_ratio=4.0)


def _reference_similarity(candidate: np.ndarray, target_response: np.ndarray | None, roi_mask: np.ndarray) -> float | None:
    if target_response is None:
        return None
    diff = np.asarray(candidate[roi_mask] - target_response[roi_mask], dtype=np.float64)
    ref = np.asarray(target_response[roi_mask], dtype=np.float64)
    if diff.size == 0:
        return None
    rmse = float(np.sqrt(np.mean(diff * diff)))
    scale = float(np.sqrt(np.mean(ref * ref))) + 1e-12
    return float(1.0 / (1.0 + rmse / scale))


def _method_stability_prior(method: str) -> float:
    method = str(method or "").lower()
    return {
        "baseline": 0.50,
        "none": 0.50,
        "mean": 0.78,
        "median": 0.86,
        "sliding": 0.74,
        "svd": 0.62,
    }.get(method, 0.70)


def background_score_terms_v2(
    *,
    original: np.ndarray,
    candidate: np.ndarray,
    roi_mask: np.ndarray,
    bg_mask: np.ndarray,
    method: str,
    cnr_gain: float,
    roi_energy_ratio: float,
    residual_ratio: float,
    background_suppression: float,
    target_response: np.ndarray | None = None,
) -> dict[str, float]:
    """Return high-level 0..1 scoring terms for a background candidate."""
    response_preservation = ratio_similarity(roi_energy_ratio, tolerance_ratio=3.0)
    continuity = _shape_similarity(original[roi_mask], candidate[roi_mask])
    contrast = logistic01(cnr_gain, slope=1.15, center=0.0)
    deep_balance = _depth_balance(original, candidate)

    over_suppression_penalty = clamp01((0.35 - float(roi_energy_ratio)) / 0.35)
    over_amplification_penalty = clamp01((float(roi_energy_ratio) - 1.8) / 1.2)
    weak_suppression_penalty = 0.20 * clamp01((float(residual_ratio) - 0.95) / 0.25)
    artifact_control = clamp01(1.0 - over_suppression_penalty - over_amplification_penalty - weak_suppression_penalty)

    terms = {
        "background_suppression": clamp01(background_suppression),
        "response_preservation": response_preservation,
        "continuity": continuity,
        "contrast": contrast,
        "deep_balance": deep_balance,
        "artifact_control": artifact_control,
        "stability": _method_stability_prior(method),
        # Aliases used by specialized goal profiles.
        "texture_preservation": 0.5 * continuity + 0.5 * response_preservation,
        "gain_stability": min(_method_stability_prior(method), artifact_control),
    }
    ref = _reference_similarity(candidate, target_response, roi_mask)
    if ref is not None:
        terms["reference_similarity"] = ref
    return {key: float(clamp01(value)) for key, value in terms.items()}


def score_background_candidate_v2(
    *,
    terms: Mapping[str, float],
    target_goal: str | None,
    target_response_available: bool = False,
) -> dict:
    """Return weighted background score and auditable breakdown."""
    profile = resolve_goal_profile(target_goal)
    profile_score = weighted_sum(terms, profile.weights)
    if target_response_available and "reference_similarity" in terms:
        final_score = 0.82 * profile_score + 0.18 * clamp01(terms["reference_similarity"])
        blend = {"goal_profile": 0.82, "reference_similarity": 0.18}
    else:
        final_score = profile_score
        blend = {"goal_profile": 1.0}
    return {
        "score": float(clamp01(final_score)),
        "goal_profile_score": float(clamp01(profile_score)),
        "goal_profile": profile.to_dict(),
        "weights": dict(profile.weights),
        "terms": {str(k): float(v) for k, v in dict(terms).items()},
        "blend": blend,
        "scoring_version": "autotune_scoring_v2",
    }


def score_workflow_recipe_v2(
    *,
    background_score: float,
    workflow_fit: float,
    compactness: float,
    target_response_available: bool,
) -> dict:
    """Score a bounded workflow recipe from its high-level components."""
    terms = {
        "background_candidate_score": clamp01(background_score),
        "workflow_fit": clamp01(workflow_fit),
        "compactness": clamp01(compactness),
        "target_response_available": 1.0 if target_response_available else 0.0,
    }
    weights = {
        "background_candidate_score": 0.44,
        "workflow_fit": 0.36,
        "compactness": 0.14,
        "target_response_available": 0.06 if target_response_available else 0.0,
    }
    return {
        "score": float(clamp01(weighted_sum(terms, weights))),
        "terms": terms,
        "weights": weights,
        "scoring_version": "autotune_scoring_v2",
    }


__all__ = [
    "background_score_terms_v2",
    "score_background_candidate_v2",
    "score_workflow_recipe_v2",
]
