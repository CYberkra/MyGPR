#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quality scoring, risk assessment and recommendation policy for auto-tuned pipelines."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.auto_tune_pipeline_geometry import _clamp_bounds
from core.auto_tune_pipeline_models import PipelineStepRecord
from core.gprmax_truth_metrics import compute_ground_truth_metrics
from core.quality_metrics import compute_benchmark_metrics, ratio_fidelity

SCORE_REJECT_DELTA = -0.02
SCORE_REVIEW_DELTA = 0.02
LOW_CONFIDENCE_THRESHOLD = 0.45
LOW_MARGIN_THRESHOLD = 0.03

def _compute_branch_metrics(
    reference: np.ndarray,
    processed: np.ndarray,
    reference_roi: dict[str, int],
    processed_roi: dict[str, int],
    ground_truth: dict[str, Any] | None,
) -> dict[str, float]:
    before_roi, after_roi = _slice_roi_pair(
        reference,
        processed,
        reference_roi,
        processed_roi,
    )
    metrics = compute_benchmark_metrics(before_roi, after_roi)
    if ground_truth:
        metrics.update(
            compute_ground_truth_metrics(
                reference,
                processed,
                ground_truth,
                reference_roi=reference_roi,
                processed_roi=processed_roi,
            )
        )
    metrics["pipeline_score"] = _pipeline_score(metrics)
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(float(value))
    }

def _slice_roi_pair(
    reference: np.ndarray,
    processed: np.ndarray,
    reference_roi: dict[str, int],
    processed_roi: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float32)
    proc = np.asarray(processed, dtype=np.float32)
    ref_bounds = _clamp_bounds(ref.shape, reference_roi)
    proc_bounds = _clamp_bounds(proc.shape, processed_roi)
    ref_roi = ref[
        ref_bounds["time_start_idx"] : ref_bounds["time_end_idx"],
        ref_bounds["dist_start_idx"] : ref_bounds["dist_end_idx"],
    ]
    proc_roi = proc[
        proc_bounds["time_start_idx"] : proc_bounds["time_end_idx"],
        proc_bounds["dist_start_idx"] : proc_bounds["dist_end_idx"],
    ]
    rows = max(1, min(ref_roi.shape[0], proc_roi.shape[0]))
    cols = max(1, min(ref_roi.shape[1], proc_roi.shape[1]))
    return ref_roi[:rows, :cols], proc_roi[:rows, :cols]

def _pipeline_score(metrics: dict[str, float]) -> float:
    band_fidelity = ratio_fidelity(metrics["target_band_energy_ratio"], tol=0.35)
    saliency_fidelity = ratio_fidelity(
        metrics["local_saliency_preservation"],
        tol=0.35,
    )
    edge_fidelity = ratio_fidelity(metrics["edge_preservation"], tol=0.35)
    deep_gain = max(0.0, float(metrics["deep_zone_contrast_gain"]) - 1.0)
    target_loss_penalty = (
        max(0.0, 0.55 - float(metrics["target_band_energy_ratio"])) * 3.0
        + max(0.0, 0.55 - float(metrics["local_saliency_preservation"])) * 4.0
        + max(0.0, 0.55 - float(metrics["edge_preservation"])) * 3.0
    )
    artifact_penalty = (
        6.0 * float(metrics["clipping_ratio_after"])
        + 4.0 * float(metrics["hot_pixel_ratio_after"])
        + 0.08 * float(metrics["kurtosis_or_spikiness_after"])
    )
    score = (
        1.2 * float(metrics["baseline_bias_reduction"])
        + 1.4 * float(metrics["low_freq_energy_reduction"])
        + 0.8 * float(metrics["horizontal_coherence_reduction"])
        + 1.8 * band_fidelity
        + 2.0 * saliency_fidelity
        + 1.4 * edge_fidelity
        + 0.4 * np.log1p(deep_gain)
        - target_loss_penalty
        - artifact_penalty
    )
    if "truth_score" in metrics:
        score = 0.65 * score + 2.2 * float(metrics["truth_score"])
    return float(score)

def _compute_metric_delta(
    manual_metrics: dict[str, float],
    auto_metrics: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(manual_metrics) & set(auto_metrics))
    return {
        key: float(auto_metrics[key] - manual_metrics[key])
        for key in keys
        if np.isfinite(manual_metrics[key]) and np.isfinite(auto_metrics[key])
    }

def _assess_step_risk(
    manual_metrics: dict[str, float],
    auto_metrics: dict[str, float],
    tune_result: dict[str, Any] | None,
) -> tuple[list[str], str, str]:
    flags: list[str] = []
    score_delta = float(auto_metrics.get("pipeline_score", 0.0)) - float(
        manual_metrics.get("pipeline_score", 0.0)
    )
    if score_delta < SCORE_REJECT_DELTA:
        flags.append("auto_worse_than_manual")
    elif abs(score_delta) <= SCORE_REVIEW_DELTA:
        flags.append("near_tie")

    if tune_result is not None:
        confidence = float(tune_result.get("selection_confidence", 1.0))
        margin = float(tune_result.get("selection_margin", 1.0))
        stats = tune_result.get("execution_stats", {}) or {}
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            flags.append("low_selection_confidence")
        if margin < LOW_MARGIN_THRESHOLD:
            flags.append("multiple_near_optima")
        if int(stats.get("constraint_adjustment_count", 0) or 0) > 0:
            flags.append("constraint_adjusted")
        if tune_result.get("constraint_warnings"):
            flags.append("constraint_adjusted")

    truth_count = float(auto_metrics.get("truth_target_count", -1.0))
    if truth_count > 0.0:
        manual_preserve = float(
            manual_metrics.get("truth_target_energy_preservation", 1.0)
        )
        auto_preserve = float(auto_metrics.get("truth_target_energy_preservation", 1.0))
        manual_truth_score = float(manual_metrics.get("truth_score", 0.0))
        auto_truth_score = float(auto_metrics.get("truth_score", 0.0))
        if auto_preserve < manual_preserve - 0.08:
            flags.append("target_truth_degraded")
        elif auto_preserve < 0.55:
            flags.append("low_truth_target_preservation")
        if auto_truth_score < manual_truth_score - 0.05:
            flags.append("target_truth_degraded")
    elif truth_count == 0.0:
        manual_fp = float(manual_metrics.get("truth_false_positive_ratio", 0.0))
        auto_fp = float(auto_metrics.get("truth_false_positive_ratio", 0.0))
        if auto_fp > manual_fp + max(0.10, abs(manual_fp) * 0.20):
            flags.append("false_positive_risk")

    if (
        float(auto_metrics.get("clipping_ratio_after", 0.0))
        > float(manual_metrics.get("clipping_ratio_after", 0.0)) + 0.01
    ):
        flags.append("overexposure_risk")
    if (
        float(auto_metrics.get("hot_pixel_ratio_after", 0.0))
        > float(manual_metrics.get("hot_pixel_ratio_after", 0.0)) + 0.01
    ):
        flags.append("overexposure_risk")

    flags = _dedupe_flags(flags)
    severe = {
        "auto_worse_than_manual",
        "target_truth_degraded",
        "false_positive_risk",
        "overexposure_risk",
    }
    caution = {
        "near_tie",
        "low_selection_confidence",
        "multiple_near_optima",
        "constraint_adjusted",
        "low_truth_target_preservation",
    }
    if any(flag in severe for flag in flags):
        return flags, "keep_manual", _risk_reason(flags, score_delta)
    if any(flag in caution for flag in flags):
        return flags, "review", _risk_reason(flags, score_delta)
    return flags, "adopt_auto", f"auto pipeline score delta={score_delta:.4f}"

def _risk_reason(flags: list[str], score_delta: float) -> str:
    if not flags:
        return f"auto pipeline score delta={score_delta:.4f}"
    return f"risk={', '.join(flags)}; auto pipeline score delta={score_delta:.4f}"

def _overall_recommendation(
    steps: list[PipelineStepRecord],
    metric_delta: dict[str, float],
    risk_flags: list[str],
) -> str:
    if any(step.recommendation == "keep_manual" for step in steps):
        return "keep_manual"
    final_delta = float(metric_delta.get("pipeline_score", 0.0))
    if final_delta < SCORE_REJECT_DELTA:
        return "keep_manual"
    if any(step.recommendation == "review" for step in steps):
        return "review"
    if risk_flags or abs(final_delta) <= SCORE_REVIEW_DELTA:
        return "review"
    return "adopt_auto"

def _dedupe_flags(flags: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for flag in flags:
        text = str(flag)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered

def _extract_warning_messages(meta: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    for warning in meta.get("runtime_warnings", []) or []:
        if isinstance(warning, dict):
            messages.append(str(warning.get("message") or warning.get("code") or warning))
        else:
            messages.append(str(warning))
    for warning in meta.get("warnings", []) or []:
        messages.append(str(warning))
    if meta.get("skipped"):
        reason = str(meta.get("reason") or "method skipped")
        messages.append(reason)
    return messages

__all__ = ['_compute_branch_metrics', '_slice_roi_pair', '_pipeline_score', '_compute_metric_delta', '_assess_step_risk', '_risk_reason', '_overall_recommendation', '_dedupe_flags', '_extract_warning_messages']
