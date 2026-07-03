#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded AutoTune workflow recipe planner.

The planner turns the compact AutoTune UI settings into a ranked list of
processing workflow recipes.  It deliberately uses a small, auditable template
space instead of an unconstrained global workflow search:

- compute lightweight diagnostics from the current B-scan;
- reuse the background-suppression candidate sweep when available;
- generate target-aware recipe templates;
- score recipe suitability + parameter suitability deterministically;
- return display/report friendly dictionaries consumed by the GUI.

This module does not execute the processing workflow and does not mutate data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from core.autotune_metrics import score_workflow_recipe_v2
from core.autotune_scoring_record import build_scoring_v2_record
from core.autotune_recipe import AutoTuneRecipeStep, resolve_recipe_goal
from core.autotune_scoring_weights import resolve_scoring_weights


@dataclass(frozen=True)
class WorkflowDiagnostics:
    """Small B-scan descriptors used for bounded recipe scoring."""

    samples: int
    traces: int
    drift_strength: float
    stripe_strength: float
    continuity: float
    deep_energy_ratio: float
    local_anomaly_density: float
    spike_ratio: float
    target_response_available: bool = False

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowRecipeCandidate:
    """Rankable workflow recipe returned to the AutoTune page."""

    name: str
    params: str
    score: float
    status: str
    target_goal: str
    roi_mode: str
    recipe_steps: tuple[AutoTuneRecipeStep, ...]
    scoring_metrics: tuple[str, ...]
    scoring_weights: dict[str, float]
    scoring_terms: dict[str, float]
    diagnostics: WorkflowDiagnostics
    background_candidate: dict = field(default_factory=dict)
    note: str = ""
    score_version: str = "autotune_scoring_v2"
    workflow_score_weights: dict[str, float] = field(default_factory=dict)
    workflow_score_breakdown: dict = field(default_factory=dict)
    scoring_record: dict = field(default_factory=dict)
    candidate_space_context: dict = field(default_factory=dict)

    @property
    def flow_text(self) -> str:
        return " → ".join(step.label for step in self.recipe_steps if step.enabled)

    def to_dict(self) -> dict:
        row = {
            "name": self.name,
            "method": "workflow_recipe",
            "params": self.params,
            "score": float(self.score),
            "status": self.status,
            "target_goal": self.target_goal,
            "roi_mode": self.roi_mode,
            "recipe_steps": [asdict(step) for step in self.recipe_steps],
            "workflow_flow": self.flow_text,
            "scoring_metrics": self.scoring_metrics,
            "scoring_weights": dict(self.scoring_weights),
            "scoring_terms": dict(self.scoring_terms),
            "score_version": self.score_version,
            "workflow_score_weights": dict(self.workflow_score_weights),
            "workflow_score_breakdown": dict(self.workflow_score_breakdown),
            "score_breakdown": dict(self.workflow_score_breakdown),
            "autotune_scoring_record": dict(self.scoring_record),
            "diagnostics": self.diagnostics.to_dict(),
            "background_candidate": dict(self.background_candidate or {}),
            "background_low_benefit": bool(self.background_candidate.get("background_low_benefit", False)),
            "candidate_space_hash": self.candidate_space_context.get("candidate_space_hash") or self.background_candidate.get("candidate_space_hash"),
            "candidate_space_profile_id": self.candidate_space_context.get("candidate_space_profile_id") or self.background_candidate.get("candidate_space_profile_id"),
            "candidate_space_config_version": self.candidate_space_context.get("candidate_space_config_version") or self.background_candidate.get("candidate_space_config_version"),
            "candidate_space_recipe_ids": list(self.candidate_space_context.get("candidate_space_recipe_ids") or self.background_candidate.get("candidate_space_recipe_ids") or ()),
            "candidate_space_context": dict(self.candidate_space_context or {}),
            "note": self.note,
            # Compatibility fields for existing candidate/trial tables.
            "roi_energy_ratio": float(self.background_candidate.get("roi_energy_ratio", 0.0) or 0.0),
            "background_suppression": float(self.background_candidate.get("background_suppression", 0.0) or 0.0),
            "cnr_gain": float(self.background_candidate.get("cnr_gain", 0.0) or 0.0),
            "residual_ratio": float(self.background_candidate.get("residual_ratio", 1.0) or 1.0),
            "warning": self.note or self.status,
        }
        return row


def _finite_2d(data) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Workflow planner expects 2D B-scan data, got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("Workflow planner received empty data")
    if not np.isfinite(arr).all():
        finite = arr[np.isfinite(arr)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)
    return arr


def _norm01(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(max(0.0, min(1.0, (float(value) - lo) / (hi - lo))))


def diagnose_bscan(data, *, target_response_available: bool = False) -> WorkflowDiagnostics:
    """Compute deterministic lightweight diagnostics for recipe selection."""
    arr = _finite_2d(data)
    samples, traces = int(arr.shape[0]), int(arr.shape[1])
    centered = arr - float(np.median(arr))
    scale = float(np.percentile(np.abs(centered), 95)) + 1e-12
    z = centered / scale

    # Trace/sample summaries are robust enough for UI recipe gating.
    trace_mean = np.mean(z, axis=0)
    sample_mean = np.mean(z, axis=1)
    drift_strength = _norm01(float(np.std(sample_mean)), 0.02, 0.45)
    stripe_strength = _norm01(float(np.std(trace_mean)), 0.02, 0.50)

    if traces > 1:
        a = z[:, :-1].ravel()
        b = z[:, 1:].ravel()
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        continuity = float(max(0.0, min(1.0, np.dot(a, b) / denom)))
    else:
        continuity = 0.0

    top = z[: max(1, samples // 3), :]
    bottom = z[max(0, samples * 2 // 3) :, :]
    top_energy = float(np.mean(top * top)) + 1e-12
    bottom_energy = float(np.mean(bottom * bottom)) + 1e-12
    deep_energy_ratio = float(max(0.0, min(2.0, bottom_energy / top_energy)))

    abs_z = np.abs(z)
    strong = float(np.percentile(abs_z, 98))
    local_anomaly_density = float(np.mean(abs_z >= max(1.0, strong)))
    spike_ratio = float(np.mean(abs_z >= 3.5))

    return WorkflowDiagnostics(
        samples=samples,
        traces=traces,
        drift_strength=drift_strength,
        stripe_strength=stripe_strength,
        continuity=continuity,
        deep_energy_ratio=deep_energy_ratio,
        local_anomaly_density=local_anomaly_density,
        spike_ratio=spike_ratio,
        target_response_available=bool(target_response_available),
    )


def _odd_window(value: int, *, minimum: int = 9, maximum: int = 121) -> int:
    value = max(minimum, min(maximum, int(value)))
    return value if value % 2 else value + 1 if value < maximum else value - 1


def _auto_dewow_params(diag: WorkflowDiagnostics, *, conservative: bool = False) -> str:
    base = diag.samples // (12 if conservative else 18)
    return f"window={_odd_window(base, minimum=11, maximum=101)}"


def _auto_bandpass_params(goal: str, diag: WorkflowDiagnostics) -> str:
    if goal in {"滑坡基覆界面 / 潜在滑移面", "含水软弱带", "深部弱反射增强"}:
        return "range=低频保留, order=4"
    if goal in {"局部异常增强", "裂隙/破碎带保留"}:
        return "range=偏宽带通, order=4"
    if goal == "连续界面保留":
        return "range=保守带通, order=4"
    return "range=auto, order=4"


def _auto_gain_params(goal: str, diag: WorkflowDiagnostics) -> str:
    wide = goal in {"滑坡基覆界面 / 潜在滑移面", "深部弱反射增强", "连续界面保留"}
    if goal == "含水软弱带":
        return "mode=温和增益, strength=moderate"
    if wide:
        return f"mode=AGC, window={_odd_window(diag.samples // 8, minimum=31, maximum=151)}"
    return f"mode=AGC, window={_odd_window(diag.samples // 14, minimum=21, maximum=111)}"


_REAL_BACKGROUND_METHODS = {"mean", "median", "svd", "sliding"}
_BASELINE_BACKGROUND_METHODS = {"baseline", "none", "skip"}


def _background_method_key(row: Mapping) -> str:
    return str(row.get("method") or "").strip().lower()


def _is_baseline_background(row: Mapping) -> bool:
    method = _background_method_key(row)
    text = f"{row.get('name', '')} {row.get('params', '')} {row.get('status', '')}".lower()
    return method in _BASELINE_BACKGROUND_METHODS or "不处理" in text or "method=none" in text


def _low_benefit_message() -> str:
    return "背景抑制收益较弱，已采用温和背景抑制方法。"


def _fallback_background_candidates(*, low_benefit: bool = False) -> list[dict]:
    status = "收益较弱" if low_benefit else "备选"
    note = _low_benefit_message() if low_benefit else ""
    return [
        {
            "name": "中位数背景扣除",
            "method": "median",
            "params": "method=median",
            "score": 0.56 if low_benefit else 0.66,
            "status": status,
            "background_low_benefit": bool(low_benefit),
            "note": note,
            "warning": note,
        },
        {
            "name": "均值背景扣除",
            "method": "mean",
            "params": "method=mean",
            "score": 0.52 if low_benefit else 0.58,
            "status": status,
            "background_low_benefit": bool(low_benefit),
            "note": note,
            "warning": note,
        },
    ]


def _candidate_space_context_from_rows(background_results: Sequence[Mapping] | None) -> dict:
    """Extract a shared V1 candidate-space context from background trial rows."""
    for row in background_results or []:
        if not isinstance(row, Mapping):
            continue
        ctx = row.get("candidate_space_context")
        if isinstance(ctx, Mapping) and ctx.get("candidate_space_hash"):
            return dict(ctx)
        if row.get("candidate_space_hash"):
            return {
                "candidate_space_hash": row.get("candidate_space_hash"),
                "candidate_space_profile_id": row.get("candidate_space_profile_id"),
                "candidate_space_config_version": row.get("candidate_space_config_version"),
                "candidate_space_recipe_ids": list(row.get("candidate_space_recipe_ids") or ()),
            }
    return {}


def _background_pool(background_results: Sequence[Mapping] | None) -> list[dict]:
    """Return executable background-suppression candidates for workflow recipes.

    ``baseline`` remains useful in the detailed candidate table as a no-processing
    reference, but it must not become the recommended workflow background step.
    When the background sweep ranks baseline highest or no executable method is
    available, the planner still selects a real, mild background-suppression
    method and marks the recipe as low-benefit instead of skipping the step.
    """
    all_rows = [dict(row) for row in (background_results or []) if str(row.get("status", "")) != "已跳过"]
    all_rows.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    baseline_was_best = bool(all_rows and _is_baseline_background(all_rows[0]))
    real_rows = [row for row in all_rows if _background_method_key(row) in _REAL_BACKGROUND_METHODS and not _is_baseline_background(row)]
    real_rows.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)

    if real_rows:
        if baseline_was_best:
            for row in real_rows:
                row["background_low_benefit"] = True
                row["status"] = "收益较弱"
                row["note"] = row.get("note") or _low_benefit_message()
                row["warning"] = row.get("warning") or _low_benefit_message()
        return real_rows[:4]

    # No executable background candidate survived the sweep. Keep the workflow
    # usable and auditable by falling back to mild real methods, never baseline.
    return _fallback_background_candidates(low_benefit=bool(all_rows))


def _goal_templates(goal: str) -> list[dict]:
    common = ["zero_time", "dewow", "bandpass", "background", "gain"]
    if goal == "局部异常增强":
        return [
            {"name": "局部异常增强流程", "steps": common, "fit": {"local": 0.34, "stripe": 0.18, "continuity": 0.06}},
            {"name": "强杂波抑制流程", "steps": ["zero_time", "dewow", "bandpass", "background", "gain", "display"], "fit": {"stripe": 0.30, "local": 0.22}},
        ]
    if goal == "连续界面保留":
        return [
            {"name": "连续界面保留流程", "steps": common, "fit": {"continuity": 0.36, "deep": 0.14, "stripe": 0.08}},
            {"name": "温和背景抑制流程", "steps": ["zero_time", "dewow", "bandpass", "background", "gain"], "fit": {"continuity": 0.28, "spike_low": 0.12}},
        ]
    if goal == "滑坡基覆界面 / 潜在滑移面":
        return [
            {"name": "基覆界面保留流程", "steps": common, "fit": {"continuity": 0.28, "deep": 0.26, "stripe": 0.08}},
            {"name": "深部界面增强流程", "steps": ["zero_time", "dewow", "bandpass", "background", "gain"], "fit": {"deep": 0.34, "continuity": 0.18}},
        ]
    if goal == "裂隙/破碎带保留":
        return [
            {"name": "裂隙纹理保留流程", "steps": ["zero_time", "dewow", "bandpass", "background", "denoise", "gain"], "fit": {"local": 0.20, "spike_low": 0.18, "stripe": 0.12}},
            {"name": "断续反射增强流程", "steps": ["zero_time", "dewow", "bandpass", "background", "gain"], "fit": {"local": 0.24, "continuity_mid": 0.16}},
        ]
    if goal == "含水软弱带":
        return [
            {"name": "含水软弱带保留流程", "steps": common, "fit": {"deep": 0.24, "continuity": 0.20, "stripe": 0.08}},
            {"name": "衰减带温和增强流程", "steps": ["zero_time", "dewow", "bandpass", "background", "gain"], "fit": {"deep": 0.30, "spike_low": 0.10}},
        ]
    if goal == "深部弱反射增强":
        return [
            {"name": "深部弱反射增强流程", "steps": common, "fit": {"deep": 0.40, "stripe": 0.08}},
            {"name": "宽窗增益稳定流程", "steps": ["zero_time", "dewow", "bandpass", "background", "gain"], "fit": {"deep": 0.30, "spike_low": 0.12}},
        ]
    return [
        {"name": "均衡稳健流程", "steps": common, "fit": {"stripe": 0.18, "continuity": 0.16, "deep": 0.10, "local": 0.10}},
        {"name": "轻量快速流程", "steps": ["zero_time", "dewow", "bandpass", "background"], "fit": {"stripe": 0.16, "spike_low": 0.08}},
    ]


def _fit_score(fit: Mapping[str, float], diag: WorkflowDiagnostics) -> float:
    values = {
        "stripe": diag.stripe_strength,
        "continuity": diag.continuity,
        "continuity_mid": 1.0 - abs(diag.continuity - 0.45) / 0.45,
        "deep": _norm01(diag.deep_energy_ratio, 0.10, 0.95),
        "local": min(1.0, diag.local_anomaly_density / 0.025),
        "spike_low": 1.0 - min(1.0, diag.spike_ratio / 0.035),
    }
    return float(sum(float(weight) * max(0.0, min(1.0, values.get(key, 0.0))) for key, weight in fit.items()))


def _step_spec(key: str, *, goal: str, diag: WorkflowDiagnostics, bg: Mapping) -> AutoTuneRecipeStep:
    if key == "zero_time":
        return AutoTuneRecipeStep(key="zero_time", label="零时校正", method="保持当前校正", params="使用当前设置")
    if key == "dewow":
        conservative = goal in {"连续界面保留", "滑坡基覆界面 / 潜在滑移面", "含水软弱带", "深部弱反射增强"}
        return AutoTuneRecipeStep(key="dewow", label="Dewow", method="移动窗口去低频漂移", params=_auto_dewow_params(diag, conservative=conservative))
    if key == "bandpass":
        return AutoTuneRecipeStep(key="bandpass", label="频带滤波", method="Butterworth bandpass", params=_auto_bandpass_params(goal, diag))
    if key == "background":
        name = str(bg.get("name") or bg.get("method") or "auto")
        if _is_baseline_background(bg):
            name = "中位数背景扣除"
            bg = {**dict(bg), "method": "median", "params": "method=median", "background_low_benefit": True}
        params = str(bg.get("params") or "params=auto")
        if bg.get("background_low_benefit") and "收益较弱" not in params:
            params = f"{params}; 收益较弱"
        return AutoTuneRecipeStep(
            key="background",
            label="背景抑制",
            method=name,
            params=params,
        )
    if key == "denoise":
        return AutoTuneRecipeStep(key="denoise", label="轻度去尖峰", method="Hampel / 中值", params="strength=low")
    if key == "gain":
        label = "深部增益" if goal in {"滑坡基覆界面 / 潜在滑移面", "深部弱反射增强"} else "增益"
        return AutoTuneRecipeStep(key="gain", label=label, method="AGC / 温和增益", params=_auto_gain_params(goal, diag))
    if key == "display":
        return AutoTuneRecipeStep(key="display", label="显示增强", method="percentile stretch", params="display-only")
    return AutoTuneRecipeStep(key=key, label=key, method="auto", params="params=auto")


def plan_workflow_recipes(
    data,
    *,
    target_goal: str | None = "均衡推荐",
    roi_mode: str | None = "none",
    scoring_metrics: Iterable[str] | None = None,
    target_response=None,
    background_results: Sequence[Mapping] | None = None,
    max_candidates: int = 12,
    candidate_space_context: Mapping | None = None,
) -> list[dict]:
    """Return ranked workflow recipe dictionaries for the AutoTune GUI."""
    target_available = target_response is not None
    diag = diagnose_bscan(data, target_response_available=target_available)
    goal = resolve_recipe_goal(target_goal)
    resolved_goal, metrics, weights = resolve_scoring_weights(
        target_goal=goal,
        scoring_metrics=scoring_metrics,
        target_response_available=target_available,
    )
    bg_pool = _background_pool(background_results)
    candidate_space_ctx = dict(candidate_space_context or _candidate_space_context_from_rows(background_results))
    templates = _goal_templates(resolved_goal)

    candidates: list[WorkflowRecipeCandidate] = []
    for template_idx, template in enumerate(templates):
        for bg_idx, bg in enumerate(bg_pool):
            steps = tuple(_step_spec(key, goal=resolved_goal, diag=diag, bg=bg) for key in template["steps"])
            bg_score = max(0.0, min(1.0, float(bg.get("score", 0.0) or 0.0)))
            fit = _fit_score(template.get("fit", {}), diag)
            compactness = 1.0 - max(0, len(steps) - 5) * 0.035
            workflow_breakdown = score_workflow_recipe_v2(
                background_score=bg_score,
                workflow_fit=fit,
                compactness=compactness,
                target_response_available=target_available,
            )
            score = max(0.0, min(0.98, float(workflow_breakdown["score"])))
            name = f"{template['name']} · {bg.get('name', '背景候选')}"
            params = "；".join(f"{step.label}:{step.params}" for step in steps if step.enabled and step.key != "zero_time")
            terms = {key: float(value) for key, value in workflow_breakdown.get("terms", {}).items()}
            bg_terms = bg.get("scoring_terms") if isinstance(bg, Mapping) else None
            if isinstance(bg_terms, Mapping):
                for key, value in bg_terms.items():
                    try:
                        terms[str(key)] = float(value)
                    except Exception:
                        continue
            note = str(bg.get("note") or bg.get("warning") or "").strip() or "基于当前数据诊断和候选参数生成。"
            workflow_score_weights = {key: float(value) for key, value in workflow_breakdown.get("weights", {}).items()}
            workflow_score_breakdown = dict(workflow_breakdown)
            candidate_payload = {
                "name": name,
                "method": "workflow_recipe",
                "params": params,
                "score": score,
                "status": "推荐流程" if not candidates else "备选流程",
                "target_goal": resolved_goal,
                "roi_mode": str(roi_mode or "none"),
                "recipe_steps": [asdict(step) for step in steps],
                "workflow_flow": " → ".join(step.label for step in steps if step.enabled),
                "scoring_metrics": metrics,
                "scoring_weights": dict(weights),
                "scoring_terms": terms,
                "workflow_score_weights": workflow_score_weights,
                "workflow_score_breakdown": workflow_score_breakdown,
                "background_candidate": dict(bg),
                "candidate_space_hash": candidate_space_ctx.get("candidate_space_hash") or bg.get("candidate_space_hash"),
                "candidate_space_profile_id": candidate_space_ctx.get("candidate_space_profile_id") or bg.get("candidate_space_profile_id"),
                "candidate_space_config_version": candidate_space_ctx.get("candidate_space_config_version") or bg.get("candidate_space_config_version"),
                "candidate_space_recipe_ids": list(candidate_space_ctx.get("candidate_space_recipe_ids") or bg.get("candidate_space_recipe_ids") or ()),
                "candidate_space_context": dict(candidate_space_ctx or {}),
                "diagnostics": diag.to_dict(),
                "note": note,
                "background_low_benefit": bool(bg.get("background_low_benefit", False)),
                "selection_rule": "bounded_recipe_scoring_v2",
            }
            scoring_record = build_scoring_v2_record(
                candidate_payload,
                target_goal=resolved_goal,
                roi_mode=str(roi_mode or "none"),
                target_response_available=target_available,
            )
            candidates.append(
                WorkflowRecipeCandidate(
                    name=name,
                    params=params,
                    score=score,
                    status="推荐流程" if not candidates else "备选流程",
                    target_goal=resolved_goal,
                    roi_mode=str(roi_mode or "none"),
                    recipe_steps=steps,
                    scoring_metrics=metrics,
                    scoring_weights=dict(weights),
                    scoring_terms=terms,
                    diagnostics=diag,
                    background_candidate=dict(bg),
                    note=note,
                    score_version=str(workflow_breakdown.get("scoring_version", "autotune_scoring_v2")),
                    workflow_score_weights=workflow_score_weights,
                    workflow_score_breakdown=workflow_score_breakdown,
                    scoring_record=scoring_record,
                    candidate_space_context=dict(candidate_space_ctx or {}),
                )
            )

    candidates.sort(key=lambda item: item.score, reverse=True)
    ranked: list[dict] = []
    for rank, candidate in enumerate(candidates[: max(1, int(max_candidates))], start=1):
        row = candidate.to_dict()
        row["rank"] = rank
        row["status"] = "推荐流程" if rank == 1 else "备选流程"
        ranked.append(row)
    return ranked


__all__ = [
    "WorkflowDiagnostics",
    "WorkflowRecipeCandidate",
    "diagnose_bscan",
    "plan_workflow_recipes",
]
