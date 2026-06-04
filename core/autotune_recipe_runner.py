#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert AutoTune workflow recipes into executable processing tasks.

The workflow planner returns compact, UI-friendly recipe dictionaries.  This
module maps that bounded recipe representation onto the existing MyGPR
``WorkflowMethod`` / ``ProcessingWorker`` execution path.  It intentionally does
not introduce new algorithms, external commands, or a free-form scripting layer.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from core.methods_registry import PROCESSING_METHODS
from core.workflow_data import WorkflowMethod


@dataclass(frozen=True)
class RecipeRunnerStep:
    """Resolved execution step for a workflow recipe."""

    recipe_key: str
    label: str
    method_id: str | None
    method_name: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecipeExecutionPlan:
    """Executable representation consumed by GUI processing workers."""

    name: str
    target_goal: str
    roi_mode: str
    score: float
    steps: tuple[RecipeRunnerStep, ...]
    scoring_record: dict[str, Any] = field(default_factory=dict)

    @property
    def executable_steps(self) -> tuple[RecipeRunnerStep, ...]:
        return tuple(step for step in self.steps if step.enabled and not step.skipped and step.method_id)

    @property
    def skipped_steps(self) -> tuple[RecipeRunnerStep, ...]:
        return tuple(step for step in self.steps if step.skipped or not step.enabled or not step.method_id)

    def to_workflow_methods(self) -> list[WorkflowMethod]:
        methods: list[WorkflowMethod] = []
        for order, step in enumerate(self.executable_steps):
            method_id = str(step.method_id)
            category = _category_for_method(method_id)
            methods.append(
                WorkflowMethod(
                    category=category,
                    method_id=method_id,
                    enabled=True,
                    order=order,
                    params=dict(step.params),
                )
            )
        return methods

    def to_processing_tasks(self, *, out_dir: str | None = None) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for method in self.to_workflow_methods():
            method_info = PROCESSING_METHODS.get(method.method_id)
            if not method_info:
                continue
            tasks.append(
                {
                    "method_key": method.method_id,
                    "method": method_info,
                    "params": dict(method.params),
                    "out_dir": out_dir,
                    "param_source_mode": "recipe",
                    "recipe_step": method.to_dict(),
                    "autotune_scoring_record": dict(self.scoring_record or {}),
                    "autotune_recipe_plan": self.to_dict(),
                }
            )
        return tasks

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_goal": self.target_goal,
            "roi_mode": self.roi_mode,
            "score": float(self.score),
            "steps": [step.to_dict() for step in self.steps],
            "executable_count": len(self.executable_steps),
            "skipped_count": len(self.skipped_steps),
            "autotune_scoring_record": dict(self.scoring_record or {}),
        }


def build_recipe_execution_plan(recipe_result: Mapping[str, Any] | None) -> RecipeExecutionPlan:
    """Resolve a planner row into a bounded execution plan.

    Parameters
    ----------
    recipe_result:
        A row returned by ``plan_workflow_recipes``.  The function also accepts
        a minimal dict with ``recipe_steps`` for tests and future report replay.
    """
    row = dict(recipe_result or {})
    recipe_steps = row.get("recipe_steps") or []
    background_candidate = row.get("background_candidate") or {}
    resolved: list[RecipeRunnerStep] = []
    for raw_step in recipe_steps:
        step = _normalise_step(raw_step)
        resolved.append(_resolve_step(step, background_candidate=background_candidate))

    scoring_record = row.get("autotune_scoring_record")
    if not isinstance(scoring_record, dict):
        scoring_record = {}
    return RecipeExecutionPlan(
        name=str(row.get("name") or "AutoTune 推荐流程"),
        target_goal=str(row.get("target_goal") or "均衡推荐"),
        roi_mode=str(row.get("roi_mode") or "none"),
        score=float(row.get("score", 0.0) or 0.0),
        steps=tuple(resolved),
        scoring_record=dict(scoring_record),
    )


def build_recipe_processing_tasks(
    recipe_result: Mapping[str, Any] | None,
    *,
    out_dir: str | None = None,
) -> tuple[list[dict[str, Any]], RecipeExecutionPlan]:
    """Return ProcessingWorker task dictionaries for the selected recipe."""
    plan = build_recipe_execution_plan(recipe_result)
    tasks = plan.to_processing_tasks(out_dir=out_dir)
    return tasks, plan


def _normalise_step(raw_step: Any) -> dict[str, Any]:
    if hasattr(raw_step, "__dataclass_fields__"):
        return asdict(raw_step)
    if isinstance(raw_step, Mapping):
        return dict(raw_step)
    return {
        "key": "step",
        "label": str(raw_step),
        "method": "auto",
        "params": "",
        "enabled": True,
    }


def _resolve_step(step: Mapping[str, Any], *, background_candidate: Mapping[str, Any]) -> RecipeRunnerStep:
    key = str(step.get("key") or "step")
    label = str(step.get("label") or key)
    method_text = str(step.get("method") or "")
    params_text = str(step.get("params") or "")
    enabled = bool(step.get("enabled", True))

    if not enabled:
        return _skipped(key, label, method_text, "该步骤在推荐流程中未启用")
    if key in {"zero_time", "display"}:
        return _skipped(key, label, method_text, "保持当前状态或仅影响显示，不写入处理任务")

    if key == "dewow":
        return RecipeRunnerStep(
            recipe_key=key,
            label=label,
            method_id="dewow",
            method_name="Dewow",
            params={"window": _parse_int(params_text, "window", default=23, minimum=3)},
        )

    if key == "bandpass":
        return RecipeRunnerStep(
            recipe_key=key,
            label=label,
            method_id="frequency_filter_1d",
            method_name="1D frequency filter",
            params=_bandpass_params(params_text, method_text),
        )

    if key == "background":
        return _resolve_background_step(key, label, method_text, params_text, background_candidate)

    if key == "gain":
        window = _parse_int(params_text, "window", default=41, minimum=3)
        return RecipeRunnerStep(
            recipe_key=key,
            label=label,
            method_id="agcGain",
            method_name="AGC gain",
            params={"window": window, "_low_energy_guard": True},
        )

    if key == "denoise":
        return RecipeRunnerStep(
            recipe_key=key,
            label=label,
            method_id="running_average_2D",
            method_name="Sharp clutter smoothing",
            params={"ntraces": _parse_int(params_text, "ntraces", default=5, minimum=3)},
        )

    return _skipped(key, label, method_text or "auto", "当前 recipe step 暂无安全执行映射")


def _resolve_background_step(
    key: str,
    label: str,
    method_text: str,
    params_text: str,
    background_candidate: Mapping[str, Any],
) -> RecipeRunnerStep:
    method_key = str(background_candidate.get("method") or "").lower()
    text = f"{method_text} {params_text} {background_candidate.get('name', '')}".lower()
    params = str(background_candidate.get("params") or params_text or "")

    if method_key in {"baseline", "none"} or "跳过" in method_text or "不处理" in text:
        return RecipeRunnerStep(
            recipe_key=key,
            label=label,
            method_id="median_background_2D",
            method_name="Median background removal",
            params={"ntraces": 99999, "_background_low_benefit": True},
        )

    if method_key == "svd" or "svd" in text:
        return RecipeRunnerStep(
            recipe_key=key,
            label=label,
            method_id="svd_bg",
            method_name="SVD background removal",
            params={"rank": _parse_int(params, "rank", default=1, minimum=1)},
        )

    if method_key == "median" or "median" in text or "中位" in method_text:
        ntraces = _parse_int(params, "ntraces", default=99999, minimum=1)
        return RecipeRunnerStep(
            recipe_key=key,
            label=label,
            method_id="median_background_2D",
            method_name="Median background removal",
            params={"ntraces": ntraces},
        )

    if method_key == "sliding" or "sliding" in text or "滑动" in method_text:
        return RecipeRunnerStep(
            recipe_key=key,
            label=label,
            method_id="subtracting_average_2D",
            method_name="Sliding mean background removal",
            params={"ntraces": _parse_int(params, "ntraces", default=31, minimum=3)},
        )

    # The background runner's mean candidate uses full-trace mean subtraction.
    return RecipeRunnerStep(
        recipe_key=key,
        label=label,
        method_id="subtracting_average_2D",
        method_name="Mean background removal",
        params={"ntraces": _parse_int(params, "ntraces", default=99999, minimum=1)},
    )


def _bandpass_params(params_text: str, method_text: str) -> dict[str, Any]:
    text = f"{params_text} {method_text}"
    low = 10.0
    high = 800.0
    taper = 0.08
    if "低频" in text or "深部" in text or "含水" in text:
        low, high, taper = 5.0, 500.0, 0.08
    elif "偏宽" in text or "纹理" in text or "异常" in text:
        low, high, taper = 10.0, 1000.0, 0.06
    elif "保守" in text or "界面" in text:
        low, high, taper = 10.0, 650.0, 0.10
    parsed_low = _parse_float(params_text, "low_freq_mhz", default=None)
    parsed_high = _parse_float(params_text, "high_freq_mhz", default=None)
    if parsed_low is not None:
        low = parsed_low
    if parsed_high is not None:
        high = parsed_high
    parsed_taper = _parse_float(params_text, "taper_ratio", default=None)
    if parsed_taper is not None:
        taper = parsed_taper
    return {
        "filter_type": "bandpass",
        "low_freq_mhz": float(low),
        "high_freq_mhz": float(high),
        "taper_ratio": float(taper),
    }


def _skipped(key: str, label: str, method_name: str, reason: str) -> RecipeRunnerStep:
    return RecipeRunnerStep(
        recipe_key=key,
        label=label,
        method_id=None,
        method_name=method_name or label,
        params={},
        enabled=False,
        skipped=True,
        skip_reason=reason,
    )


def _parse_int(text: str, key: str, *, default: int, minimum: int = 1) -> int:
    value = _parse_float(text, key, default=None)
    if value is None and key == "rank":
        match = re.search(r"rank\s*[=:]?\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
    if value is None and key == "window":
        match = re.search(r"window\s*[=:]?\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
    if value is None:
        value = float(default)
    result = max(int(minimum), int(round(value)))
    if result % 2 == 0 and key in {"window", "ntraces"}:
        result += 1
    return result


def _parse_float(text: str, key: str, *, default: float | None) -> float | None:
    pattern = rf"{re.escape(key)}\s*[=:]\s*([-+]?\d+(?:\.\d+)?)"
    match = re.search(pattern, text or "", flags=re.IGNORECASE)
    if not match:
        return default
    try:
        return float(match.group(1))
    except Exception:
        return default


def _category_for_method(method_id: str) -> str:
    if method_id in {"set_zero_time", "dewow", "time_cut", "trace_qc", "equidistant_trace_resample"}:
        return "preprocessing"
    if method_id in {"frequency_filter_1d", "subtracting_average_2D", "median_background_2D", "svd_bg", "fk_filter", "ccbs"}:
        return "background_removal"
    if method_id in {"agcGain", "compensatingGain", "sec_gain", "energy_decay_gain", "amplitude_scale"}:
        return "gain"
    if method_id in {"running_average_2D", "trace_median_filter", "trace_savgol_filter"}:
        return "artifact_suppression"
    if method_id in PROCESSING_METHODS:
        return str(PROCESSING_METHODS[method_id].get("auto_tune_family") or "processing")
    return "processing"


__all__ = [
    "RecipeRunnerStep",
    "RecipeExecutionPlan",
    "build_recipe_execution_plan",
    "build_recipe_processing_tasks",
]
