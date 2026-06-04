#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight AutoTune workflow recipe models.

These helpers describe the recommended processing recipe shown in the GUI.
They do not execute processing and do not claim global workflow optimality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class AutoTuneRecipeStep:
    """A single recommended processing step for UI/report presentation."""

    key: str
    label: str
    method: str
    params: str = "--"
    enabled: bool = True
    source: str = "auto"


@dataclass(frozen=True)
class AutoTuneRecipe:
    """A display-safe AutoTune workflow recommendation."""

    target_goal: str
    roi_mode: str
    steps: tuple[AutoTuneRecipeStep, ...]
    score: float = 0.0
    data_mode: str = "无参考标签"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def flow_text(self) -> str:
        return " → ".join(step.label for step in self.steps if step.enabled)

    @property
    def parameter_text(self) -> str:
        return "\n".join(
            f"{step.label}：{step.method}，{step.params}" for step in self.steps if step.enabled
        )


_GOAL_PIPELINES: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "均衡推荐": (
        ("zero_time", "零时校正", "保持当前校正", "使用已导入/当前设置"),
        ("dewow", "Dewow", "移动窗口去低频漂移", "window=auto"),
        ("bandpass", "频带滤波", "Butterworth bandpass", "range=auto"),
        ("background", "背景抑制", "auto", "params=auto"),
        ("gain", "增益", "AGC", "window=auto"),
    ),
    "局部异常增强": (
        ("zero_time", "零时校正", "保持当前校正", "使用已导入/当前设置"),
        ("dewow", "Dewow", "移动窗口去低频漂移", "window=auto"),
        ("bandpass", "频带滤波", "偏宽带通", "range=auto"),
        ("background", "背景抑制", "auto", "params=auto"),
        ("gain", "局部增强", "轻度 AGC / 对比增强", "strength=moderate"),
    ),
    "连续界面保留": (
        ("zero_time", "零时校正", "保持当前校正", "使用已导入/当前设置"),
        ("dewow", "Dewow", "保守去低频漂移", "window=auto"),
        ("bandpass", "频带滤波", "保守带通", "range=auto"),
        ("background", "背景抑制", "auto", "params=auto"),
        ("gain", "轻度增益", "AGC", "window=wide"),
    ),
    "滑坡基覆界面 / 潜在滑移面": (
        ("zero_time", "零时校正", "保持当前校正", "使用已导入/当前设置"),
        ("dewow", "Dewow", "保守去低频漂移", "window=auto"),
        ("bandpass", "频带滤波", "界面保留带通", "range=auto"),
        ("background", "背景抑制", "auto", "params=auto"),
        ("gain", "深部保留增益", "宽窗 AGC", "window=wide"),
    ),
    "裂隙/破碎带保留": (
        ("zero_time", "零时校正", "保持当前校正", "使用已导入/当前设置"),
        ("dewow", "Dewow", "移动窗口去低频漂移", "window=auto"),
        ("bandpass", "频带滤波", "纹理保留带通", "range=auto"),
        ("background", "背景抑制", "auto", "params=auto"),
        ("denoise", "轻度去尖峰", "Hampel / 中值", "strength=low"),
        ("gain", "增益", "AGC", "window=auto"),
    ),
    "含水软弱带": (
        ("zero_time", "零时校正", "保持当前校正", "使用已导入/当前设置"),
        ("dewow", "Dewow", "保守去低频漂移", "window=auto"),
        ("bandpass", "频带滤波", "低频信息保留", "range=auto"),
        ("background", "背景抑制", "auto", "params=auto"),
        ("gain", "衰减带增强", "温和增益", "strength=moderate"),
    ),
    "深部弱反射增强": (
        ("zero_time", "零时校正", "保持当前校正", "使用已导入/当前设置"),
        ("dewow", "Dewow", "保守去低频漂移", "window=auto"),
        ("bandpass", "频带滤波", "深部弱反射保留", "range=auto"),
        ("background", "背景抑制", "auto", "params=auto"),
        ("gain", "深部增益", "宽窗 AGC", "window=wide"),
    ),
}

_ALIASES = {
    "balanced": "均衡推荐",
    "default": "均衡推荐",
    "anomaly": "局部异常增强",
    "local_anomaly": "局部异常增强",
    "interface": "连续界面保留",
    "layer": "连续界面保留",
    "landslide_interface": "滑坡基覆界面 / 潜在滑移面",
    "fracture": "裂隙/破碎带保留",
    "wet_weak_zone": "含水软弱带",
    "weak_deep": "深部弱反射增强",
    "deep_weak_reflection": "深部弱反射增强",
}


def resolve_recipe_goal(target_goal: str | None) -> str:
    raw = str(target_goal or "均衡推荐").strip()
    if raw in _GOAL_PIPELINES:
        return raw
    return _ALIASES.get(raw.lower(), "均衡推荐")


def _recipe_steps_from_dicts(items: Sequence[Mapping] | None) -> list[AutoTuneRecipeStep]:
    steps: list[AutoTuneRecipeStep] = []
    for item in items or []:
        try:
            steps.append(
                AutoTuneRecipeStep(
                    key=str(item.get("key", "step")),
                    label=str(item.get("label", item.get("key", "step"))),
                    method=str(item.get("method", "auto")),
                    params=str(item.get("params", "--")),
                    enabled=bool(item.get("enabled", True)),
                    source=str(item.get("source", "auto")),
                )
            )
        except AttributeError:
            if isinstance(item, AutoTuneRecipeStep):
                steps.append(item)
    return steps


def build_workflow_recipe(
    *,
    target_goal: str | None,
    roi_mode: str | None,
    best_candidate_name: str = "--",
    best_candidate_params: str = "--",
    best_score: float = 0.0,
    target_response_available: bool = False,
    backend_mode: str = "UI 预览",
    recipe_steps: Sequence[Mapping] | None = None,
) -> AutoTuneRecipe:
    """Build a display recipe from the current recommendation state."""
    goal = resolve_recipe_goal(target_goal)
    steps = _recipe_steps_from_dicts(recipe_steps)
    if not steps:
        steps = []
        for key, label, method, params in _GOAL_PIPELINES.get(goal, _GOAL_PIPELINES["均衡推荐"]):
            if key == "background" and best_candidate_name and best_candidate_name != "--":
                method = best_candidate_name
                params = best_candidate_params or "params=auto"
            steps.append(AutoTuneRecipeStep(key=key, label=label, method=method, params=params))

    data_mode = "有参考响应" if target_response_available else "无参考标签"
    notes = [
        f"目标倾向：{goal}",
        f"区域范围：{_roi_mode_text(roi_mode)}",
        f"评分来源：{backend_mode} · {data_mode}",
    ]
    if best_score:
        notes.append(f"综合分数：{best_score:.2f}")
    return AutoTuneRecipe(
        target_goal=goal,
        roi_mode=str(roi_mode or "none"),
        steps=tuple(steps),
        score=float(best_score or 0.0),
        data_mode=data_mode,
        notes=tuple(notes),
    )


def _roi_mode_text(roi_mode: str | None) -> str:
    return {
        "none": "全图",
        "auto": "自动建议区域",
        "manual": "手动框选",
    }.get(str(roi_mode or "none"), "全图")


__all__ = [
    "AutoTuneRecipe",
    "AutoTuneRecipeStep",
    "build_workflow_recipe",
    "resolve_recipe_goal",
]
