#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow registry contract helpers for MyGPR Studio.

This module is a non-invasive registry facade.  It validates the current
workflow presets, stage definitions, and canvas output recommendations without
moving the legacy data structures yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.methods_registry import PROCESSING_METHODS
from core.preset_profiles import RECOMMENDED_RUN_PROFILES, WORKFLOW_PRESETS
from core.workflow_data import (
    METHOD_CATEGORIES,
    QUICK_PRESETS,
    WORKFLOW_STAGE_DEFINITIONS,
    WORKFLOW_STAGE_BY_ID,
    WorkflowMethod,
)


OUTPUT_EFFECT_LABELS: dict[str, str] = {
    "bscan": "查看此步 B-scan",
    "compare": "前后对比",
    "qc": "QC 指标",
    "spectrum": "频谱 / 能量分布",
    "evidence": "导出此步结果",
}

OUTPUT_EFFECT_TITLES: dict[str, str] = {
    "bscan": "B-scan Preview",
    "compare": "Before / After",
    "qc": "QC Metrics",
    "spectrum": "Spectrum View",
    "evidence": "Evidence Snapshot",
}

OUTPUT_EFFECT_EMPTY_MESSAGES: dict[str, str] = {
    "bscan": "暂无预览，请先运行到来源节点",
    "compare": "暂无对比，请先运行到来源节点",
    "qc": "暂无 QC 指标，请先运行到来源节点",
    "spectrum": "暂无频谱，请先运行到来源节点",
    "evidence": "暂无 Evidence，请先运行到来源节点",
}

NEXT_METHOD_RECOMMENDATIONS: dict[str, list[str]] = {
    "zero_time": ["dewow", "frequency_filter_1d"],
    "trace_correction": ["subtracting_average_2D", "frequency_filter_1d"],
    "background_clutter": ["sec_gain", "wavelet_svd", "svd_subspace"],
    "gain": ["svd_subspace", "wavelet_svd"],
    "spatial_denoise": ["kirchhoff_migration", "stolt_migration", "time_to_depth"],
    "velocity_model": ["geometry_depth_context"],
    "geometry_depth": ["sec_gain", "kirchhoff_migration"],
}

METHOD_FALLBACK_RECOMMENDATIONS: dict[str, list[str]] = {
    "set_zero_time": ["dewow", "frequency_filter_1d"],
    "dewow": ["subtracting_average_2D", "frequency_filter_1d"],
    "subtracting_average_2D": ["sec_gain", "wavelet_svd", "svd_subspace"],
    "sec_gain": ["svd_subspace", "wavelet_svd"],
    "agcGain": ["svd_subspace", "wavelet_svd"],
    "energy_decay_gain": ["svd_subspace", "wavelet_svd"],
    "compensatingGain": ["svd_subspace", "wavelet_svd"],
}

MYGPR_STANDARD_METHOD_ORDER: list[str] = [
    "set_zero_time",
    "dewow",
    "subtracting_average_2D",
    "sec_gain",
    "svd_subspace",
]


@dataclass(frozen=True)
class WorkflowStageSpec:
    """Typed view of one workflow stage definition."""

    stage_id: str
    label: str
    default_method: str
    candidate_methods: tuple[str, ...]
    warning: str = ""


@dataclass(frozen=True)
class WorkflowPresetSpec:
    """Typed view of a quick workflow preset."""

    preset_key: str
    name: str
    method_ids: tuple[str, ...]
    stage_ids: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowRegistryIssue:
    """A registry consistency issue."""

    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class WorkflowRegistrySnapshot:
    """Report-friendly overview of the workflow registry."""

    stage_count: int
    quick_preset_count: int
    recommended_profile_count: int
    output_effect_kinds: tuple[str, ...]
    issues: tuple[WorkflowRegistryIssue, ...] = field(default_factory=tuple)


def stage_specs() -> list[WorkflowStageSpec]:
    """Return typed workflow stage definitions."""
    specs: list[WorkflowStageSpec] = []
    for item in WORKFLOW_STAGE_DEFINITIONS:
        specs.append(
            WorkflowStageSpec(
                stage_id=str(item.get("id", "")),
                label=str(item.get("label", "")),
                default_method=str(item.get("default_method", "")),
                candidate_methods=tuple(str(key) for key in item.get("candidate_methods", [])),
                warning=str(item.get("warning", "")),
            )
        )
    return specs


def quick_preset_spec(preset_key: str) -> WorkflowPresetSpec:
    """Return a typed view of one built-in quick preset."""
    preset = QUICK_PRESETS[preset_key]
    methods = list(preset.get("methods", []))
    return WorkflowPresetSpec(
        preset_key=preset_key,
        name=str(preset.get("name", preset_key)),
        method_ids=tuple(str(item.get("method_id", "")) for item in methods),
        stage_ids=tuple(str(item.get("stage_id", "")) for item in methods),
    )


def default_params_for_method(method_id: str) -> dict[str, object]:
    """Return registry defaults for a processing method."""
    params: dict[str, object] = {}
    for meta in PROCESSING_METHODS.get(str(method_id), {}).get("params", []):
        name = str(meta.get("name", ""))
        if name:
            params[name] = meta.get("default", "")
    return params


def candidate_methods_for_workflow_method(method: WorkflowMethod) -> list[str]:
    """Return valid candidate method IDs for a workflow node."""
    stage = WORKFLOW_STAGE_BY_ID.get(getattr(method, "stage_id", ""), {})
    candidates = [
        str(item)
        for item in stage.get("candidate_methods", [])
        if str(item) in PROCESSING_METHODS
    ]
    if not candidates:
        method_id = str(getattr(method, "method_id", ""))
        if method_id in PROCESSING_METHODS:
            candidates = [method_id]
        category = METHOD_CATEGORIES.get(str(getattr(method, "category", "")), {})
        for key in category.get("methods", []):
            if key in PROCESSING_METHODS and key not in candidates:
                candidates.append(str(key))
    method_id = str(getattr(method, "method_id", ""))
    if method_id in PROCESSING_METHODS and method_id not in candidates:
        candidates.insert(0, method_id)
    return candidates


def recommended_next_methods_for(method: WorkflowMethod | None = None, *, method_id: str = "", stage_id: str = "") -> list[str]:
    """Return recommended next processing methods for an output-drop action."""
    current_method = str(method_id or getattr(method, "method_id", ""))
    current_stage = str(stage_id or getattr(method, "stage_id", ""))
    stage_methods = list(NEXT_METHOD_RECOMMENDATIONS.get(current_stage, []))
    if not stage_methods:
        stage_methods = list(METHOD_FALLBACK_RECOMMENDATIONS.get(current_method, []))

    result: list[str] = []
    for candidate in stage_methods:
        if candidate in PROCESSING_METHODS and candidate != current_method and candidate not in result:
            result.append(candidate)
    return result


def workflow_port_labels(method: WorkflowMethod) -> tuple[str, str]:
    """Return compact visual input/output labels for current first-pass ports."""
    method_id = str(getattr(method, "method_id", ""))
    stage_id = str(getattr(method, "stage_id", ""))
    category = str(getattr(method, "category", ""))
    if method_id == "raw_input":
        return "source", "data"
    if method_id == "bscan_preview":
        return "data", "preview"
    if method_id == "manual_velocity_model" or stage_id == "velocity_model":
        return "data", "velocity"
    if method_id in {"geometry_depth_context", "time_to_depth"} or stage_id == "geometry_depth":
        return "data / velocity", "depth"
    if method_id in {"kirchhoff_migration", "stolt_migration"} or category == "migration":
        return "data / velocity", "image"
    if category in {"export", "evidence"}:
        return "data", "evidence"
    return "data", "data"


def validate_workflow_registry() -> list[WorkflowRegistryIssue]:
    """Validate the current workflow registry facade."""
    issues: list[WorkflowRegistryIssue] = []

    def add(severity: str, code: str, message: str) -> None:
        issues.append(WorkflowRegistryIssue(severity, code, message))

    stage_ids = set()
    for stage in stage_specs():
        if not stage.stage_id:
            add("error", "empty_stage_id", f"Workflow stage has empty id: {stage!r}")
            continue
        if stage.stage_id in stage_ids:
            add("error", "duplicate_stage_id", f"Duplicate workflow stage id: {stage.stage_id}")
        stage_ids.add(stage.stage_id)
        if stage.default_method not in PROCESSING_METHODS:
            add("error", "missing_stage_default", f"{stage.stage_id} default method not registered: {stage.default_method}")
        for method_id in stage.candidate_methods:
            if method_id not in PROCESSING_METHODS:
                add("error", "missing_stage_candidate", f"{stage.stage_id} candidate method not registered: {method_id}")

    for category_id, category in METHOD_CATEGORIES.items():
        for method_id in category.get("methods", []):
            if method_id not in PROCESSING_METHODS:
                add("error", "missing_category_method", f"{category_id} method not registered: {method_id}")

    for preset_key, preset in QUICK_PRESETS.items():
        methods = list(preset.get("methods", []))
        if not methods:
            add("warning", "empty_quick_preset", f"Quick preset has no methods: {preset_key}")
        for index, method in enumerate(methods):
            method_id = str(method.get("method_id", ""))
            stage_id = str(method.get("stage_id", ""))
            if method_id not in PROCESSING_METHODS:
                add("error", "missing_preset_method", f"{preset_key}[{index}] method not registered: {method_id}")
            if stage_id and stage_id not in stage_ids:
                add("error", "missing_preset_stage", f"{preset_key}[{index}] stage not registered: {stage_id}")

    for profile_key, profile in RECOMMENDED_RUN_PROFILES.items():
        for method_id in profile.get("order", []):
            if method_id not in PROCESSING_METHODS:
                add("error", "missing_profile_method", f"{profile_key} method not registered: {method_id}")

    for profile_key, workflow in WORKFLOW_PRESETS.items():
        stage_config = workflow.get("stages", {}) if isinstance(workflow, dict) else {}
        for stage_key, enabled_methods in stage_config.items():
            if not isinstance(enabled_methods, dict):
                add("error", "invalid_workflow_stage", f"{profile_key}.{stage_key} must be a method map")
                continue
            for method_id, enabled in enabled_methods.items():
                if enabled and method_id not in PROCESSING_METHODS:
                    add("error", "missing_workflow_preset_method", f"{profile_key}.{stage_key} method not registered: {method_id}")

    mygpr_spec = quick_preset_spec("mygpr_standard")
    if list(mygpr_spec.method_ids) != MYGPR_STANDARD_METHOD_ORDER:
        add(
            "error",
            "mygpr_standard_order_changed",
            "MyGPR standard preset no longer matches the agreed five-step chain.",
        )
    profile_order = RECOMMENDED_RUN_PROFILES.get("mygpr_standard", {}).get("order", [])
    if list(profile_order) != MYGPR_STANDARD_METHOD_ORDER:
        add(
            "error",
            "mygpr_standard_profile_order_changed",
            "MyGPR standard recommended profile no longer matches the agreed five-step chain.",
        )

    for kind in OUTPUT_EFFECT_LABELS:
        if kind not in OUTPUT_EFFECT_TITLES or kind not in OUTPUT_EFFECT_EMPTY_MESSAGES:
            add("error", "incomplete_output_effect", f"Output effect kind is missing title/empty-message metadata: {kind}")

    return issues


def workflow_registry_snapshot() -> WorkflowRegistrySnapshot:
    """Return a lightweight registry health snapshot."""
    return WorkflowRegistrySnapshot(
        stage_count=len(WORKFLOW_STAGE_DEFINITIONS),
        quick_preset_count=len(QUICK_PRESETS),
        recommended_profile_count=len(RECOMMENDED_RUN_PROFILES),
        output_effect_kinds=tuple(OUTPUT_EFFECT_LABELS),
        issues=tuple(validate_workflow_registry()),
    )
