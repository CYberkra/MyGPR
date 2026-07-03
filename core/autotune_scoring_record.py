#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serializable AutoTune scoring v2 record helpers.

This module closes the gap between internal scoring and auditable UI/report
outputs.  It keeps the record compact enough for GUI display while preserving
all numeric score terms needed for trial tables, exported reports and future
Evidence manifests.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from core.autotune_goal_profiles import resolve_goal_profile


def _float_dict(values: Mapping | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in dict(values or {}).items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def _compact_terms(terms: Mapping | None) -> dict[str, float]:
    """Return report-friendly v2 terms, stripping legacy aliases."""
    raw = _float_dict(terms)
    preferred = [
        "background_suppression",
        "response_preservation",
        "continuity",
        "contrast",
        "deep_balance",
        "artifact_control",
        "stability",
        "texture_preservation",
        "fracture_response",
        "attenuation_preservation",
        "gain_stability",
        "reference_similarity",
        "background_candidate_score",
        "workflow_fit",
        "compactness",
        "target_response_available",
    ]
    compact: dict[str, float] = {}
    for key in preferred:
        if key in raw:
            compact[key] = raw[key]
        elif f"v2_{key}" in raw:
            compact[key] = raw[f"v2_{key}"]
    for key, value in raw.items():
        if key.startswith("v2_") and key[3:] not in compact:
            compact[key[3:]] = value
    return compact


def _notes_from_record(row: Mapping, *, target_response_available: bool) -> list[str]:
    notes: list[str] = []
    if target_response_available:
        notes.append("检测到参考响应，reference_similarity 已参与 scoring v2。")
    else:
        notes.append("未检测到参考响应，scoring v2 使用无参考标签启发式指标。")
    message = str(row.get("note") or row.get("warning") or row.get("status") or "").strip()
    if message:
        notes.append(message)
    if row.get("background_low_benefit"):
        low_msg = "背景抑制收益较弱，已采用温和背景抑制方法。"
        if low_msg not in notes:
            notes.append(low_msg)
    return list(dict.fromkeys(notes))


def build_scoring_v2_record(
    row: Mapping,
    *,
    target_goal: str | None,
    roi_mode: str | None,
    target_response_available: bool,
) -> dict:
    """Build a stable scoring v2 record for candidate/UI/report outputs."""
    profile = resolve_goal_profile(target_goal or row.get("target_goal"))
    workflow_breakdown = row.get("workflow_score_breakdown") or row.get("score_breakdown") or {}
    background = row.get("background_candidate") if isinstance(row.get("background_candidate"), Mapping) else {}
    background_breakdown = {}
    if isinstance(background, Mapping):
        background_breakdown = dict(background.get("score_breakdown") or {})
    background_terms = {}
    if isinstance(background_breakdown, Mapping):
        background_terms = _compact_terms(background_breakdown.get("terms"))
    if not background_terms and isinstance(background, Mapping):
        background_terms = _compact_terms(background.get("scoring_terms"))

    workflow_terms = {}
    workflow_weights = {}
    if isinstance(workflow_breakdown, Mapping):
        workflow_terms = _compact_terms(workflow_breakdown.get("terms") or row.get("scoring_terms"))
        workflow_weights = _float_dict(workflow_breakdown.get("weights") or row.get("workflow_score_weights"))
    if not workflow_terms:
        workflow_terms = _compact_terms(row.get("scoring_terms"))
    if not workflow_weights:
        workflow_weights = _float_dict(row.get("workflow_score_weights"))

    record = {
        "autotune_scoring_version": "autotune_scoring_v2",
        "target_goal": profile.label,
        "roi_mode": str(roi_mode or row.get("roi_mode") or "none"),
        "data_mode": "有参考响应" if target_response_available else "无参考标签",
        "final_score": float(row.get("score", 0.0) or 0.0),
        "goal_profile": profile.to_dict(),
        "goal_weights": dict(profile.weights),
        "workflow_score": {
            "score": float(row.get("score", 0.0) or 0.0),
            "terms": workflow_terms,
            "weights": workflow_weights,
        },
        "background_score": {
            "method": str(background.get("method") or row.get("method") or ""),
            "name": str(background.get("name") or row.get("name") or ""),
            "params": str(background.get("params") or ""),
            "score": float(background.get("score", 0.0) or 0.0),
            "terms": background_terms,
            "weights": _float_dict(background_breakdown.get("weights") if isinstance(background_breakdown, Mapping) else {}),
            "breakdown": background_breakdown,
        },
        "diagnostics": dict(row.get("diagnostics") or {}),
        "candidate_space": {
            "candidate_space_hash": row.get("candidate_space_hash") or background.get("candidate_space_hash"),
            "profile_id": row.get("candidate_space_profile_id") or background.get("candidate_space_profile_id"),
            "config_version": row.get("candidate_space_config_version") or background.get("candidate_space_config_version"),
            "recipe_ids": list(row.get("candidate_space_recipe_ids") or background.get("candidate_space_recipe_ids") or ()),
            "context": dict(row.get("candidate_space_context") or background.get("candidate_space_context") or {}),
        },
        "selection_rule": str(row.get("selection_rule") or "bounded_recipe_scoring_v2"),
        "notes": _notes_from_record(row, target_response_available=target_response_available),
    }
    return record


def summarize_terms(terms: Mapping | None, *, limit: int = 6) -> str:
    """Format score terms for compact UI text."""
    compact = _compact_terms(terms)
    if not compact:
        return "--"
    items = sorted(compact.items(), key=lambda kv: abs(kv[1]), reverse=True)[: max(1, int(limit))]
    return ", ".join(f"{key}={value:.2f}" for key, value in items)


def summarize_record(record: Mapping | None) -> str:
    if not record:
        return "scoring v2 记录未生成。"
    workflow = record.get("workflow_score") if isinstance(record.get("workflow_score"), Mapping) else {}
    background = record.get("background_score") if isinstance(record.get("background_score"), Mapping) else {}
    lines = [
        f"scoring v2：{float(record.get('final_score', 0.0) or 0.0):.2f}",
        f"目标权重：{summarize_terms(record.get('goal_weights'), limit=7)}",
        f"流程评分：{summarize_terms(workflow.get('terms'), limit=4)}",
    ]
    if background:
        lines.append(
            f"背景候选：{background.get('name', '--')} | {summarize_terms(background.get('terms'), limit=4)}"
        )
    notes = record.get("notes") if isinstance(record.get("notes"), Sequence) else []
    for note in list(notes)[:3]:
        lines.append(f"说明：{note}")
    return "\n".join(lines)


__all__ = ["build_scoring_v2_record", "summarize_record", "summarize_terms"]
