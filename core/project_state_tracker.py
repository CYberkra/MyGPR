#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent project-state tracker for cross-module linkage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.field_project_models import atomic_write_json, local_now
from core.project_dependency_rules import resolve_event_impact
from core.project_events import ProjectEvent, ProjectEventType
from core.schema_registry import DEFAULT_SCHEMA_REGISTRY

PROJECT_STATE_SCHEMA = "mygpr.project_state.v1"
STATE_MODULES = ("processing", "targets", "spatial", "report")
MAX_EVENTS = 30

DATA_REVISION_EVENTS = {
    ProjectEventType.LINE_IMPORTED,
    ProjectEventType.LINE_DELETED,
    ProjectEventType.LINE_SOURCE_RELINKED,
    ProjectEventType.LINE_SOURCE_STATUS_CHECKED,
    ProjectEventType.TRAJECTORY_IMPORTED,
    ProjectEventType.QC_UPDATED,
    ProjectEventType.BSCAN_ORIENTATION_FIXED,
    ProjectEventType.PROCESSING_RESULT_SAVED,
    ProjectEventType.PROCESSING_RESULT_DELETED,
    ProjectEventType.TARGETS_CHANGED,
    ProjectEventType.ANNOTATION_SAVED,
    ProjectEventType.ANNOTATION_CONFIRMED,
    ProjectEventType.SPATIAL_MARKED_STALE,
    ProjectEventType.SPATIAL_RESULTS_REFRESHED,
    ProjectEventType.SPATIAL_EXPORT_GENERATED,
}


def default_project_state() -> dict[str, Any]:
    return {
        "schema": PROJECT_STATE_SCHEMA,
        "data_revision": 0,
        "selected_line_id": None,
        "dirty": {key: False for key in STATE_MODULES},
        "stale_reasons": {key: [] for key in STATE_MODULES},
        "last_events": [],
        "updated_at": local_now(),
    }


class ProjectStateTracker:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.path = self.project_root / "metadata" / "project_state.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            state = default_project_state()
            self.save(state)
            return state
        loaded = DEFAULT_SCHEMA_REGISTRY.load_path(
            self.path,
            family="mygpr.project_state",
            quarantine_root=self.path.parent / "quarantine",
        )
        if loaded.read_only:
            raise PermissionError("项目状态由更高版本 MyGPR 创建，只能以只读恢复模式打开。")
        payload = loaded.payload
        dirty = payload.setdefault("dirty", {})
        reasons = payload.setdefault("stale_reasons", {})
        for key in STATE_MODULES:
            dirty.setdefault(key, False)
            reasons.setdefault(key, [])
        payload.setdefault("last_events", [])
        payload.setdefault("data_revision", 0)
        payload.setdefault("selected_line_id", None)
        payload.setdefault("updated_at", local_now())
        return payload

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, state)

    def record_event(self, event: ProjectEvent) -> dict[str, Any]:
        state = self.load()
        impact = resolve_event_impact(event)
        dirty = state.setdefault("dirty", {})
        reasons = state.setdefault("stale_reasons", {})
        for key in STATE_MODULES:
            dirty.setdefault(key, False)
            reasons.setdefault(key, [])

        if event.event_type == ProjectEventType.LINE_SELECTED:
            state["selected_line_id"] = event.line_id

        for module in impact.clear_modules:
            if module in dirty:
                dirty[module] = False
                reasons[module] = []

        for module in impact.dirty_modules:
            if module in dirty:
                dirty[module] = True

        if impact.spatial_stale:
            dirty["spatial"] = True
            _append_unique(reasons["spatial"], impact.spatial_reason)
        if impact.report_stale:
            dirty["report"] = True
            _append_unique(reasons["report"], impact.report_reason)

        if event.event_type == ProjectEventType.REPORT_GENERATED:
            dirty["report"] = False
            reasons["report"] = []
        if event.event_type == ProjectEventType.SPATIAL_RESULTS_REFRESHED:
            dirty["spatial"] = False
            reasons["spatial"] = []

        if event.event_type in DATA_REVISION_EVENTS:
            state["data_revision"] = int(state.get("data_revision") or 0) + 1
        state["updated_at"] = local_now()
        event_payload = event.to_dict()
        last_events = list(state.get("last_events") or [])
        last_events.append(event_payload)
        state["last_events"] = last_events[-MAX_EVENTS:]
        self.save(state)
        return state

    def mark_report_generated(self) -> dict[str, Any]:
        return self.record_event(ProjectEvent.create(ProjectEventType.REPORT_GENERATED, project_root=self.project_root, reason="成果报告已生成"))


def _append_unique(values: list[Any], value: str) -> None:
    text = str(value or "").strip()
    if not text:
        return
    if text not in values:
        values.append(text)
    del values[:-8]


def load_project_state(project_root: str | Path) -> dict[str, Any]:
    return ProjectStateTracker(project_root).load()


def record_project_event(event: ProjectEvent) -> dict[str, Any]:
    return ProjectStateTracker(event.project_root).record_event(event)


__all__ = [
    "PROJECT_STATE_SCHEMA",
    "ProjectStateTracker",
    "default_project_state",
    "load_project_state",
    "record_project_event",
]
