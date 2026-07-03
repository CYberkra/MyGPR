#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-level events used to keep MyGPR modules in sync.

The UI should not guess which downstream pages need refreshing after every
operation.  Callers emit a compact ProjectEvent and the dependency rules/state
tracker decide which derived modules become stale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.field_project_models import local_now


class ProjectEventType:
    PROJECT_OPENED = "PROJECT_OPENED"
    PROJECT_CLOSED = "PROJECT_CLOSED"
    PROJECT_DELETED = "PROJECT_DELETED"
    LINE_SELECTED = "LINE_SELECTED"
    LINE_IMPORTED = "LINE_IMPORTED"
    LINE_DELETED = "LINE_DELETED"
    LINE_SOURCE_RELINKED = "LINE_SOURCE_RELINKED"
    LINE_SOURCE_STATUS_CHECKED = "LINE_SOURCE_STATUS_CHECKED"
    TRAJECTORY_IMPORTED = "TRAJECTORY_IMPORTED"
    QC_UPDATED = "QC_UPDATED"
    BSCAN_ORIENTATION_FIXED = "BSCAN_ORIENTATION_FIXED"
    PROCESSING_RESULT_SAVED = "PROCESSING_RESULT_SAVED"
    PROCESSING_RESULT_DELETED = "PROCESSING_RESULT_DELETED"
    TARGETS_CHANGED = "TARGETS_CHANGED"
    TARGET_SELECTED = "TARGET_SELECTED"
    SPATIAL_MARKED_STALE = "SPATIAL_MARKED_STALE"
    SPATIAL_RESULTS_REFRESHED = "SPATIAL_RESULTS_REFRESHED"
    SPATIAL_EXPORT_GENERATED = "SPATIAL_EXPORT_GENERATED"
    REPORT_MARKED_STALE = "REPORT_MARKED_STALE"
    REPORT_GENERATED = "REPORT_GENERATED"


@dataclass(frozen=True)
class ProjectEvent:
    event_type: str
    project_root: str
    line_id: str | None = None
    reason: str = ""
    affected_modules: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=local_now)
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: str,
        *,
        project_root: str | Path,
        line_id: str | None = None,
        reason: str = "",
        affected_modules: list[str] | None = None,
        changed_paths: list[str | Path] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> "ProjectEvent":
        return cls(
            event_type=str(event_type),
            project_root=str(Path(project_root).resolve()),
            line_id=str(line_id) if line_id else None,
            reason=str(reason or ""),
            affected_modules=list(affected_modules or []),
            changed_paths=[str(path) for path in (changed_paths or [])],
            payload=dict(payload or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ProjectEvent", "ProjectEventType"]
