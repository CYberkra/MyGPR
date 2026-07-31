#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent cross-module workspace context for the five-page field workbench.

The project-state tracker answers whether derived modules are stale.  This file
answers a different question: *which concrete line/version should each page
open when the user follows a linkage action?*  Keeping that context outside the
widgets prevents direct page-to-page coupling and makes close/reopen restore the
same engineering object rather than merely the same page.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.field_project_models import atomic_write_json, local_now
from core.schema_registry import DEFAULT_SCHEMA_REGISTRY

WORKSPACE_CONTEXT_SCHEMA = "mygpr.workspace_context.v1"
VALID_WORKSPACES = {
    "data_management",
    "processing_lab",
    "interpretation",
    "spatial",
    "delivery",
}


def default_workspace_context() -> dict[str, Any]:
    return {
        "schema": WORKSPACE_CONTEXT_SCHEMA,
        "active_workspace": "data_management",
        "selected_line_id": None,
        "processing_source_by_line": {},
        "annotation_by_line": {},
        "selected_spatial_result_id": "",
        "selected_report_id": "",
        "last_handoff": {},
        "updated_at": local_now(),
    }


class WorkspaceContextStore:
    """Read/write the current five-page engineering context atomically."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.path = self.project_root / "metadata" / "workspace_context.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            payload = default_workspace_context()
            self.save(payload)
            return payload
        loaded = DEFAULT_SCHEMA_REGISTRY.load_path(
            self.path,
            family="mygpr.workspace_context",
            quarantine_root=self.path.parent / "quarantine",
        )
        if loaded.read_only:
            raise PermissionError("工作区上下文由更高版本 MyGPR 创建，只能以只读恢复模式打开。")
        payload = dict(loaded.payload)
        payload.setdefault("active_workspace", "data_management")
        payload.setdefault("selected_line_id", None)
        payload.setdefault("processing_source_by_line", {})
        payload.setdefault("annotation_by_line", {})
        payload.setdefault("selected_spatial_result_id", "")
        payload.setdefault("selected_report_id", "")
        payload.setdefault("last_handoff", {})
        payload.setdefault("updated_at", local_now())
        if payload["active_workspace"] not in VALID_WORKSPACES:
            payload["active_workspace"] = "data_management"
        return payload

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        data["schema"] = WORKSPACE_CONTEXT_SCHEMA
        if data.get("active_workspace") not in VALID_WORKSPACES:
            data["active_workspace"] = "data_management"
        data["updated_at"] = local_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, data)
        return data

    def update(self, **changes: Any) -> dict[str, Any]:
        payload = self.load()
        payload.update(changes)
        return self.save(payload)

    def set_workspace(self, workspace: str) -> dict[str, Any]:
        return self.update(active_workspace=workspace if workspace in VALID_WORKSPACES else "data_management")

    def set_line(self, line_id: str | None) -> dict[str, Any]:
        return self.update(selected_line_id=str(line_id) if line_id else None)

    def bind_processing_source(self, line_id: str, source_result_id: str) -> dict[str, Any]:
        payload = self.load()
        mapping = dict(payload.get("processing_source_by_line") or {})
        if line_id and source_result_id:
            mapping[str(line_id)] = str(source_result_id)
        payload["processing_source_by_line"] = mapping
        payload["selected_line_id"] = str(line_id) if line_id else payload.get("selected_line_id")
        return self.save(payload)

    def bind_annotation(
        self,
        line_id: str,
        *,
        version: str = "",
        status: str = "",
        source_result_id: str = "",
    ) -> dict[str, Any]:
        payload = self.load()
        mapping = dict(payload.get("annotation_by_line") or {})
        mapping[str(line_id)] = {
            "version": str(version or ""),
            "status": str(status or ""),
            "source_result_id": str(source_result_id or ""),
            "updated_at": local_now(),
        }
        payload["annotation_by_line"] = mapping
        payload["selected_line_id"] = str(line_id) if line_id else payload.get("selected_line_id")
        return self.save(payload)

    def set_spatial_result(self, result_id: str | None) -> dict[str, Any]:
        return self.update(selected_spatial_result_id=str(result_id or ""))

    def set_report(self, report_id: str | None) -> dict[str, Any]:
        return self.update(selected_report_id=str(report_id or ""))

    def record_handoff(
        self,
        *,
        source_workspace: str,
        target_workspace: str,
        line_id: str | None = None,
        artifact_type: str = "",
        artifact_id: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        payload = self.load()
        payload["active_workspace"] = target_workspace if target_workspace in VALID_WORKSPACES else "data_management"
        if line_id:
            payload["selected_line_id"] = str(line_id)
        payload["last_handoff"] = {
            "source_workspace": str(source_workspace or ""),
            "target_workspace": str(target_workspace or ""),
            "line_id": str(line_id or ""),
            "artifact_type": str(artifact_type or ""),
            "artifact_id": str(artifact_id or ""),
            "reason": str(reason or ""),
            "created_at": local_now(),
        }
        return self.save(payload)


__all__ = [
    "VALID_WORKSPACES",
    "WORKSPACE_CONTEXT_SCHEMA",
    "WorkspaceContextStore",
    "default_workspace_context",
]
