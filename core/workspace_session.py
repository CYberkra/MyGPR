#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global workbench layouts and per-project document sessions."""

from __future__ import annotations

import json
from typing import Any

from core.app_paths import get_settings_dir
from core.project_service import ProjectService, atomic_write_json


class WorkspaceSessionService:
    def __init__(self, project: ProjectService):
        self.project = project

    def save_global_layout(self, workspace: str, layout: dict[str, Any]) -> None:
        self.save_global_layout_for(workspace, layout)

    @classmethod
    def save_global_layout_for(cls, workspace: str, layout: dict[str, Any]) -> None:
        path = cls._global_path()
        payload = cls._read(path)
        payload[str(workspace)] = layout
        atomic_write_json(path, payload)

    def load_global_layout(self, workspace: str) -> dict[str, Any]:
        return self.load_global_layout_for(workspace)

    @classmethod
    def load_global_layout_for(cls, workspace: str) -> dict[str, Any]:
        return dict(cls._read(cls._global_path()).get(str(workspace), {}))

    def save_project_session(
        self,
        *,
        open_documents: list[str],
        selected_line_id: str | None,
        active_workspace: str,
    ) -> None:
        atomic_write_json(
            self.project.resolve_relative_path("workspace/session.json"),
            {
                "schema": "mygpr.workspace_session.v1",
                "open_documents": list(open_documents),
                "selected_line_id": selected_line_id,
                "active_workspace": active_workspace,
            },
        )

    def load_project_session(self) -> dict[str, Any]:
        return self._read(self.project.resolve_relative_path("workspace/session.json"))

    @staticmethod
    def _global_path():
        from pathlib import Path

        return Path(get_settings_dir()) / "workbench_layouts.json"

    @staticmethod
    def _read(path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}


__all__ = ["WorkspaceSessionService"]
