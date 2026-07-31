#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crash marker and recovery evidence for one opened project session."""
from __future__ import annotations

import os
import socket
import uuid
from pathlib import Path
from typing import Any

from core.field_project_models import local_now
from core.storage_primitives import atomic_write_json

PROJECT_RUNTIME_SCHEMA = "mygpr.project_runtime_session.v1"


class ProjectRuntimeGuard:
    """Persist whether the previous writable GUI session closed cleanly.

    The marker is informational and never blocks opening a project.  It allows
    the project-open path to run a linkage/integrity audit after an unclean exit
    and gives field users a concrete recovery record rather than a generic
    "the application may have crashed" warning.
    """

    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve()
        self.path = self.root / "metadata" / "runtime_session.json"
        self.session_id = uuid.uuid4().hex

    def read_previous(self) -> dict[str, Any] | None:
        try:
            import json
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def previous_unclean(self) -> bool:
        previous = self.read_previous()
        return bool(previous and previous.get("state") == "open")

    def mark_open(self, *, active_workspace: str = "", line_id: str = "") -> dict[str, Any]:
        previous = self.read_previous()
        payload = {
            "schema": PROJECT_RUNTIME_SCHEMA,
            "session_id": self.session_id,
            "state": "open",
            "opened_at": local_now(),
            "closed_at": "",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "active_workspace": str(active_workspace or ""),
            "selected_line_id": str(line_id or ""),
            "previous_unclean": bool(previous and previous.get("state") == "open"),
            "previous_session": {
                "session_id": str((previous or {}).get("session_id") or ""),
                "opened_at": str((previous or {}).get("opened_at") or ""),
                "active_workspace": str((previous or {}).get("active_workspace") or ""),
                "selected_line_id": str((previous or {}).get("selected_line_id") or ""),
            },
        }
        atomic_write_json(self.path, payload)
        return payload

    def checkpoint(self, *, active_workspace: str = "", line_id: str = "") -> None:
        payload = self.read_previous() or {}
        if str(payload.get("session_id") or "") != self.session_id:
            return
        payload.update(
            {
                "schema": PROJECT_RUNTIME_SCHEMA,
                "state": "open",
                "active_workspace": str(active_workspace or payload.get("active_workspace") or ""),
                "selected_line_id": str(line_id or payload.get("selected_line_id") or ""),
                "checkpoint_at": local_now(),
            }
        )
        atomic_write_json(self.path, payload)

    def mark_clean_close(self, *, active_workspace: str = "", line_id: str = "") -> None:
        payload = self.read_previous() or {}
        if payload and str(payload.get("session_id") or "") not in {"", self.session_id}:
            return
        payload.update(
            {
                "schema": PROJECT_RUNTIME_SCHEMA,
                "session_id": self.session_id,
                "state": "closed",
                "closed_at": local_now(),
                "active_workspace": str(active_workspace or payload.get("active_workspace") or ""),
                "selected_line_id": str(line_id or payload.get("selected_line_id") or ""),
            }
        )
        atomic_write_json(self.path, payload)


__all__ = ["PROJECT_RUNTIME_SCHEMA", "ProjectRuntimeGuard"]
