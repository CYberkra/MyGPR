#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Identity marker used to prevent deleting an arbitrary directory as a project."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.storage_primitives import atomic_write_json

PROJECT_ROOT_MARKER_RELATIVE = Path("metadata") / "project_root_marker.json"
PROJECT_ROOT_MARKER_SCHEMA = "mygpr.project_root_marker.v1"


def ensure_project_root_marker(root: str | Path, project_id: str) -> Path:
    root_path = Path(root).resolve()
    marker = root_path / PROJECT_ROOT_MARKER_RELATIVE
    expected = {
        "schema": PROJECT_ROOT_MARKER_SCHEMA,
        "project_id": str(project_id),
    }
    if marker.exists():
        if marker.is_symlink() or not marker.is_file():
            raise ValueError(f"项目根标记不安全：{marker}")
        try:
            payload: Any = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"项目根标记损坏：{marker}") from exc
        if payload.get("schema") != expected["schema"] or str(payload.get("project_id") or "") != expected["project_id"]:
            raise ValueError(f"项目根标记与工程身份不一致：{marker}")
        if payload != expected:
            atomic_write_json(marker, expected)
        return marker
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, expected)
    return marker


def validate_project_root_marker(root: str | Path, project_id: str) -> Path:
    root_path = Path(root).resolve()
    marker = root_path / PROJECT_ROOT_MARKER_RELATIVE
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("缺少有效的 MyGPR 项目根标记，拒绝永久删除。")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("MyGPR 项目根标记不可读取，拒绝永久删除。") from exc
    if payload.get("schema") != PROJECT_ROOT_MARKER_SCHEMA:
        raise ValueError("MyGPR 项目根标记版本无效，拒绝永久删除。")
    if str(payload.get("project_id") or "") != str(project_id):
        raise ValueError("MyGPR 项目根标记与 project.json 不一致，拒绝永久删除。")
    return marker


__all__ = [
    "PROJECT_ROOT_MARKER_RELATIVE",
    "ensure_project_root_marker",
    "validate_project_root_marker",
]
