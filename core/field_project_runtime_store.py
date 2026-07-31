#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime metrics, catalog and audit helpers for formal field projects."""
from __future__ import annotations

import hashlib
import sqlite3
import re
import uuid
from pathlib import Path
from typing import Any

from core.field_project_models import validate_line_id
from core.storage_primitives import utc_now


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _branch_slug(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name or "branch").strip()).strip("-")
    return text[:32] or "branch"


class FieldProjectRuntimeStoreMixin:
    """Provide project-size metrics and the project-level catalog facade."""

    def total_raw_size_mb(self) -> float:
        return sum(line.raw_size_mb for line in self.list_lines())

    def storage_usage_mb(self) -> float:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total / (1024 * 1024)

    def append_log(self, message: str) -> None:
        log_path = self.root / "logs" / "field_workbench.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{self.now()}] {message}\n")
        storage = getattr(self, "storage", None)
        if getattr(storage, "is_hybrid", False):
            try:
                storage.catalog.append_audit(
                    "workbench_log", object_type="project", object_id=self.manifest.project_id,
                    payload={"message": str(message)},
                )
            except (sqlite3.Error, OSError, TypeError, ValueError):
                # The human-readable log must remain available even if the
                # catalog is temporarily locked by another worker.
                pass

    def list_processing_branches(self, line_id: str) -> list[dict[str, Any]]:
        """Return durable processing branches for one line.

        Legacy projects expose a synthetic main branch only at runtime.  It is
        not persisted until the project is explicitly migrated.
        """
        safe_line_id = validate_line_id(line_id)
        storage = getattr(self, "storage", None)
        if not getattr(storage, "is_hybrid", False):
            return [{
                "branch_id": f"{safe_line_id}:main", "line_id": safe_line_id,
                "name": "主分支", "head_artifact_id": "", "status": "legacy",
            }]
        branches = storage.catalog.list_branches(line_id=safe_line_id)
        if not branches and not self.read_only:
            storage.catalog.ensure_branch(
                line_id=safe_line_id, branch_id=f"{safe_line_id}:main", name="主分支"
            )
            branches = storage.catalog.list_branches(line_id=safe_line_id)
        return branches

    def create_processing_branch(
        self,
        line_id: str,
        name: str,
        *,
        from_artifact_id: str = "",
        parent_branch_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a named branch and optionally seed its head from an artifact."""
        self.assert_writable()
        safe_line_id = validate_line_id(line_id)
        display_name = str(name or "新分支").strip() or "新分支"
        storage = getattr(self, "storage", None)
        if not getattr(storage, "is_hybrid", False):
            raise RuntimeError("旧项目需先迁移到 Hybrid Store 才能创建处理分支。")
        branch_id = f"{safe_line_id}:{_branch_slug(display_name)}-{uuid.uuid4().hex[:8]}"
        storage.catalog.ensure_branch(
            line_id=safe_line_id,
            branch_id=branch_id,
            name=display_name,
            parent_branch_id=parent_branch_id,
            head_artifact_id=from_artifact_id,
        )
        storage.catalog.append_audit(
            "processing_branch_created",
            object_type="processing_branch",
            object_id=branch_id,
            payload={
                "line_id": safe_line_id,
                "name": display_name,
                "from_artifact_id": str(from_artifact_id or ""),
                "parent_branch_id": str(parent_branch_id or ""),
            },
        )
        return next(item for item in storage.catalog.list_branches(line_id=safe_line_id) if item["branch_id"] == branch_id)

    def register_project_export(
        self,
        path: str | Path,
        *,
        export_kind: str,
        source_artifact_id: str = "",
        metadata: dict[str, Any] | None = None,
        status: str = "generated",
    ) -> dict[str, Any] | None:
        """Register an immutable external deliverable in the SQLite catalog."""
        storage = getattr(self, "storage", None)
        if not getattr(storage, "is_hybrid", False):
            return None
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        try:
            rel = candidate.relative_to(self.root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("导出登记仅允许项目目录内文件。") from exc
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        payload = {
            "export_id": f"exp_{uuid.uuid4().hex}",
            "export_kind": str(export_kind or "file"),
            "source_artifact_id": str(source_artifact_id or ""),
            "path": rel,
            "status": str(status or "generated"),
            "sha256": _sha256_path(candidate),
            "metadata": {
                **dict(metadata or {}),
                "size_bytes": int(candidate.stat().st_size),
            },
            "created_at": utc_now(),
        }
        storage.catalog.register_export(payload)
        storage.catalog.append_audit(
            "export_registered", object_type="export", object_id=payload["export_id"], payload=payload
        )
        return payload

    def list_project_exports(self, *, export_kind: str | None = None) -> list[dict[str, Any]]:
        storage = getattr(self, "storage", None)
        if not getattr(storage, "is_hybrid", False):
            return []
        return storage.catalog.list_exports(export_kind=export_kind)


__all__ = ["FieldProjectRuntimeStoreMixin"]
