#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crash-recoverable journal for SQLite + HDF5 artifact commits.

SQLite and HDF5 cannot participate in one native ACID transaction.  This
module supplies a small write-ahead journal that makes the intended operation
recoverable and idempotent after process termination or power loss.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.hdf5_line_container import (
    processing_artifact_exists,
    read_processing_artifact_record,
)
from core.storage_primitives import atomic_write_json, fsync_directory, utc_now

HYBRID_ARTIFACT_TRANSACTION_SCHEMA = "mygpr.hybrid_artifact_transaction.v1"


@dataclass(frozen=True, slots=True)
class HybridRecoveryAction:
    transaction_id: str
    artifact_id: str
    line_id: str
    action: str
    success: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "artifact_id": self.artifact_id,
            "line_id": self.line_id,
            "action": self.action,
            "success": self.success,
            "message": self.message,
        }


class HybridArtifactTransaction:
    """One durable artifact commit intent."""

    def __init__(self, journal: "HybridArtifactTransactionJournal", path: Path, payload: dict[str, Any]) -> None:
        self._journal = journal
        self.path = path
        self.payload = dict(payload)

    @property
    def transaction_id(self) -> str:
        return str(self.payload["transaction_id"])

    def update(self, state: str, **changes: Any) -> None:
        self.payload.update(changes)
        self.payload["state"] = str(state)
        self.payload["updated_at"] = utc_now()
        atomic_write_json(self.path, self.payload)

    def complete(self) -> None:
        self.path.unlink(missing_ok=True)
        fsync_directory(self.path.parent)
        self._journal._cleanup_empty_dirs()


class HybridArtifactTransactionJournal:
    """Persistent recovery log for processing artifact commits."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / ".transactions" / "hybrid_artifacts"

    def begin(
        self,
        *,
        line_id: str,
        artifact_id: str,
        branch_id: str,
        parent_artifact_id: str,
        h5_path: str,
        manifest: dict[str, Any],
        params: dict[str, Any],
    ) -> HybridArtifactTransaction:
        self.root.mkdir(parents=True, exist_ok=True)
        transaction_id = uuid.uuid4().hex
        payload = {
            "schema": HYBRID_ARTIFACT_TRANSACTION_SCHEMA,
            "transaction_id": transaction_id,
            "operation": "commit_processing_artifact",
            "state": "prepared",
            "line_id": str(line_id),
            "artifact_id": str(artifact_id),
            "branch_id": str(branch_id),
            "parent_artifact_id": str(parent_artifact_id or ""),
            "h5_path": str(h5_path),
            "manifest": dict(manifest),
            "params": dict(params),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        path = self.root / f"{transaction_id}.json"
        atomic_write_json(path, payload)
        return HybridArtifactTransaction(self, path, payload)

    def pending_paths(self) -> tuple[Path, ...]:
        if not self.root.exists():
            return ()
        return tuple(sorted(path for path in self.root.glob("*.json") if path.is_file()))

    def recover(
        self,
        catalog: Any,
        line_container_path: Callable[[str], Path],
    ) -> tuple[HybridRecoveryAction, ...]:
        actions = [
            self._recover_path(path, catalog, line_container_path)
            for path in self.pending_paths()
        ]
        self._cleanup_empty_dirs()
        return tuple(actions)

    def _recover_path(
        self,
        journal_path: Path,
        catalog: Any,
        line_container_path: Callable[[str], Path],
    ) -> HybridRecoveryAction:
        transaction_id = journal_path.stem
        artifact_id = ""
        line_id = ""
        try:
            payload = self._load_payload(journal_path)
            transaction_id = str(payload.get("transaction_id") or transaction_id)
            artifact_id = str(payload["artifact_id"])
            line_id = str(payload["line_id"])
            container = line_container_path(line_id)
            action = self._reconcile(payload, catalog, container)
            journal_path.unlink(missing_ok=True)
            fsync_directory(journal_path.parent)
            return HybridRecoveryAction(transaction_id, artifact_id, line_id, action, True)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return HybridRecoveryAction(
                transaction_id,
                artifact_id,
                line_id,
                "recovery_failed",
                False,
                str(exc),
            )

    @staticmethod
    def _load_payload(journal_path: Path) -> dict[str, Any]:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
        if payload.get("schema") != HYBRID_ARTIFACT_TRANSACTION_SCHEMA:
            raise ValueError("unsupported hybrid transaction schema")
        if payload.get("operation") != "commit_processing_artifact":
            raise ValueError("unsupported hybrid transaction operation")
        if not str(payload.get("artifact_id") or "") or not str(payload.get("line_id") or ""):
            raise ValueError("transaction is missing line_id or artifact_id")
        return payload

    def _reconcile(self, payload: dict[str, Any], catalog: Any, container: Path) -> str:
        artifact_id = str(payload["artifact_id"])
        line_id = str(payload["line_id"])
        transaction_id = str(payload.get("transaction_id") or "")
        h5_present = processing_artifact_exists(container, artifact_id)
        catalog_row = catalog.get_artifact(artifact_id)
        if h5_present and catalog_row is None:
            self._roll_forward(payload, catalog, container)
            event_type = "hybrid_transaction_roll_forward"
            action = "roll_forward_catalog"
        elif catalog_row is not None and not h5_present:
            catalog.delete_artifact(artifact_id)
            event_type = "hybrid_transaction_rollback_catalog"
            action = "rollback_catalog"
        elif catalog_row is not None and h5_present:
            event_type = "hybrid_transaction_reconciled"
            action = "reconciled_committed"
        else:
            event_type = "hybrid_transaction_discarded"
            action = "discarded_uncommitted"
        catalog.append_audit(
            event_type,
            object_type="processing_artifact",
            object_id=artifact_id,
            payload={"transaction_id": transaction_id, "line_id": line_id},
        )
        return action

    def _artifact_relative_path(self, record: dict[str, Any], payload: dict[str, Any], container: Path) -> str:
        """Resolve the artifact file path for catalog registration.

        Sidecar 写入后 ``record["h5_file"]`` 记录实际落盘文件；崩溃恢复
        （roll_forward）据此把 catalog ``h5_path`` 指向 sidecar。旧 payload
        的 ``h5_path`` 是项目相对路径（legacy 布局），直接原样沿用，不经过
        CWD 锚定的 ``resolve()``。
        """
        h5_file = record.get("h5_file")
        if h5_file:
            resolved = Path(str(h5_file)).resolve()
            try:
                return resolved.relative_to(self.project_root).as_posix()
            except ValueError:
                return resolved.as_posix()
        fallback = str(payload.get("h5_path") or container.relative_to(self.project_root).as_posix())
        if Path(fallback).is_absolute():
            return Path(fallback).as_posix()
        return fallback

    def _roll_forward(self, payload: dict[str, Any], catalog: Any, container: Path) -> None:
        artifact_id = str(payload["artifact_id"])
        line_id = str(payload["line_id"])
        record = read_processing_artifact_record(container, artifact_id)
        manifest = record["manifest"]
        branch_id = str(payload.get("branch_id") or manifest.get("branch_id") or f"{line_id}:main")
        parent_id = str(payload.get("parent_artifact_id") or manifest.get("parent_artifact_id") or "")
        catalog.ensure_branch(
            line_id=line_id,
            branch_id=branch_id,
            name=branch_id,
            head_artifact_id=parent_id,
        )
        catalog.register_artifact(
            {
                "artifact_id": artifact_id,
                "line_id": line_id,
                "artifact_kind": "processing",
                "artifact_role": manifest.get("artifact_role") or "processing_result",
                "branch_id": branch_id,
                "parent_artifact_id": parent_id,
                "h5_path": self._artifact_relative_path(record, payload, container),
                "dataset_path": record["dataset_path"],
                "status": manifest.get("status") or "success",
                "dtype": record["dtype"],
                "shape": record["shape"],
                "sha256": record["sha256"],
                "params": record["params"],
                "manifest": manifest,
                "created_at": manifest.get("saved_at")
                or manifest.get("created_at")
                or manifest.get("committed_at"),
            }
        )

    def _cleanup_empty_dirs(self) -> None:
        try:
            self.root.rmdir()
        except OSError:
            return
        try:
            self.root.parent.rmdir()
        except OSError:
            pass


__all__ = [
    "HYBRID_ARTIFACT_TRANSACTION_SCHEMA",
    "HybridArtifactTransaction",
    "HybridArtifactTransactionJournal",
    "HybridRecoveryAction",
]
