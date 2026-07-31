#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage backend abstraction for field projects."""
from __future__ import annotations

from abc import ABC, abstractmethod
import sqlite3
from pathlib import Path
from typing import Any

from core.gpr_data_model import GPRDataSet
from core.field_project_models import validate_line_id
from core.hybrid_transaction_journal import HybridArtifactTransactionJournal
from core.hdf5_line_container import (
    initialize_line_container,
    delete_processing_artifact,
    load_processing_dataset,
    load_raw_dataset,
    validate_line_container,
    write_processing_artifact,
    write_raw_dataset,
)
from core.project_catalog import ProjectCatalog
from core.storage_uri import make_h5_uri

HYBRID_STORAGE_BACKEND = "hybrid_hdf5_sqlite_v1"
LEGACY_STORAGE_BACKEND = "legacy_files_v2"


class ProjectStorageBackend(ABC):
    def __init__(self, root: str | Path, manifest: Any, *, read_only: bool = False) -> None:
        self.root = Path(root).resolve()
        self.manifest = manifest
        self.read_only = bool(read_only)

    @abstractmethod
    def ensure_structure(self, *, recover_transactions: bool = True) -> None: ...

    @property
    @abstractmethod
    def is_hybrid(self) -> bool: ...


class LegacyProjectStorageBackend(ProjectStorageBackend):
    @property
    def is_hybrid(self) -> bool:
        return False

    def ensure_structure(self, *, recover_transactions: bool = True) -> None:
        return None


class HybridProjectStorageBackend(ProjectStorageBackend):
    def __init__(self, root: str | Path, manifest: Any, *, read_only: bool = False) -> None:
        super().__init__(root, manifest, read_only=read_only)
        catalog_rel = str(getattr(manifest, "catalog_path", "") or "catalog.sqlite")
        self.catalog_path = self.root / catalog_rel
        self.catalog = ProjectCatalog(self.catalog_path, read_only=read_only)
        self.transaction_journal = HybridArtifactTransactionJournal(self.root)
        self.last_recovery_actions = ()

    @property
    def is_hybrid(self) -> bool:
        return True

    def ensure_structure(self, *, recover_transactions: bool = True) -> None:
        for relative in (
            "data/lines", "attachments", "exports", "cache/previews", "cache/staging",
            "logs", "backups", "metadata",
        ):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        self.catalog.initialize(project_id=self.manifest.project_id, project_name=self.manifest.name)
        if not self.read_only and recover_transactions:
            self.recover_pending_transactions()

    def recover_pending_transactions(self):
        """Reconcile interrupted SQLite/HDF5 artifact commits."""
        if self.read_only:
            return ()
        actions = self.transaction_journal.recover(self.catalog, self.line_container_path)
        self.last_recovery_actions = actions
        failures = [action for action in actions if not action.success]
        if failures:
            details = "; ".join(action.message or action.transaction_id for action in failures)
            raise RuntimeError(f"混合存储事务恢复失败：{details}")
        return actions

    def line_container_path(self, line_id: str) -> Path:
        safe_line_id = validate_line_id(line_id)
        candidate = (self.root / "data" / "lines" / f"{safe_line_id}.h5").resolve()
        candidate.relative_to(self.root)
        return candidate

    def line_container_relative_path(self, line_id: str) -> str:
        return self.line_container_path(line_id).relative_to(self.root).as_posix()

    def save_raw_dataset(self, line_id: str, dataset: GPRDataSet, *, cancel_requested=None, progress_callback=None) -> tuple[Path, str]:
        result = write_raw_dataset(
            self.line_container_path(line_id), dataset,
            project_id=self.manifest.project_id, line_id=line_id,
            cancel_requested=cancel_requested, progress_callback=progress_callback,
        )
        return result

    def load_raw_dataset(self, line_id: str) -> GPRDataSet:
        return load_raw_dataset(self.line_container_path(line_id), line_id=line_id)

    def save_processing_artifact(
        self,
        *,
        line_id: str,
        artifact_id: str,
        matrix: Any,
        manifest: dict[str, Any],
        params: dict[str, Any],
        branch_id: str,
        parent_artifact_id: str = "",
        cancel_requested=None,
        progress_callback=None,
    ) -> dict[str, Any]:
        container = self.line_container_path(line_id)
        if not container.exists():
            initialize_line_container(container, project_id=self.manifest.project_id, line_id=line_id)
        self.catalog.ensure_branch(line_id=line_id, branch_id=branch_id, name=branch_id)
        resolved_parent = parent_artifact_id or self.catalog.branch_head(branch_id)
        manifest = {**manifest, "branch_id": branch_id, "parent_artifact_id": resolved_parent}
        relative = container.relative_to(self.root).as_posix()
        transaction = self.transaction_journal.begin(
            line_id=line_id,
            artifact_id=artifact_id,
            branch_id=branch_id,
            parent_artifact_id=resolved_parent,
            h5_path=relative,
            manifest=manifest,
            params=params,
        )
        result: dict[str, Any] | None = None
        try:
            result = write_processing_artifact(
                container,
                artifact_id=artifact_id,
                matrix=matrix,
                manifest=manifest,
                params=params,
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
            )
            transaction.update(
                "hdf5_committed",
                dataset_path=result["dataset_path"],
                dtype=result["dtype"],
                shape=result["shape"],
                sha256=result["sha256"],
                manifest=result["manifest"],
            )
            self.catalog.register_artifact(
                self._artifact_catalog_payload(
                    line_id=line_id,
                    artifact_id=artifact_id,
                    branch_id=branch_id,
                    parent_artifact_id=resolved_parent,
                    h5_path=relative,
                    params=params,
                    result=result,
                )
            )
        except (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError, KeyError):
            if self._rollback_failed_artifact_commit(container, artifact_id, result):
                self._complete_transaction(transaction, "rolled_back")
            raise
        self._complete_transaction(transaction, "catalog_committed")
        assert result is not None
        return {
            **result,
            "h5_path": relative,
            "data_uri": make_h5_uri(relative, result["dataset_path"]),
            "parent_artifact_id": resolved_parent,
            "branch_id": branch_id,
        }

    @staticmethod
    def _artifact_catalog_payload(
        *,
        line_id: str,
        artifact_id: str,
        branch_id: str,
        parent_artifact_id: str,
        h5_path: str,
        params: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        manifest = result["manifest"]
        return {
            "artifact_id": artifact_id,
            "line_id": line_id,
            "artifact_kind": "processing",
            "artifact_role": manifest.get("artifact_role") or "processing_result",
            "branch_id": branch_id,
            "parent_artifact_id": parent_artifact_id,
            "h5_path": h5_path,
            "dataset_path": result["dataset_path"],
            "status": manifest.get("status") or "success",
            "dtype": result["dtype"],
            "shape": result["shape"],
            "sha256": result["sha256"],
            "params": params,
            "manifest": manifest,
            "created_at": manifest.get("saved_at") or manifest.get("created_at"),
        }

    def _rollback_failed_artifact_commit(
        self,
        container: Path,
        artifact_id: str,
        result: dict[str, Any] | None,
    ) -> bool:
        rollback_ok = True
        try:
            self.catalog.delete_artifact(artifact_id)
        except (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError, KeyError):
            rollback_ok = False
        try:
            if result is not None:
                delete_processing_artifact(container, artifact_id)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            rollback_ok = False
        return rollback_ok

    @staticmethod
    def _complete_transaction(transaction: Any, state: str) -> None:
        # The catalog and HDF5 commit are authoritative once both are durable.
        # Journal cleanup failure is reconciled on the next writable open.
        try:
            transaction.update(state)
            transaction.complete()
        except OSError:
            pass

    def rollback_processing_artifact(self, line_id: str, artifact_id: str) -> None:
        """Rollback an artifact whose final catalog/sidecar commit failed."""
        self.catalog.delete_artifact(artifact_id)
        delete_processing_artifact(self.line_container_path(line_id), artifact_id)

    def load_processing_artifact(self, line_id: str, artifact_id: str, *, raw_dataset: GPRDataSet | None = None) -> GPRDataSet:
        return load_processing_dataset(self.line_container_path(line_id), artifact_id=artifact_id, raw_dataset=raw_dataset)

    def validate_line(self, line_id: str) -> list[str]:
        return validate_line_container(
            self.line_container_path(line_id),
            project_id=self.manifest.project_id,
            line_id=line_id,
        )


def create_storage_backend(root: str | Path, manifest: Any, *, read_only: bool = False) -> ProjectStorageBackend:
    backend = str(getattr(manifest, "storage_backend", "") or LEGACY_STORAGE_BACKEND)
    if backend == HYBRID_STORAGE_BACKEND:
        return HybridProjectStorageBackend(root, manifest, read_only=read_only)
    return LegacyProjectStorageBackend(root, manifest, read_only=read_only)


__all__ = [
    "HYBRID_STORAGE_BACKEND",
    "LEGACY_STORAGE_BACKEND",
    "HybridProjectStorageBackend",
    "LegacyProjectStorageBackend",
    "ProjectStorageBackend",
    "create_storage_backend",
]
