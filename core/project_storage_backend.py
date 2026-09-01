#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage backend abstraction for field projects."""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from core.gpr_data_model import GPRDataSet, time_to_depth_axis
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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return {}


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    """Legacy npy/sidecar 存储后端。

    无 SQLite catalog；成果以 ``processed/{line_id}/{line_id}_processed_*.npy``
    + 旁车 JSON（manifest/params）落盘，读取经 ``index_processing_artifacts``
    的 npy 回退扫描。这里补齐 adapter 无条件调用的读取接口，使旧项目
    成果预览/二次处理不再 AttributeError（P0-3）。
    """

    @property
    def is_hybrid(self) -> bool:
        return False

    def ensure_structure(self, *, recover_transactions: bool = True) -> None:
        return None

    def load_processing_artifact(
        self,
        line_id: str,
        artifact_id: str,
        *,
        raw_dataset: GPRDataSet | None = None,
    ) -> GPRDataSet:
        """从 legacy npy + 旁车 JSON 重建处理成果数据集。

        旁车命名约定与 ``core/processing_artifact_index.py`` 的 npy 回退扫描一致：
        ``{line_id}_processing_manifest_{timestamp}.json`` / ``{line_id}_params_{timestamp}.json``，
        缺失时回退到无时间戳的 legacy 文件。
        """
        safe_line = validate_line_id(line_id)
        processed_root = self.root / "processed" / safe_line
        data_path = processed_root / f"{artifact_id}.npy"
        if not data_path.exists():
            raise FileNotFoundError(
                f"legacy 处理成果不存在: {artifact_id!r} for line {safe_line!r}"
            )
        timestamp = artifact_id
        prefix = f"{safe_line}_processed_"
        if timestamp.startswith(prefix):
            timestamp = timestamp[len(prefix):]
        manifest_path = processed_root / f"{safe_line}_processing_manifest_{timestamp}.json"
        if not manifest_path.exists():
            manifest_path = processed_root / f"{safe_line}_processing_manifest.json"
        params_path = processed_root / f"{safe_line}_params_{timestamp}.json"
        if not params_path.exists():
            params_path = processed_root / f"{safe_line}_params.json"
        manifest = _load_json(manifest_path) if manifest_path.exists() else {}
        params = _load_json(params_path) if params_path.exists() else {}
        input_dataset = (
            params.get("input_dataset")
            if isinstance(params.get("input_dataset"), dict) else {}
        )
        # legacy 回退路径同样不得整文件读入内存；float32 npy 以 memmap
        # 透传给 GPRDataSet.from_matrix（copy=False 不会物化）。
        matrix = np.load(data_path, mmap_mode="r", allow_pickle=False)
        # P1-1：优先用持久化的输出 header 重建物理轴（time_cut/set_zero_time 改变时窗/零点）
        output_header = (
            manifest.get("output_header")
            if isinstance(manifest.get("output_header"), dict) else {}
        )
        out_time_window = _coerce_float(
            output_header.get("total_time_ns") or output_header.get("time_window_ns")
        )
        out_offset = _coerce_float(output_header.get("time_cut_offset_ns")) or 0.0
        if out_time_window and out_time_window > 0:
            time_window = out_time_window
        else:
            time_window = _coerce_float(
                input_dataset.get("time_window_ns")
                or manifest.get("time_window_ns") or 250.0
            ) or 250.0
        sample_count = int(matrix.shape[0])
        time_axis = None
        if output_header and sample_count > 0:
            time_axis = out_offset + np.linspace(0.0, time_window, sample_count, dtype=np.float32)
        metadata = {
            **manifest,
            "artifact_id": artifact_id,
            "input_dataset": input_dataset,
        }
        dataset = GPRDataSet.from_matrix(
            line_id=safe_line,
            matrix=matrix,
            length_m=_coerce_float(input_dataset.get("length_m")),
            time_window_ns=time_window,
            dielectric_constant=_coerce_float(
                input_dataset.get("dielectric_constant")
                or manifest.get("dielectric_constant") or 9.0
            ) or 9.0,
            source_path=str(input_dataset.get("source_path") or ""),
            format_name=str(input_dataset.get("format_name") or "memory"),
            metadata=metadata,
        )
        if time_axis is not None and len(time_axis) == dataset.sample_count:
            dataset.time_axis_ns = time_axis
            dataset.depth_axis_m = time_to_depth_axis(time_axis, dataset.dielectric_constant)
        return dataset


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
            "artifact_kind": str(
                (params or {}).get("artifact_kind") or "processing"
            ),
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
