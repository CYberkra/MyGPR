#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-destructive migration from the legacy field layout to Hybrid Store v1."""
from __future__ import annotations

import json
import shutil
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.field_project_models import FieldLineRecord, atomic_write_json, local_now
from core.processing_artifact_index import index_processing_artifacts
from core.project_storage_backend import HYBRID_STORAGE_BACKEND, HybridProjectStorageBackend

MIGRATION_SCHEMA = "mygpr.storage_migration.v1"


@dataclass(frozen=True)
class StorageMigrationResult:
    migration_id: str
    project_root: str
    source_backend: str
    target_backend: str
    line_count: int
    raw_dataset_count: int
    processing_artifact_count: int
    retained_legacy_files: bool
    report_path: str
    started_at: str
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _MigrationProgress:
    total: int
    callback: Callable[[int, int, str], None] | None = None
    completed: int = 0

    def tick(self, message: str) -> None:
        self.completed += 1
        if self.callback is not None:
            self.callback(self.completed, self.total, message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return {}


def _check_cancel(cancel_requested) -> None:
    if cancel_requested is not None and cancel_requested():
        from core.job_manager import JobCancelled
        raise JobCancelled("项目存储迁移已取消")


def _staging_backend(store, staging_root: Path) -> HybridProjectStorageBackend:
    staged_manifest = deepcopy(store.manifest)
    staged_manifest.storage_backend = HYBRID_STORAGE_BACKEND
    staged_manifest.catalog_path = "catalog.sqlite"
    staged_manifest.line_container_pattern = "data/lines/{line_id}.h5"
    staged_manifest.legacy_layout = True
    backend = HybridProjectStorageBackend(staging_root, staged_manifest, read_only=False)
    backend.ensure_structure()
    return backend


def _migrate_raw_line(store, backend, staging_root: Path, line, *, cancel_requested) -> tuple[FieldLineRecord, int]:
    migrated = FieldLineRecord.from_dict(asdict(line))
    raw_count = 0
    if line.gpr_dataset_path:
        dataset = store.load_gpr_dataset(line.line_id)
        h5_path, _digest = backend.save_raw_dataset(
            line.line_id,
            dataset,
            cancel_requested=cancel_requested,
            progress_callback=None,
        )
        migrated.gpr_dataset_path = h5_path.relative_to(staging_root).as_posix()
        migrated.raw_rows = int(dataset.sample_count)
        migrated.trace_count = int(dataset.trace_count)
        migrated.raw_size_mb = round(h5_path.stat().st_size / (1024 * 1024), 3)
        raw_count = 1
    backend.catalog.upsert_line(
        migrated,
        h5_path=(backend.line_container_relative_path(line.line_id) if migrated.gpr_dataset_path else ""),
    )
    return migrated, raw_count


def _migrate_line_artifacts(
    store,
    backend,
    line,
    records: list[Any],
    *,
    migration_id: str,
    cancel_requested,
    progress: _MigrationProgress,
) -> tuple[str, list[dict[str, Any]], int]:
    latest_pointer = ""
    pointers: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.created_at):
        _check_cancel(cancel_requested)
        source_path = store.root / record.data_path
        if not source_path.exists():
            raise FileNotFoundError(f"处理结果缺失：{record.data_path}")
        matrix = np.load(source_path, mmap_mode="r", allow_pickle=False)
        manifest = _read_json(store.root / record.manifest_path) if record.manifest_path else {}
        params_payload = _read_json(store.root / record.params_path) if record.params_path else {}
        collected_params = params_payload.get("params") if isinstance(params_payload.get("params"), dict) else {}
        branch_id = str(manifest.get("branch_id") or f"{line.line_id}:main")
        migrated = backend.save_processing_artifact(
            line_id=line.line_id,
            artifact_id=record.artifact_id,
            matrix=matrix,
            manifest={
                **manifest,
                "artifact_id": record.artifact_id,
                "line_id": line.line_id,
                "method_id": record.method_id,
                "method_name": record.method_name,
                "artifact_role": record.role,
                "params_path": record.params_path,
                "manifest_path": record.manifest_path,
                "saved_at": record.created_at or local_now(),
                "storage_mode": "hdf5_line_container",
            },
            params=collected_params,
            branch_id=branch_id,
            parent_artifact_id=str(manifest.get("parent_artifact_id") or ""),
        )
        pointer_rel = f"processed/{line.line_id}/{record.artifact_id}.artifact"
        pointers.append({
            "path": pointer_rel,
            "payload": {
                "schema": "mygpr.artifact_pointer.v1",
                "artifact_id": record.artifact_id,
                "line_id": line.line_id,
                "data_uri": migrated["data_uri"],
                "h5_path": migrated["h5_path"],
                "dataset_path": migrated["dataset_path"],
                "manifest_path": record.manifest_path,
                "params_path": record.params_path,
                "created_at": record.created_at or local_now(),
                "migrated_by": migration_id,
            },
        })
        latest_pointer = pointer_rel
        progress.tick(f"迁移处理版本 {record.artifact_id}")
    return latest_pointer, pointers, len(pointers)


def _stage_project_data(
    store,
    backend,
    staging_root: Path,
    lines: list[Any],
    artifacts_by_line: dict[str, list[Any]],
    *,
    migration_id: str,
    cancel_requested,
    progress: _MigrationProgress,
) -> tuple[list[FieldLineRecord], list[dict[str, Any]], int, int]:
    migrated_lines: list[FieldLineRecord] = []
    pointer_records: list[dict[str, Any]] = []
    raw_count = 0
    artifact_count = 0
    for line in lines:
        _check_cancel(cancel_requested)
        migrated, raw_delta = _migrate_raw_line(
            store, backend, staging_root, line, cancel_requested=cancel_requested
        )
        raw_count += raw_delta
        progress.tick(f"迁移测线 {line.line_id}")
        latest, pointers, artifact_delta = _migrate_line_artifacts(
            store,
            backend,
            line,
            artifacts_by_line.get(line.line_id, []),
            migration_id=migration_id,
            cancel_requested=cancel_requested,
            progress=progress,
        )
        if latest:
            migrated.processed_result = latest
        migrated.updated_at = local_now()
        backend.catalog.upsert_line(
            migrated,
            h5_path=(backend.line_container_relative_path(line.line_id) if migrated.gpr_dataset_path else ""),
        )
        migrated_lines.append(migrated)
        pointer_records.extend(pointers)
        artifact_count += artifact_delta
    return migrated_lines, pointer_records, raw_count, artifact_count


def _validate_staged_backend(backend, migrated_lines: list[FieldLineRecord]) -> None:
    ok, message = backend.catalog.integrity_check()
    if not ok:
        raise RuntimeError(f"迁移目录数据库完整性检查失败：{message}")
    for line in migrated_lines:
        if not line.gpr_dataset_path:
            continue
        issues = backend.validate_line(line.line_id)
        if issues:
            raise RuntimeError(f"{line.line_id} HDF5 校验失败：{'；'.join(issues)}")
    backend.catalog.checkpoint(truncate=True)


def _commit_staged_files(
    store,
    staging_root: Path,
    pointer_records: list[dict[str, Any]],
) -> list[Path]:
    committed: list[Path] = []
    final_data_root = store.root / "data" / "lines"
    final_data_root.mkdir(parents=True, exist_ok=True)
    for staged_file in sorted((staging_root / "data" / "lines").glob("*.h5")):
        destination = final_data_root / staged_file.name
        if destination.exists():
            raise FileExistsError(f"目标 HDF5 已存在：{destination}")
        staged_file.replace(destination)
        committed.append(destination)
    final_catalog = store.root / "catalog.sqlite"
    if final_catalog.exists():
        raise FileExistsError(f"目标目录数据库已存在：{final_catalog}")
    (staging_root / "catalog.sqlite").replace(final_catalog)
    committed.append(final_catalog)
    for pointer in pointer_records:
        pointer_path = store.root / str(pointer["path"])
        if pointer_path.exists():
            raise FileExistsError(f"目标处理指针已存在：{pointer_path}")
        atomic_write_json(pointer_path, dict(pointer["payload"]))
        committed.append(pointer_path)
    return committed


def _activate_hybrid_project(store, migrated_lines: list[FieldLineRecord], migration_payload: dict[str, Any]) -> None:
    store.manifest.storage_backend = HYBRID_STORAGE_BACKEND
    store.manifest.catalog_path = "catalog.sqlite"
    store.manifest.line_container_pattern = "data/lines/{line_id}.h5"
    store.manifest.legacy_layout = True
    policy = dict(getattr(store.manifest, "storage_policy", {}) or {})
    policy.update(
        single_writer=True,
        atomic_commit=True,
        bounded_memory=True,
        immutable_source_files=True,
        immutable_raw=False,
        normalized_raw_write_policy="controlled_replace_with_backup",
        legacy_files_retained=True,
    )
    store.manifest.storage_policy = policy
    store.manifest.set_lines(migrated_lines)
    store.storage = HybridProjectStorageBackend(store.root, store.manifest, read_only=False)
    store.save_manifest()
    store.storage.catalog.append_audit(
        "storage_migration_committed",
        object_type="project",
        object_id=store.manifest.project_id,
        payload=migration_payload,
    )


def _rollback_migration(store, original_manifest, original_storage, backup_manifest: Path, committed: list[Path]) -> None:
    for path in reversed(committed):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    if backup_manifest.exists():
        shutil.copy2(backup_manifest, store.root / store.MANIFEST_NAME)
    store.manifest = original_manifest
    store.storage = original_storage


def migrate_project_to_hybrid(store, *, cancel_requested=None, progress_callback=None) -> StorageMigrationResult:
    """Migrate arrays to HDF5/SQLite without deleting the legacy source files."""
    if store.read_only:
        raise PermissionError("只读项目不能迁移存储结构。")
    source_backend = str(getattr(store.manifest, "storage_backend", "") or "legacy_files_v2")
    if source_backend == HYBRID_STORAGE_BACKEND:
        raise ValueError("项目已经使用 Hybrid HDF5 + SQLite 存储。")

    started_at = local_now()
    migration_id = f"storage_v3_{uuid.uuid4().hex[:12]}"
    staging_root = store.root / "cache" / "staging" / migration_id
    backup_root = store.root / "backups" / f"before_{migration_id}"
    report_path = store.root / "metadata" / "migrations" / f"{migration_id}.json"
    staging_root.mkdir(parents=True, exist_ok=False)
    backup_root.mkdir(parents=True, exist_ok=False)
    backup_manifest = backup_root / store.MANIFEST_NAME
    shutil.copy2(store.root / store.MANIFEST_NAME, backup_manifest)

    original_manifest = deepcopy(store.manifest)
    original_storage = store.storage
    committed: list[Path] = []
    succeeded = False
    try:
        lines = store.list_lines()
        records = index_processing_artifacts(store.root)
        artifacts_by_line: dict[str, list[Any]] = {}
        for record in records:
            artifacts_by_line.setdefault(record.line_id, []).append(record)
        progress = _MigrationProgress(
            total=max(len(lines) + len(records), 1),
            callback=progress_callback,
        )
        backend = _staging_backend(store, staging_root)
        migrated_lines, pointers, raw_count, artifact_count = _stage_project_data(
            store,
            backend,
            staging_root,
            lines,
            artifacts_by_line,
            migration_id=migration_id,
            cancel_requested=cancel_requested,
            progress=progress,
        )
        _validate_staged_backend(backend, migrated_lines)
        committed.extend(_commit_staged_files(store, staging_root, pointers))
        migration_payload = {
            "migration_id": migration_id,
            "source_backend": source_backend,
            "raw_dataset_count": raw_count,
            "processing_artifact_count": artifact_count,
            "legacy_files_retained": True,
        }
        _activate_hybrid_project(store, migrated_lines, migration_payload)
        result = StorageMigrationResult(
            migration_id=migration_id,
            project_root=str(store.root),
            source_backend=source_backend,
            target_backend=HYBRID_STORAGE_BACKEND,
            line_count=len(migrated_lines),
            raw_dataset_count=raw_count,
            processing_artifact_count=artifact_count,
            retained_legacy_files=True,
            report_path=report_path.relative_to(store.root).as_posix(),
            started_at=started_at,
            finished_at=local_now(),
        )
        atomic_write_json(
            report_path,
            {
                "schema": MIGRATION_SCHEMA,
                **result.to_dict(),
                "backup_manifest": backup_manifest.relative_to(store.root).as_posix(),
                "note": "旧 raw/processed 文件未删除，可在验证通过后通过单独清理流程回收空间。",
            },
        )
        store.append_log(
            f"完成 Hybrid Store 迁移：lines={len(migrated_lines)}, "
            f"raw={raw_count}, artifacts={artifact_count}"
        )
        succeeded = True
        return result
    finally:
        if not succeeded:
            _rollback_migration(store, original_manifest, original_storage, backup_manifest, committed)
        shutil.rmtree(staging_root, ignore_errors=True)


__all__ = ["MIGRATION_SCHEMA", "StorageMigrationResult", "migrate_project_to_hybrid"]
