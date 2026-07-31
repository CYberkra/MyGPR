#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable MyGPR project storage, locking, and result persistence."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from core.array_storage import atomic_save_npy, atomic_save_npz_compressed
from core.project_models import LineRecordV1, ProcessingResultV1, ProjectManifestV1
from core.project_repository import ProjectAccessMode, ProjectRepository, ProjectSession
from core.schema_registry import DEFAULT_SCHEMA_REGISTRY
from core.storage_primitives import ProjectLockError, atomic_write_json, utc_now


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProjectService:
    MANIFEST_NAME = "project.mygpr.json"

    def __init__(
        self,
        root: Path,
        manifest: ProjectManifestV1,
        *,
        acquire_lock: bool = True,
        recover_stale_lock: bool = False,
        session: ProjectSession | None = None,
    ):
        self.root = root.resolve()
        self.manifest = manifest
        self._lock_path = self.root / ".mygpr.lock"
        self._session: ProjectSession | None = session
        self._has_lock = session is not None
        if acquire_lock and self._session is None:
            try:
                self._session = ProjectRepository.open_session(
                    self.root,
                    mode=ProjectAccessMode.WRITE,
                    recover_stale=recover_stale_lock,
                    allow_reentrant=False,
                )
            except ProjectLockError as exc:
                raise RuntimeError(f"Project is already open for writing: {self.root}") from exc
            self._has_lock = True

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        name: str,
        temporary: bool = False,
    ) -> "ProjectService":
        root_path = Path(root).resolve()
        if (root_path / cls.MANIFEST_NAME).exists():
            raise FileExistsError(f"MyGPR project already exists: {root_path}")
        root_path.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        manifest = ProjectManifestV1(
            project_id=str(uuid.uuid4()),
            name=name,
            temporary=temporary,
            created_at=now,
            updated_at=now,
        )
        service = cls(root_path, manifest)
        service.ensure_structure()
        service.save_manifest()
        return service

    @classmethod
    def open(cls, root: str | Path) -> "ProjectService":
        return cls._open_locked(root, recover_stale_lock=False)

    @classmethod
    def recover(cls, root: str | Path) -> "ProjectService":
        """Open a project after proving its existing writer lock is stale."""
        return cls._open_locked(root, recover_stale_lock=True)

    @classmethod
    def _open_locked(cls, root: str | Path, *, recover_stale_lock: bool) -> "ProjectService":
        root_path = Path(root).resolve()
        try:
            session = ProjectRepository.open_session(
                root_path,
                mode=ProjectAccessMode.WRITE,
                recover_stale=recover_stale_lock,
                allow_reentrant=False,
            )
        except ProjectLockError as exc:
            raise RuntimeError(f"Project is already open for writing: {root_path}") from exc
        try:
            payload = cls._read_manifest(root_path)
            return cls(
                root_path,
                ProjectManifestV1.from_dict(payload),
                acquire_lock=False,
                session=session,
            )
        except Exception:
            session.close()
            raise

    @classmethod
    def _read_manifest(cls, root_path: Path) -> dict[str, Any]:
        loaded = DEFAULT_SCHEMA_REGISTRY.load_path(
            root_path / cls.MANIFEST_NAME,
            family="mygpr.project",
            write_migrated=True,
            quarantine_root=root_path / "workspace" / "quarantine",
        )
        if loaded.read_only:
            raise PermissionError(f"Project schema is newer than this MyGPR build: {loaded.source_schema}")
        return loaded.payload

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._has_lock = False

    def _write_json(self, path: Path, payload: Any, *, label: str) -> None:
        if self._session is None:
            atomic_write_json(path, payload)
            return
        with self._session.transaction(label) as transaction:
            transaction.track(path)
            atomic_write_json(path, payload)

    def ensure_structure(self) -> None:
        for name in ("raw", "lines", "qc", "results", "interpretations", "workspace", "exports"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def save_manifest(self) -> None:
        self.manifest.updated_at = utc_now()
        self._write_json(
            self.root / self.MANIFEST_NAME,
            self.manifest.to_dict(),
            label="save-project-manifest",
        )

    def resolve_relative_path(self, relative: str | Path) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Project path escapes root: {relative}") from exc
        return candidate

    def add_line(self, line: LineRecordV1) -> None:
        line.updated_at = utc_now()
        if not line.created_at:
            line.created_at = line.updated_at
        line_path = self.resolve_relative_path(f"lines/{line.line_id}/line.json")
        manifest_path = self.root / self.MANIFEST_NAME
        is_new = line.line_id not in self.manifest.line_ids
        if is_new:
            self.manifest.line_ids.append(line.line_id)
            self.manifest.updated_at = utc_now()
        if self._session is None:
            atomic_write_json(line_path, line.to_dict())
            if is_new:
                atomic_write_json(manifest_path, self.manifest.to_dict())
            return
        with self._session.transaction(f"save-line-{line.line_id}") as transaction:
            transaction.track(line_path)
            atomic_write_json(line_path, line.to_dict())
            if is_new:
                transaction.track(manifest_path)
                atomic_write_json(manifest_path, self.manifest.to_dict())

    def get_line(self, line_id: str) -> LineRecordV1:
        path = self.resolve_relative_path(f"lines/{line_id}/line.json")
        return LineRecordV1.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_lines(self) -> list[LineRecordV1]:
        return [self.get_line(line_id) for line_id in self.manifest.line_ids]

    def save_processing_result(
        self,
        line_id: str,
        data: np.ndarray,
        *,
        name: str,
        processing_chain: list[dict[str, Any]],
        header_info: dict[str, Any] | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> ProcessingResultV1:
        result_id = f"R-{uuid.uuid4().hex[:12]}"
        base = self.resolve_relative_path(f"results/{line_id}/{result_id}")
        base.mkdir(parents=True, exist_ok=True)
        data_path = base / "data.npy"
        metadata_file = base / "trace_metadata.npz"
        trace_metadata_path: str | None = None
        if trace_metadata:
            trace_metadata_path = metadata_file.relative_to(self.root).as_posix()
        result = ProcessingResultV1(
            result_id=result_id,
            line_id=line_id,
            name=name,
            data_path=data_path.relative_to(self.root).as_posix(),
            processing_chain=_json_safe(processing_chain),
            trace_metadata_path=trace_metadata_path,
            header_info=_json_safe(header_info or {}),
            created_at=utc_now(),
        )
        result_path = base / "result.json"
        if self._session is None:
            atomic_save_npy(data_path, data)
            if trace_metadata:
                atomic_save_npz_compressed(metadata_file, trace_metadata)
            atomic_write_json(result_path, result.to_dict())
            return result
        with self._session.transaction(f"save-result-{result_id}") as transaction:
            for path in (data_path, metadata_file if trace_metadata else None, result_path):
                if path is not None:
                    transaction.track(path)
            atomic_save_npy(data_path, data)
            if trace_metadata:
                atomic_save_npz_compressed(metadata_file, trace_metadata)
            atomic_write_json(result_path, result.to_dict())
        return result

    def load_processing_result(
        self,
        result_id: str,
        *,
        line_id: str | None = None,
    ) -> dict[str, Any]:
        """Load one versioned processing result and its optional metadata."""
        if line_id is None:
            matches = list((self.root / "results").glob(f"*/{result_id}/result.json"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Processing result not found or ambiguous: {result_id}")
            record_path = matches[0]
        else:
            record_path = self.resolve_relative_path(
                f"results/{line_id}/{result_id}/result.json"
            )
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        record = ProcessingResultV1.from_dict(payload)
        data = np.load(self.resolve_relative_path(record.data_path), allow_pickle=False)
        trace_metadata: dict[str, np.ndarray] = {}
        if record.trace_metadata_path:
            with np.load(
                self.resolve_relative_path(record.trace_metadata_path),
                allow_pickle=False,
            ) as archive:
                trace_metadata = {
                    str(key): np.array(archive[key], copy=True) for key in archive.files
                }
        return {
            "record": record,
            "data": np.array(data, copy=True),
            "header_info": dict(record.header_info),
            "trace_metadata": trace_metadata,
        }

    def list_processing_results(
        self,
        line_id: str | None = None,
    ) -> list[ProcessingResultV1]:
        pattern = f"{line_id}/*/result.json" if line_id else "*/*/result.json"
        records: list[ProcessingResultV1] = []
        for path in sorted((self.root / "results").glob(pattern)):
            records.append(
                ProcessingResultV1.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            )
        return records

    def __enter__(self) -> "ProjectService":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["ProjectService", "atomic_write_json", "sha256_path", "utc_now"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else np.array([])
        return {
            "kind": "ndarray_summary",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
