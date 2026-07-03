#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable MyGPR project storage, locking, and result persistence."""

from __future__ import annotations

import hashlib
import errno
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core.project_models import LineRecordV1, ProcessingResultV1, ProjectManifestV1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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
    ):
        self.root = root.resolve()
        self.manifest = manifest
        self._lock_path = self.root / ".mygpr.lock"
        self._has_lock = False
        if acquire_lock:
            self._acquire_lock(recover_stale=recover_stale_lock)

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
        root_path = Path(root).resolve()
        payload = cls._read_manifest(root_path)
        return cls(root_path, ProjectManifestV1.from_dict(payload))

    @classmethod
    def recover(cls, root: str | Path) -> "ProjectService":
        """Open a project after proving its existing writer lock is stale."""
        root_path = Path(root).resolve()
        payload = cls._read_manifest(root_path)
        return cls(
            root_path,
            ProjectManifestV1.from_dict(payload),
            recover_stale_lock=True,
        )

    @classmethod
    def _read_manifest(cls, root_path: Path) -> dict[str, Any]:
        payload = json.loads((root_path / cls.MANIFEST_NAME).read_text(encoding="utf-8"))
        if payload.get("schema") != "mygpr.project.v1":
            raise ValueError(f"Unsupported MyGPR project schema: {payload.get('schema')!r}")
        return payload

    def _acquire_lock(self, *, recover_stale: bool = False) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            if not recover_stale or not self._remove_stale_lock():
                raise RuntimeError(f"Project is already open for writing: {self.root}") from exc
            descriptor = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "opened_at": utc_now()}, stream)
        self._has_lock = True

    def _remove_stale_lock(self) -> bool:
        try:
            payload = json.loads(self._lock_path.read_text(encoding="utf-8"))
            pid = int(payload["pid"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False
        if self._process_is_running(pid):
            return False
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            return True
        return True

    @staticmethod
    def _process_is_running(pid: int) -> bool:
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True
        if os.name == "nt":
            try:
                import ctypes

                process_query_limited_information = 0x1000
                still_active = 259
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(
                    process_query_limited_information,
                    False,
                    pid,
                )
                if not handle:
                    return kernel32.GetLastError() == 5
                try:
                    exit_code = ctypes.c_ulong()
                    if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                        return True
                    return exit_code.value == still_active
                finally:
                    kernel32.CloseHandle(handle)
            except Exception:
                return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:
            if exc.errno == errno.ESRCH or getattr(exc, "winerror", None) == 87:
                return False
            return True
        return True

    def close(self) -> None:
        if self._has_lock:
            self._lock_path.unlink(missing_ok=True)
            self._has_lock = False

    def ensure_structure(self) -> None:
        for name in ("raw", "lines", "qc", "results", "interpretations", "workspace", "exports"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def save_manifest(self) -> None:
        self.manifest.updated_at = utc_now()
        atomic_write_json(self.root / self.MANIFEST_NAME, self.manifest.to_dict())

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
        atomic_write_json(self.resolve_relative_path(f"lines/{line.line_id}/line.json"), line.to_dict())
        if line.line_id not in self.manifest.line_ids:
            self.manifest.line_ids.append(line.line_id)
            self.save_manifest()

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
        np.save(data_path, np.asarray(data))
        trace_metadata_path: str | None = None
        if trace_metadata:
            metadata_file = base / "trace_metadata.npz"
            np.savez_compressed(
                metadata_file,
                **{str(key): np.asarray(value) for key, value in trace_metadata.items()},
            )
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
        atomic_write_json(base / "result.json", result.to_dict())
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
