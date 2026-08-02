#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable filesystem primitives shared by every MyGPR project store.

The helpers in this module are deliberately Qt-free.  They provide the one
write path used by project manifests, registries, job journals and report
seals: write a sibling temporary file, flush it, fsync it, atomically replace
the destination and fsync the parent directory when the platform supports it.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import threading
import uuid
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python <3.11 fallback
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after an atomic rename.

    Windows does not expose portable directory fsync through Python.  On POSIX
    this closes the final durability gap between ``os.replace`` and an abrupt
    power loss.
    """
    if os.name == "nt":
        return
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_file(path: str | Path) -> None:
    """Flush one existing file to stable storage where the OS permits it."""
    source = Path(path)
    try:
        with source.open("rb") as stream:
            os.fsync(stream.fileno())
    except OSError:
        # Some virtual/network filesystems do not expose a durable fsync.
        # Callers still retain atomic rename and higher-level recovery logs.
        return


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
        fsync_directory(target.parent)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, str(text).encode(encoding))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False))


@contextmanager
def atomic_output_path(path: str | Path, *, suffix: str = ".tmp"):
    """Yield a sibling temporary path and atomically publish it on success.

    The caller may use any library-specific writer against the yielded path.
    On normal exit the file is flushed by the caller, atomically replaces the
    destination, and the parent directory is fsynced where supported.  On
    failure the temporary file is removed and the last valid destination is
    preserved.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}{suffix}")
    published = False
    try:
        yield temporary
        if not temporary.is_file():
            raise FileNotFoundError(f"Atomic output writer did not create temporary file: {temporary}")
        fsync_file(temporary)
        os.replace(temporary, target)
        fsync_directory(target.parent)
        published = True
    finally:
        if not published:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class ProjectAccessMode(StrEnum):
    AUTO = "auto"
    WRITE = "write"
    READ_ONLY = "read_only"


class ProjectLockError(RuntimeError):
    pass


class ProjectReadOnlyError(PermissionError):
    pass


_PROCESS_LOCKS: dict[Path, tuple[str, int]] = {}
_PROCESS_LOCK_GUARD = threading.RLock()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _boot_id() -> str:
    """Return a stable boot identifier when available (Linux)."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return ""


def _process_start_marker(pid: int) -> str:
    """Return a marker that distinguishes PID reuse on Linux."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        # The second field can contain spaces inside parentheses.  Field 22 is
        # therefore indexed from the tail after the final ')'.
        tail = raw.rsplit(")", 1)[1].strip().split()
        return tail[19] if len(tail) > 19 else ""
    except (OSError, UnicodeError, ValueError, IndexError):
        return ""


@dataclass
class ProjectLock(AbstractContextManager["ProjectLock"]):
    root: Path
    mode: ProjectAccessMode = ProjectAccessMode.AUTO
    recover_stale: bool = False
    lock_name: str = ".mygpr.lock"
    allow_reentrant: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.lock_path = self.root / self.lock_name
        self.token = uuid.uuid4().hex
        self.read_only = self.mode == ProjectAccessMode.READ_ONLY
        self._held = False
        self._reentrant = False

    @property
    def writable(self) -> bool:
        return self._held and not self.read_only

    @property
    def reentrant(self) -> bool:
        return bool(self._reentrant)

    def acquire(self) -> "ProjectLock":
        if self.read_only:
            if not self.root.is_dir():
                raise FileNotFoundError(self.root)
            self._held = True
            return self
        self.root.mkdir(parents=True, exist_ok=True)
        with _PROCESS_LOCK_GUARD:
            existing = _PROCESS_LOCKS.get(self.root)
            if existing is not None:
                if not self.allow_reentrant:
                    if self.mode == ProjectAccessMode.AUTO:
                        self.read_only = True
                        self._held = True
                        return self
                    raise ProjectLockError(f"项目已有写入实例：{self.root}")
                token, count = existing
                _PROCESS_LOCKS[self.root] = (token, count + 1)
                self.token = token
                self._held = True
                self._reentrant = True
                return self
        payload = {
            "schema": "mygpr.project_lock.v2",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "boot_id": _boot_id(),
            "process_start": _process_start_marker(os.getpid()),
            "token": self.token,
            "created_at": utc_now(),
        }
        while True:
            try:
                fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                stale = self._is_stale()
                if stale and self.recover_stale:
                    self.lock_path.unlink(missing_ok=True)
                    continue
                if self.mode == ProjectAccessMode.AUTO:
                    self.read_only = True
                    self._held = True
                    return self
                raise ProjectLockError(f"项目已有写入实例：{self.root}")
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                fsync_directory(self.root)
                with _PROCESS_LOCK_GUARD:
                    _PROCESS_LOCKS[self.root] = (self.token, 1)
                self._held = True
                return self

    def _is_stale(self) -> bool:
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return True
        lock_host = str(payload.get("host") or "")
        # v0.9.27 and earlier lock files had no host field.  Treat those as
        # local locks so stale recovery remains backwards compatible.
        if lock_host and lock_host != socket.gethostname():
            return False
        try:
            pid = int(payload.get("pid") or 0)
        except (TypeError, ValueError):
            return True
        if not _pid_alive(pid):
            return True
        stored_boot = str(payload.get("boot_id") or "")
        current_boot = _boot_id()
        if stored_boot and current_boot and stored_boot != current_boot:
            return True
        stored_start = str(payload.get("process_start") or "")
        current_start = _process_start_marker(pid)
        if stored_start and current_start and stored_start != current_start:
            return True
        return False

    def assert_writable(self) -> None:
        if not self.writable:
            raise ProjectReadOnlyError(f"项目以只读方式打开，不能写入：{self.root}")

    def release(self) -> None:
        if not self._held:
            return
        if self.read_only:
            self._held = False
            return
        remove_file = False
        with _PROCESS_LOCK_GUARD:
            current = _PROCESS_LOCKS.get(self.root)
            if current is not None and current[0] == self.token:
                if current[1] > 1:
                    _PROCESS_LOCKS[self.root] = (current[0], current[1] - 1)
                else:
                    _PROCESS_LOCKS.pop(self.root, None)
                    remove_file = True
        if remove_file:
            try:
                payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
                if payload.get("token") == self.token:
                    self.lock_path.unlink(missing_ok=True)
                    fsync_directory(self.root)
            except (OSError, ValueError, TypeError):
                pass
        self._held = False

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@dataclass
class FileTransaction(AbstractContextManager["FileTransaction"]):
    """Small rollback-capable transaction for a finite set of project files."""

    root: Path
    label: str = "project-write"

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.transaction_id = uuid.uuid4().hex
        self.stage_root = self.root / ".transactions" / self.transaction_id
        self.backups = self.stage_root / "before"
        self.journal_path = self.stage_root / "transaction.json"
        self._tracked: dict[Path, bool] = {}
        self._committed = False

    def __enter__(self) -> "FileTransaction":
        self.backups.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.journal_path, {
            "schema": "mygpr.file_transaction.v1",
            "transaction_id": self.transaction_id,
            "label": self.label,
            "state": "active",
            "created_at": utc_now(),
            "files": [],
        })
        return self

    def track(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        try:
            relative = target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("事务只能跟踪项目目录内文件") from exc
        if target in self._tracked:
            return target
        existed = target.exists()
        self._tracked[target] = existed
        if existed and target.is_file():
            backup = self.backups / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(backup, target.read_bytes())
        self._write_journal("active")
        return target

    def _write_journal(self, state: str) -> None:
        atomic_write_json(self.journal_path, {
            "schema": "mygpr.file_transaction.v1",
            "transaction_id": self.transaction_id,
            "label": self.label,
            "state": state,
            "updated_at": utc_now(),
            "files": [
                {"path": str(path.relative_to(self.root)), "existed": existed}
                for path, existed in self._tracked.items()
            ],
        })

    def commit(self) -> None:
        self._committed = True
        self._write_journal("committed")

    def rollback(self) -> None:
        for target, existed in reversed(list(self._tracked.items())):
            relative = target.relative_to(self.root)
            backup = self.backups / relative
            if existed and backup.exists():
                atomic_write_bytes(target, backup.read_bytes())
            elif not existed:
                target.unlink(missing_ok=True)
        self._write_journal("rolled_back")

    def _cleanup(self) -> None:
        shutil.rmtree(self.stage_root, ignore_errors=True)
        try:
            self.stage_root.parent.rmdir()
        except OSError:
            pass

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None or not self._committed:
            # If rollback raises, cleanup is intentionally skipped so the
            # journal and backups remain available for diagnosis/recovery.
            self.rollback()
        self._cleanup()


def recover_file_transactions(root: str | Path) -> tuple[dict[str, Any], ...]:
    """Rollback or clean file transactions left by an interrupted process.

    Active journals are rolled back from their ``before`` snapshots.  Journals
    already marked committed/rolled_back are cleanup-only.  Malformed journals
    are retained for diagnosis and returned as failed recovery records.
    """
    project_root = Path(root).resolve()
    transactions_root = project_root / ".transactions"
    if not transactions_root.exists():
        return ()
    results: list[dict[str, Any]] = []
    for stage_root in sorted(path for path in transactions_root.iterdir() if path.is_dir()):
        if stage_root.name == "hybrid_artifacts":
            continue
        journal_path = stage_root / "transaction.json"
        transaction_id = stage_root.name
        try:
            payload = json.loads(journal_path.read_text(encoding="utf-8"))
            if payload.get("schema") != "mygpr.file_transaction.v1":
                raise ValueError("unsupported file transaction schema")
            transaction_id = str(payload.get("transaction_id") or transaction_id)
            state = str(payload.get("state") or "active")
            if state == "active":
                files = payload.get("files")
                if not isinstance(files, list):
                    raise ValueError("transaction files must be a list")
                for item in reversed(files):
                    relative = Path(str(item.get("path") or ""))
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"unsafe transaction path: {relative}")
                    target = (project_root / relative).resolve()
                    target.relative_to(project_root)
                    existed = bool(item.get("existed"))
                    backup = (stage_root / "before" / relative).resolve()
                    backup.relative_to(stage_root.resolve())
                    if existed:
                        if not backup.is_file():
                            raise FileNotFoundError(f"transaction backup missing: {backup}")
                        atomic_write_bytes(target, backup.read_bytes())
                    else:
                        target.unlink(missing_ok=True)
                        fsync_directory(target.parent)
                action = "rolled_back"
            else:
                action = "cleaned"
            shutil.rmtree(stage_root)
            fsync_directory(transactions_root)
            results.append({
                "transaction_id": transaction_id,
                "success": True,
                "action": action,
                "state": state,
            })
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            results.append({
                "transaction_id": transaction_id,
                "success": False,
                "action": "recovery_failed",
                "message": str(exc),
            })
    try:
        transactions_root.rmdir()
    except OSError:
        pass
    return tuple(results)


__all__ = [
    "FileTransaction",
    "ProjectAccessMode",
    "ProjectLock",
    "ProjectLockError",
    "ProjectReadOnlyError",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "fsync_directory",
    "fsync_file",
    "recover_file_transactions",
    "utc_now",
]
