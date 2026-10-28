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
import logging
import os
import shutil
import socket
import sqlite3
import sys
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
    """Flush one existing file to stable storage where the OS permits it.

    Windows ``FlushFileBuffers`` requires a handle with write access, so the
    file is opened ``rb+``; a read-only handle would silently fail fsync on
    Windows.  Read-only files keep the previous best-effort behaviour.

    本函数面向"按路径 fsync"或句柄可能只读的场景。对已经以写模式打开的
    句柄（``"wb"``/``"a"``/``mkstemp``），直接 ``os.fsync(stream.fileno())``
    即可，不必经此函数重开文件；错误语义（是否吞 OSError）由调用方决定。
    """
    source = Path(path)
    try:
        with source.open("rb+") as stream:
            os.fsync(stream.fileno())
    except OSError:
        # Read-only files, and some virtual/network filesystems, do not
        # expose a durable fsync.  Callers still retain atomic rename and
        # higher-level recovery logs.
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


def _atomic_copy_file(source: Path, target: Path) -> None:
    """Copy transaction snapshots with bounded memory, including large HDF5."""
    with atomic_output_path(target) as temporary:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _same_file_contents(first: Path, second: Path) -> bool:
    if not second.is_file() or first.stat().st_size != second.stat().st_size:
        return False
    with first.open('rb') as left, second.open('rb') as right:
        while True:
            block = left.read(1024 * 1024)
            if block != right.read(1024 * 1024):
                return False
            if not block:
                return True


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


_LOGGER = logging.getLogger(__name__)


class ProjectReadOnlyError(PermissionError):
    pass


_PROCESS_LOCKS: dict[Path, tuple[str, int]] = {}
_PROCESS_LOCK_GUARD = threading.RLock()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        # os.kill(pid, 0) 在 Windows 上对不存在的 PID 会抛
        # OSError(WinError 6 句柄无效)，且可能以 SystemError 形态逃逸
        # （"<class 'OSError'> returned a result with an exception set"，
        # 用户可见为新建/打开项目报错）。改用 Win32 OpenProcess 判活。
        # use_last_error=True 才能可靠读到 GetLastError 错误码。
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_ACCESS_DENIED = 5
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            # 不存在的 PID → ERROR_INVALID_PARAMETER/ERROR_INVALID_SID 类；
            # 存在但拒访（PPL/服务/DACL）→ ERROR_ACCESS_DENIED。
            # 拒访必须视为存活：宁可让用户手删陈旧锁，也不能把活锁
            # 恢复掉造成双实例写同一项目（数据安全红线）。
            if ctypes.get_last_error() == ERROR_ACCESS_DENIED:
                return True
            return False
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
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
    """Return a marker that distinguishes PID reuse across platforms.

    Linux reads the kernel ``starttime`` field from ``/proc/<pid>/stat``;
    Windows derives the process creation FILETIME via ``GetProcessTimes``.
    Returns "" when the marker is unavailable (the stale check then simply
    skips the reuse comparison instead of mis-detecting a live lock).
    """
    pid = int(pid)
    if os.name == "nt":
        return _windows_process_start_marker(pid)
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # The second field can contain spaces inside parentheses.  Field 22 is
        # therefore indexed from the tail after the final ')'.
        tail = raw.rsplit(")", 1)[1].strip().split()
        return tail[19] if len(tail) > 19 else ""
    except (OSError, UnicodeError, ValueError, IndexError):
        return ""


def _windows_process_start_marker(pid: int) -> str:
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return ""
        return f"{creation.dwHighDateTime:08x}{creation.dwLowDateTime:08x}"
    finally:
        kernel32.CloseHandle(handle)


@contextmanager
def _project_lock_file_guard(path: Path):
    """Serialize lock-file creation/recovery across processes.

    The stable guard file must not be removed: replacing its inode would let
    contenders lock different files. The OS releases its byte lock on crash.
    """
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


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
            with _project_lock_file_guard(self.root / f"{self.lock_name}.guard"):
                return self._acquire_lock_file()

    def _acquire_lock_file(self) -> "ProjectLock":
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
                    # 陈旧锁的持有者 PID（rename 抢占后用于复核归属）
                    try:
                        stale_payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
                        stale_pid = int(stale_payload.get("pid") or 0) if isinstance(stale_payload, dict) else 0
                    except (OSError, ValueError, TypeError):
                        stale_pid = 0
                    # The OS guard serializes current-version contenders across
                    # the whole recovery/create window; rename alone cannot do so.
                    quarantine = self.lock_path.with_name(
                        f".{self.lock_path.name}.stale.{self.token}")
                    try:
                        os.replace(self.lock_path, quarantine)
                    except FileNotFoundError:
                        continue
                    try:
                        # Inspect the quarantined file, never the now-vacant
                        # primary path. A legacy writer may not use the guard.
                        quarantine_payload = json.loads(
                            quarantine.read_text(encoding="utf-8"))
                        if not isinstance(quarantine_payload, dict):
                            quarantine_payload = {}
                        if int(quarantine_payload.get("pid") or 0) != stale_pid:
                            # 内容被换过 → 归还锁文件，回 O_EXCL 循环
                            os.replace(quarantine, self.lock_path)
                            continue
                        quarantine.unlink(missing_ok=True)
                        continue
                    except (OSError, ValueError, TypeError):
                        quarantine.unlink(missing_ok=True)
                    continue
                if self.mode == ProjectAccessMode.AUTO:
                    # 静默降级是历史事故源头（P1 评审）：锁存在且未恢复时
                    # AUTO 转只读，用户跑到写 catalog 才报错。至少留痕。
                    _LOGGER.warning(
                        "项目锁被占用且未恢复，会话降级为只读：%s",
                        self.root)
                    self.read_only = True
                    self._held = True
                    return self
                raise ProjectLockError(f"项目已有写入实例：{self.root}")
            else:
                flushed = False
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as stream:
                        json.dump(payload, stream, ensure_ascii=False, indent=2)
                        stream.flush()
                        os.fsync(stream.fileno())
                    fsync_directory(self.root)
                    flushed = True
                finally:
                    # This O_EXCL-created file is ours and the OS guard is held.
                    # Leaving a complete live-PID payload would prevent retries.
                    if not flushed:
                        self.lock_path.unlink(missing_ok=True)
                with _PROCESS_LOCK_GUARD:
                    _PROCESS_LOCKS[self.root] = (self.token, 1)
                self._held = True
                return self

    def _is_stale(self) -> bool:
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return True
        if not isinstance(payload, dict):
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
                if isinstance(payload, dict) and payload.get("token") == self.token:
                    self.lock_path.unlink(missing_ok=True)
                    fsync_directory(self.root)
            except (OSError, ValueError, TypeError):
                pass
            finally:
                self._held = False
        else:
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
        self._trees: dict[Path, bool] = {}
        self._catalog_path: str | None = None
        self.catalog_commit_key = 'file_transaction:' + self.transaction_id
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

    def bind_catalog(self, path: Path) -> None:
        self._catalog_path = str(Path(path).resolve().relative_to(self.root))
        self._write_journal("active")

    def track_tree(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        relative = target.relative_to(self.root)
        if not relative.parts or relative.parts[0] == '.transactions':
            raise ValueError("不能跟踪整个项目目录")
        if target in self._trees:
            return target
        # Finish all backups before publishing the tree entry.  A failed
        # snapshot must never make recovery remove untracked original files.
        for child in sorted(target.rglob("*")):
            if child.is_file():
                self.track(child)
        self._trees[target] = target.exists()
        self._write_journal("active")
        return target

    def track(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        try:
            relative = target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("事务只能跟踪项目目录内文件") from exc
        if not relative.parts or relative.parts[0] == '.transactions':
            raise ValueError('不能跟踪项目根目录或事务恢复目录')
        if target.is_dir():
            raise IsADirectoryError(target)
        if target in self._tracked:
            return target
        existed = target.exists()
        if existed and target.is_file():
            backup = self.backups / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            _atomic_copy_file(target, backup)
        self._tracked[target] = existed
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
            "trees": [{"path": str(path.relative_to(self.root)), "existed": existed}
                      for path, existed in self._trees.items()],
            "catalog_path": self._catalog_path,
            "catalog_commit_key": self.catalog_commit_key,
        })

    def commit(self) -> None:
        self._write_journal("committed")
        self._committed = True

    def rollback(self) -> None:
        _remove_created_tree_files(self.root, self._trees, self._tracked)
        for target, existed in reversed(list(self._tracked.items())):
            relative = target.relative_to(self.root)
            backup = self.backups / relative
            if existed and backup.exists():
                if not _same_file_contents(backup, target):
                    _atomic_copy_file(backup, target)
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


def _remove_created_tree_files(root: Path, trees: dict[Path, bool], tracked: dict[Path, bool]) -> None:
    for path in (*trees, *tracked):
        if not path.relative_to(root).parts:
            raise ValueError("事务不能指向整个项目目录")
    for tree, existed in trees.items():
        tree.relative_to(root)
        for path in sorted(tree.rglob("*"), reverse=True):
            path.resolve().relative_to(root)
            if path.is_file() and path.resolve() not in tracked:
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        if not existed and tree.exists() and not any(tree.iterdir()):
            tree.rmdir()


def _transaction_path(root: Path, relative: Any) -> Path:
    path = Path(str(relative or ''))
    if not path.parts or path.is_absolute() or '..' in path.parts:
        raise ValueError(f"unsafe transaction path: {relative}")
    resolved = (root / path).resolve()
    parts = resolved.relative_to(root).parts
    if not parts or parts[0] == '.transactions':
        raise ValueError("事务不能指向项目根目录或事务恢复目录")
    return resolved


def _catalog_committed(root: Path, payload: dict[str, Any]) -> bool:
    path = _transaction_path(root, payload['catalog_path'])
    key = str(payload.get('catalog_commit_key') or '')
    if not key:
        raise ValueError('missing catalog commit key')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        row = db.execute('SELECT value FROM catalog_meta WHERE key=?', (key,)).fetchone()
        return bool(row and row[0] == 'committed')
    finally:
        db.close()


def _restore_transaction_files(root: Path, stage: Path, payload: dict[str, Any]) -> None:
    files = payload.get('files')
    trees = payload.get('trees', [])
    if not isinstance(files, list) or not isinstance(trees, list):
        raise ValueError('transaction files and trees must be lists')
    # Validate every path and required backup before the first mutation.
    targets = [(_transaction_path(root, item.get('path')), bool(item.get('existed')))
               for item in files]
    tree_paths = {_transaction_path(root, item.get('path')): bool(item.get('existed'))
                  for item in trees}
    backups = {}
    for target, existed in targets:
        backup = (stage / 'before' / target.relative_to(root)).resolve()
        backup.relative_to(stage.resolve())
        if existed and not backup.is_file():
            raise FileNotFoundError(f'transaction backup missing: {backup}')
        backups[target] = backup
    _remove_created_tree_files(root, tree_paths, dict(targets))
    for target, existed in reversed(targets):
        if existed:
            if not _same_file_contents(backups[target], target):
                _atomic_copy_file(backups[target], target)
        else:
            target.unlink(missing_ok=True)
            fsync_directory(target.parent)


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
            if state not in {'active', 'committed', 'rolled_back'}:
                raise ValueError(f'unsupported transaction state: {state}')
            if payload.get('catalog_path'):
                state = 'committed' if _catalog_committed(project_root, payload) else 'active'
            if state == "active":
                _restore_transaction_files(project_root, stage_root, payload)
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
        except (OSError, ValueError, TypeError, KeyError, AttributeError, sqlite3.Error) as exc:
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
