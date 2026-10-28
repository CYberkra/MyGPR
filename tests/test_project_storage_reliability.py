from __future__ import annotations

import json
import hashlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import core.storage_primitives as primitives
from core.field_project_store import FieldProjectStore
from core.project_repository import ProjectRepository
from core.schema_registry import DocumentCorruptionError
from core.storage_primitives import FileTransaction, ProjectAccessMode, ProjectLock, atomic_write_json


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


def test_future_project_open_does_not_recover_pending_transaction(tmp_path: Path) -> None:
    root = tmp_path / "future"
    with FieldProjectStore.create_empty(root):
        payload = json.loads((root / "project.json").read_text(encoding="utf-8"))
    target = root / "metadata" / "future-state.json"
    target.write_bytes(b"old-state")
    transaction = FileTransaction(root)
    transaction.__enter__()
    transaction.track(target)
    target.write_bytes(b"future-in-progress-state")
    atomic_write_json(root / "project.json", {**payload, "schema": "mygpr.field_project.v99"})
    before = _snapshot(root)

    with FieldProjectStore.open(root, access_mode=ProjectAccessMode.WRITE) as opened:
        assert opened.read_only
        assert opened.session.recovery_results == ()

    assert _snapshot(root) == before


def test_read_only_corrupt_project_does_not_write_quarantine(tmp_path: Path) -> None:
    root = tmp_path / "corrupt"
    root.mkdir()
    (root / "project.json").write_bytes(b"{bad-json")
    before = _snapshot(root)
    with pytest.raises(DocumentCorruptionError):
        FieldProjectStore.open(root, access_mode=ProjectAccessMode.READ_ONLY)
    assert _snapshot(root) == before


def test_schema_changes_before_writer_lock_falls_back_without_recovery(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "future-race"
    with FieldProjectStore.create_empty(root):
        payload = json.loads((root / "project.json").read_text(encoding="utf-8"))
    target = root / "metadata" / "state.bin"
    target.write_bytes(b"before")
    transaction = FileTransaction(root)
    transaction.__enter__()
    transaction.track(target)
    target.write_bytes(b"pending-future-state")
    original_open = ProjectRepository.open_session
    expected = {}

    def change_before_lock(path, **kwargs):
        if kwargs["mode"] == ProjectAccessMode.WRITE:
            atomic_write_json(root / "project.json", {**payload, "schema": "mygpr.field_project.v99"})
            expected.update(_snapshot(root))
        return original_open(path, **kwargs)

    monkeypatch.setattr(ProjectRepository, "open_session", change_before_lock)
    with FieldProjectStore.open(root, access_mode=ProjectAccessMode.WRITE) as opened:
        assert opened.read_only
        assert opened.session.recovery_results == ()
        assert target.read_bytes() == b"pending-future-state"
    assert expected
    assert _snapshot(root) == expected
    assert not (root / ".mygpr.lock").exists()


@pytest.mark.parametrize("failure_stage", ["structure", "manifest", "log"])
def test_failed_project_creation_releases_writer_lease(tmp_path: Path, monkeypatch, failure_stage: str) -> None:
    root = tmp_path / failure_stage
    method = {"structure": "ensure_structure", "manifest": "save_manifest", "log": "append_log"}[failure_stage]

    def fail(*args, **kwargs):
        raise OSError("injected creation I/O failure")

    monkeypatch.setattr(FieldProjectStore, method, fail)
    with pytest.raises(OSError, match="injected"):
        FieldProjectStore.create_empty(root)
    assert not (root / ".mygpr.lock").exists()
    with ProjectRepository.open_session(root, mode=ProjectAccessMode.WRITE, allow_reentrant=False) as session:
        assert session.writable


def test_stale_recovery_does_not_delete_competing_process_lock(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "race"
    root.mkdir()
    lock_path = root / ".mygpr.lock"
    atomic_write_json(lock_path, {"pid": 0, "host": socket.gethostname(), "token": "stale"})
    result_path = tmp_path / "competitor.json"
    stop_path = tmp_path / "stop"
    code = """
import json, sys, time
from pathlib import Path
from core.storage_primitives import ProjectLock, ProjectAccessMode, ProjectLockError
root, result, stop = map(Path, sys.argv[1:])
try:
    lock = ProjectLock(root, mode=ProjectAccessMode.WRITE, allow_reentrant=False).acquire()
except ProjectLockError:
    result.write_text(json.dumps({'writable': False}), encoding='utf-8')
else:
    result.write_text(json.dumps({'writable': True, 'token': lock.token}), encoding='utf-8')
    try:
        deadline = time.monotonic() + 15
        while not stop.exists() and time.monotonic() < deadline:
            time.sleep(.01)
    finally:
        lock.release()

"""
    child = None
    original_replace = os.replace

    def replace_and_contend(source, destination):
        nonlocal child
        original_replace(source, destination)
        if Path(source) == lock_path and ".stale." in Path(destination).name:
            child = subprocess.Popen([sys.executable, "-c", code, str(root), str(result_path), str(stop_path)])
            deadline = time.monotonic() + 1
            while not result_path.exists() and time.monotonic() < deadline:
                time.sleep(.01)

    monkeypatch.setattr(primitives.os, "replace", replace_and_contend)
    lock = ProjectLock(root, mode=ProjectAccessMode.WRITE, recover_stale=True)
    try:
        lock.acquire()
        assert child is not None
        deadline = time.monotonic() + 5
        while not result_path.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["writable"] is False, "stale recovery admitted two live writers"
        assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == lock.token
    finally:
        stop_path.touch()
        if child is not None:
            child.wait(timeout=10)
        lock.release()


@pytest.mark.parametrize("payload", [[], {"pid": []}, {"pid": "broken"}])
def test_explicit_recovery_accepts_malformed_lock_payload(tmp_path: Path, payload) -> None:
    root = tmp_path / "malformed"
    root.mkdir()
    atomic_write_json(root / ".mygpr.lock", payload)
    with ProjectLock(root, mode=ProjectAccessMode.WRITE, recover_stale=True) as lock:
        lock.acquire()
        assert lock.writable


def test_killed_writer_recovers_data_and_preserves_process_exclusion(tmp_path: Path) -> None:
    root = tmp_path / "killed-writer"
    with FieldProjectStore.create_empty(root):
        pass
    target = root / "metadata" / "payload.bin"
    original = bytes(range(256)) * 16
    target.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    ready = tmp_path / "ready"
    code = """
import sys, time
from pathlib import Path
from core.project_repository import ProjectRepository, ProjectAccessMode
from core.storage_primitives import FileTransaction, atomic_write_bytes
root, ready = map(Path, sys.argv[1:])
session = ProjectRepository.open_session(root, mode=ProjectAccessMode.WRITE)
tx = FileTransaction(root)
tx.__enter__()
target = tx.track(root / 'metadata' / 'payload.bin')
atomic_write_bytes(target, b'partial-new-data')
ready.write_text('held-and-written', encoding='utf-8')
time.sleep(30)
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(root), str(ready)])
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.read_text(encoding="utf-8") == "held-and-written"
        with FieldProjectStore.open(root) as reader:
            assert reader.read_only
            assert target.read_bytes() == b"partial-new-data"
        child.kill()
        child.wait(timeout=10)
        with FieldProjectStore.open(root, access_mode=ProjectAccessMode.WRITE, recover_stale_lock=True) as writer:
            assert not writer.read_only
            assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
            assert writer.session.recovery_results[0]["action"] == "rolled_back"
            contender_code = """
import sys
from core.storage_primitives import ProjectLock, ProjectAccessMode, ProjectLockError
try:
    ProjectLock(sys.argv[1], mode=ProjectAccessMode.WRITE).acquire()
except ProjectLockError:
    sys.exit(0)
sys.exit(1)
"""
            result = subprocess.run([sys.executable, "-c", contender_code, str(root)], timeout=10)
            assert result.returncode == 0
        assert not (root / ".mygpr.lock").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_lock_flush_failure_does_not_leave_live_pid_lock(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "lock-flush"
    root.mkdir()
    original_fsync = primitives.os.fsync

    def fail(_fd):
        raise OSError("injected lock flush failure")

    monkeypatch.setattr(primitives.os, "fsync", fail)
    with pytest.raises(OSError, match="lock flush"):
        ProjectLock(root, mode=ProjectAccessMode.WRITE).acquire()
    monkeypatch.setattr(primitives.os, "fsync", original_fsync)
    assert not (root / ".mygpr.lock").exists()
    with ProjectRepository.open_session(root, mode=ProjectAccessMode.WRITE, allow_reentrant=False) as session:
        assert session.writable


def test_release_damaged_lock_revokes_local_write_without_deleting_unknown_owner(tmp_path: Path) -> None:
    root = tmp_path / "release-damage"
    lock = ProjectLock(root, mode=ProjectAccessMode.WRITE).acquire()
    atomic_write_json(root / ".mygpr.lock", [])
    lock.release()
    assert not lock.writable
    assert (root / ".mygpr.lock").read_text(encoding="utf-8") == "[]"
    lock.release()
