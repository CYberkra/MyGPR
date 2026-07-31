from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import h5py
import numpy as np

import core.storage_primitives as storage_primitives
from core.field_project_models import FieldLineRecord
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.hdf5_array_proxy import HDF5ArrayProxy
from core.hdf5_line_container import RAW_MATRIX_PATH, delete_processing_artifact, write_processing_artifact
from core.project_integrity import ProjectIntegrityAuditor
from core.project_repository import ProjectRepository
from core.storage_primitives import (
    FileTransaction,
    ProjectAccessMode,
    ProjectLock,
    atomic_write_bytes,
)
from mygpr.interfaces.backend import MyGPRBackend


def _project(root: Path) -> FieldProjectStore:
    store = FieldProjectStore.create_empty(root, name="phase4-storage")
    store.upsert_line(FieldLineRecord("L01", "Line 01"))
    matrix = np.arange(64 * 40, dtype=np.float32).reshape(64, 40)
    store.save_gpr_dataset(
        "L01",
        GPRDataSet.from_matrix("L01", matrix, length_m=20.0, time_window_ns=500.0),
    )
    return store


def test_interrupted_file_transaction_is_rolled_back_on_recovery(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "project.json"
    target.write_bytes(b"before")

    transaction = FileTransaction(root, label="simulated-crash")
    transaction.__enter__()
    transaction.track(target)
    atomic_write_bytes(target, b"after")
    # Deliberately do not exit/commit: this simulates abrupt process death.

    session = ProjectRepository.open_session(root, mode=ProjectAccessMode.WRITE)
    try:
        assert target.read_bytes() == b"before"
        assert session.recovery_results and session.recovery_results[0]["action"] == "rolled_back"
        assert not (root / ".transactions" / transaction.transaction_id).exists()
    finally:
        session.close()


def test_reentrant_session_does_not_recover_live_transaction(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "state.json"
    target.write_bytes(b"before")
    first = ProjectRepository.open_session(root, mode=ProjectAccessMode.WRITE)
    transaction = FileTransaction(root, label="live-write")
    transaction.__enter__()
    transaction.track(target)
    atomic_write_bytes(target, b"in-progress")
    second = ProjectRepository.open_session(root, mode=ProjectAccessMode.WRITE)
    try:
        assert second.lock.reentrant
        assert second.recovery_results == ()
        assert target.read_bytes() == b"in-progress"
        assert transaction.journal_path.exists()
    finally:
        second.close()
        transaction.__exit__(None, None, None)
        first.close()
    assert target.read_bytes() == b"before"


def test_project_lock_v2_rejects_reused_pid_marker(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    lock_path = root / ".mygpr.lock"
    lock_path.write_text(
        json.dumps(
            {
                "schema": "mygpr.project_lock.v2",
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "boot_id": storage_primitives._boot_id(),
                "process_start": "definitely-not-current-process",
                "token": "stale-token",
            }
        ),
        encoding="utf-8",
    )

    lock = ProjectLock(root, mode=ProjectAccessMode.WRITE, recover_stale=True).acquire()
    try:
        assert lock.writable
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["schema"] == "mygpr.project_lock.v2"
        assert payload["process_start"] == storage_primitives._process_start_marker(os.getpid())
    finally:
        lock.release()


def test_hybrid_transaction_rolls_forward_catalog_after_crash(tmp_path: Path) -> None:
    root = tmp_path / "project"
    store = _project(root)
    artifact_id = "L01_recovered_artifact"
    container = store.storage.line_container_path("L01")
    transaction = store.storage.transaction_journal.begin(
        line_id="L01",
        artifact_id=artifact_id,
        branch_id="L01:main",
        parent_artifact_id="",
        h5_path=container.relative_to(root).as_posix(),
        manifest={
            "line_id": "L01",
            "method_id": "noop",
            "method_name": "No-op",
            "saved_at": "2026-07-22T00:00:00+00:00",
            "branch_id": "L01:main",
        },
        params={},
    )
    write_processing_artifact(
        container,
        artifact_id=artifact_id,
        matrix=np.ones((64, 40), dtype=np.float32),
        manifest=transaction.payload["manifest"],
        params={},
    )
    # Journal remains in prepared state and SQLite has no artifact: crash point.
    assert store.storage.catalog.get_artifact(artifact_id) is None
    store.close()

    reopened = FieldProjectStore.open(root, access_mode=ProjectAccessMode.WRITE)
    try:
        row = reopened.storage.catalog.get_artifact(artifact_id)
        assert row is not None
        assert row["dataset_path"].endswith(f"/{artifact_id}/bscan")
        assert reopened.storage.transaction_journal.pending_paths() == ()
        assert any(
            action.artifact_id == artifact_id and action.action == "roll_forward_catalog"
            for action in reopened.storage.last_recovery_actions
        )
    finally:
        reopened.close()


def test_hybrid_transaction_rolls_back_catalog_when_hdf5_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "project"
    store = _project(root)
    raw = store.load_gpr_dataset("L01")
    saved, _ = store.save_processed_line(
        "L01", np.asarray(raw.matrix), {"method": "noop", "params": {}, "branch_id": "L01:main"}
    )
    artifact_id = saved.stem
    container = store.storage.line_container_path("L01")
    record = store.storage.catalog.get_artifact(artifact_id)
    assert record is not None
    store.storage.transaction_journal.begin(
        line_id="L01",
        artifact_id=artifact_id,
        branch_id=str(record.get("branch_id") or "L01:main"),
        parent_artifact_id=str(record.get("parent_artifact_id") or ""),
        h5_path=container.relative_to(root).as_posix(),
        manifest=dict(record.get("manifest") or {}),
        params=dict(record.get("params") or {}),
    )
    assert delete_processing_artifact(container, artifact_id)
    store.close()

    reopened = FieldProjectStore.open(root, access_mode=ProjectAccessMode.WRITE)
    try:
        assert reopened.storage.catalog.get_artifact(artifact_id) is None
        assert any(
            action.artifact_id == artifact_id and action.action == "rollback_catalog"
            for action in reopened.storage.last_recovery_actions
        )
    finally:
        reopened.close()


def test_deep_integrity_recomputes_raw_hdf5_hash(tmp_path: Path) -> None:
    store = _project(tmp_path / "project")
    try:
        healthy = ProjectIntegrityAuditor(store).audit(deep_hash=True, persist=False)
        assert "storage.raw_hash_mismatch" not in {issue.code for issue in healthy.issues}

        container = store.storage.line_container_path("L01")
        with h5py.File(container, "r+") as handle:
            handle[RAW_MATRIX_PATH][0, 0] += np.float32(7.0)
            handle.flush()

        fast = ProjectIntegrityAuditor(store).audit(deep_hash=False, persist=False)
        assert "storage.raw_hash_mismatch" not in {issue.code for issue in fast.issues}
        deep = ProjectIntegrityAuditor(store).audit(deep_hash=True, persist=False)
        mismatch = next(issue for issue in deep.issues if issue.code == "storage.raw_hash_mismatch")
        assert mismatch.severity == "error"
        assert mismatch.details["expected"] != mismatch.details["actual"]
    finally:
        store.close()


def test_backend_iterates_hdf5_blocks_without_full_materialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "project", name="chunked")
        matrix = np.arange(45 * 30, dtype=np.float32).reshape(45, 30)
        backend.projects.save_line_dataset(summary.project_id, "L01", matrix, length_m=12.0)

        def fail_materialize(self, dtype=None, copy=None):
            raise AssertionError("full HDF5 materialization is forbidden in block iterator")

        monkeypatch.setattr(HDF5ArrayProxy, "__array__", fail_materialize)
        blocks = list(
            backend.projects.iter_dataset_blocks(
                summary.project_id,
                "L01",
                block_rows=11,
                sample_start=3,
                sample_end=40,
                trace_start=4,
                trace_end=24,
            )
        )
        rebuilt = np.vstack([block for _start, _end, block in blocks])
        assert rebuilt.shape == (37, 20)
        np.testing.assert_array_equal(rebuilt, matrix[3:40, 4:24])
        assert blocks[0][0:2] == (3, 14)
        assert blocks[-1][1] == 40
    finally:
        backend.shutdown()
