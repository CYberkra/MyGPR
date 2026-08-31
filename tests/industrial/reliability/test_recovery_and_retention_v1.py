from __future__ import annotations

from pathlib import Path
from threading import Event

import numpy as np
import pytest

from core.field_project_models import FieldLineRecord
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.hdf5_line_container import write_processing_artifact
from core.project_repository import ProjectRepository
from core.storage_primitives import FileTransaction, ProjectAccessMode, atomic_write_bytes
from mygpr.application.jobs.models import JobResultSummary, JobRetentionPolicy, JobStatus
from mygpr.application.jobs.runner import InMemoryJobRunner
from mygpr.interfaces.backend import MyGPRBackend

pytestmark = [
    pytest.mark.industrial,
    pytest.mark.reliability,
    pytest.mark.integration,
    pytest.mark.requirement("REQ-STO-001", "REQ-STO-002", "REQ-JOB-001", "REQ-JOB-002"),
    pytest.mark.risk("RISK-DATA-LOSS", "RISK-HALF-COMMIT", "RISK-JOB-MEMORY", "RISK-SHUTDOWN-RACE"),
    pytest.mark.level("component"),
]


def _store(root: Path) -> FieldProjectStore:
    store = FieldProjectStore.create_empty(root, name="industrial-recovery")
    store.upsert_line(FieldLineRecord("L01", "Line 01"))
    data = np.arange(48 * 24, dtype=np.float32).reshape(48, 24)
    store.save_gpr_dataset("L01", GPRDataSet.from_matrix("L01", data, time_window_ns=300.0))
    return store


def test_file_and_hybrid_transactions_recover_idempotently(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "project.json"
    target.write_bytes(b"stable")
    transaction = FileTransaction(root, label="industrial-crash")
    transaction.__enter__()
    transaction.track(target)
    atomic_write_bytes(target, b"partial")

    first = ProjectRepository.open_session(root, mode=ProjectAccessMode.WRITE)
    try:
        assert target.read_bytes() == b"stable"
        assert first.recovery_results and first.recovery_results[0]["action"] == "rolled_back"
    finally:
        first.close()
    second = ProjectRepository.open_session(root, mode=ProjectAccessMode.WRITE)
    try:
        assert second.recovery_results == ()
        assert target.read_bytes() == b"stable"
    finally:
        second.close()

    store = _store(tmp_path / "hybrid")
    artifact_id = "L01_pending_industrial"
    container = store.storage.line_container_path("L01")
    journal = store.storage.transaction_journal.begin(
        line_id="L01", artifact_id=artifact_id, branch_id="L01:main", parent_artifact_id="",
        h5_path=container.relative_to(store.root).as_posix(),
        manifest={"line_id": "L01", "method_id": "noop", "method_name": "No-op", "saved_at": "2026-07-22T00:00:00+00:00", "branch_id": "L01:main"},
        params={},
    )
    write_processing_artifact(container, artifact_id=artifact_id, matrix=np.ones((48, 24), dtype=np.float32), manifest=journal.payload["manifest"], params={})
    store.close()
    recovered = FieldProjectStore.open(tmp_path / "hybrid", access_mode=ProjectAccessMode.WRITE)
    try:
        assert recovered.storage.catalog.get_artifact(artifact_id) is not None
        assert recovered.storage.transaction_journal.pending_paths() == ()
    finally:
        recovered.close()
    reopened = FieldProjectStore.open(tmp_path / "hybrid", access_mode=ProjectAccessMode.WRITE)
    try:
        assert reopened.storage.last_recovery_actions == ()
        assert reopened.storage.catalog.get_artifact(artifact_id) is not None
    finally:
        reopened.close()


def test_job_retention_and_shutdown_release_resources_before_projects(tmp_path: Path) -> None:
    runner = InMemoryJobRunner(
        max_workers=1,
        retention_policy=JobRetentionPolicy(max_events_per_job=6, max_result_bytes=1024, max_total_result_bytes=1024),
    )
    try:
        def bounded_operation(context):
            for index in range(20):
                context.report_progress(index, 20, str(index))
            return np.ones((128, 128), dtype=np.float64)

        job_id = runner.submit("bounded", bounded_operation)
        snapshot = runner.wait(job_id, timeout=10)
        assert snapshot.status is JobStatus.COMPLETED
        assert snapshot.result_released and isinstance(snapshot.result, JobResultSummary)
        assert len(runner.events(job_id)) <= runner._retention.max_events_per_job
    finally:
        assert runner.shutdown(wait=True, cancel_running=True, timeout=5) == ()

    backend = MyGPRBackend.create_default(max_workers=1)
    gate = Event()
    try:
        project = backend.projects.create_project(tmp_path / "shutdown-project", name="shutdown")
        original_close_all = backend.projects.close_all
        order: list[str] = []

        def operation(context):
            gate.wait(timeout=5)
            context.raise_if_cancelled()
            return "done"

        backend.jobs.submit("active", operation, resource_keys=(f"project:{project.project_id}",))

        def tracked_close_all(*args, **kwargs):
            order.append("projects-closed")
            assert not backend.jobs.active_job_ids()
            return original_close_all(*args, **kwargs)

        backend.projects.close_all = tracked_close_all  # type: ignore[method-assign]
        gate.set()
        backend.shutdown(timeout_s=10)
        assert order == ["projects-closed"]
    finally:
        gate.set()
        if backend.jobs.accepting:
            backend.shutdown()
