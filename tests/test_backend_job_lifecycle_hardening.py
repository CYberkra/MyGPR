from __future__ import annotations

from threading import Event
import time
from pathlib import Path

import numpy as np
import pytest

from mygpr.application.jobs.models import JobResultSummary, JobRetentionPolicy, JobStatus
from mygpr.application.jobs.runner import InMemoryJobRunner, JobRunnerClosedError
from mygpr.application.project.service import ProjectBusyError
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def test_runner_stops_accepting_and_bounds_events_and_large_results() -> None:
    runner = InMemoryJobRunner(
        max_workers=1,
        retention_policy=JobRetentionPolicy(max_events_per_job=8, max_result_bytes=1024),
    )

    def operation(context):
        for index in range(30):
            context.report_progress(index, 30, str(index))
        return np.ones((64, 64), dtype=np.float64)

    job_id = runner.submit("large", operation)
    snapshot = runner.wait(job_id, timeout=5)
    assert snapshot.status is JobStatus.COMPLETED
    assert snapshot.result_released
    assert isinstance(snapshot.result, JobResultSummary)
    assert len(runner.events(job_id)) <= 8
    runner.stop_accepting()
    with pytest.raises(JobRunnerClosedError):
        runner.submit("rejected", lambda _context: None)
    runner.shutdown()


def test_forget_rejects_active_job_and_removes_terminal_job() -> None:
    runner = InMemoryJobRunner(max_workers=1)
    release = Event()
    job_id = runner.submit("wait", lambda _context: release.wait(timeout=2))
    assert not runner.forget(job_id)
    release.set()
    assert runner.wait(job_id, timeout=3).is_terminal
    assert runner.forget(job_id)
    with pytest.raises(KeyError):
        runner.snapshot(job_id)
    runner.shutdown()


def test_project_job_lease_blocks_close_until_terminal_cleanup(tmp_path: Path) -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="lease")
        data = np.arange(64 * 20, dtype=np.float32).reshape(64, 20)
        backend.projects.save_line_dataset(project.project_id, "L01", data)
        gate = Event()
        original = backend.project_processing.execute_pipeline

        def delayed(*args, **kwargs):
            gate.wait(timeout=3)
            return original(*args, **kwargs)

        backend.project_processing.execute_pipeline = delayed  # type: ignore[method-assign]
        job_id = backend.submit_project_pipeline(
            project.project_id,
            "L01",
            PipelineDefinition(steps=(PipelineStep("dewow", {"window": 9}),)),
        )
        assert backend.projects.active_lease_count(project.project_id) == 1
        with pytest.raises(ProjectBusyError):
            backend.projects.close_project(project.project_id)
        gate.set()
        assert backend.jobs.wait(job_id, timeout=10).is_terminal
        for _ in range(100):
            if backend.projects.active_lease_count(project.project_id) == 0:
                break
            time.sleep(0.005)
        assert backend.projects.active_lease_count(project.project_id) == 0
        backend.projects.close_project(project.project_id)
    finally:
        backend.shutdown()


def test_runner_enforces_global_result_memory_budget() -> None:
    runner = InMemoryJobRunner(
        max_workers=1,
        retention_policy=JobRetentionPolicy(
            max_result_bytes=2 * 1024 * 1024,
            max_total_result_bytes=1536 * 1024,
        ),
    )
    try:
        first = runner.submit("first", lambda _context: np.ones((256, 1024), dtype=np.float32))
        assert runner.wait(first, timeout=5).status is JobStatus.COMPLETED
        second = runner.submit("second", lambda _context: np.ones((256, 1024), dtype=np.float32))
        assert runner.wait(second, timeout=5).status is JobStatus.COMPLETED
        first_snapshot = runner.snapshot(first)
        second_snapshot = runner.snapshot(second)
        assert first_snapshot.result_released
        assert isinstance(first_snapshot.result, JobResultSummary)
        assert not second_snapshot.result_released
        assert isinstance(second_snapshot.result, np.ndarray)
    finally:
        runner.shutdown()


def test_runner_bounds_large_event_payloads() -> None:
    runner = InMemoryJobRunner(
        max_workers=1,
        retention_policy=JobRetentionPolicy(max_event_payload_bytes=1024),
    )

    def operation(context):
        context.emit_warning({"message": "large", "values": np.ones((1024,), dtype=np.float64)})
        return None

    try:
        job_id = runner.submit("payload", operation)
        assert runner.wait(job_id, timeout=5).status is JobStatus.COMPLETED
        warning_event = next(event for event in runner.events(job_id) if event.event_type.value == "warning_raised")
        assert warning_event.payload["payload_released"] is True
        assert warning_event.payload["estimated_bytes"] > 1024
        assert "values" in warning_event.payload["keys"]
    finally:
        runner.shutdown()
