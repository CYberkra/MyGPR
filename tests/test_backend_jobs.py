#!/usr/bin/env python3
"""Job event, failure and cooperative cancellation contracts."""
from __future__ import annotations

from threading import Event
import time

from mygpr.application.jobs.models import JobEventType, JobStatus
from mygpr.application.jobs.runner import InMemoryJobRunner


def test_job_runner_completes_and_emits_ordered_events() -> None:
    runner = InMemoryJobRunner(max_workers=1)
    release = Event()
    events = []

    def operation(context):
        release.wait(timeout=2.0)
        context.report_progress(1, 2, "half")
        context.report_progress(2, 2, "done")
        return {"ok": True}

    job_id = runner.submit("demo", operation)
    runner.subscribe(job_id, events.append)
    release.set()
    snapshot = runner.wait(job_id, timeout=3.0)
    runner.shutdown()

    assert snapshot.status is JobStatus.COMPLETED
    assert snapshot.result == {"ok": True}
    assert events
    assert [event.sequence for event in events] == sorted(event.sequence for event in events)
    assert JobEventType.PROGRESS in {event.event_type for event in events}
    assert JobEventType.COMPLETED in {event.event_type for event in events}
    retained = runner.events(job_id)
    assert retained[0].event_type is JobEventType.QUEUED
    assert retained[-1].event_type is JobEventType.COMPLETED
    assert runner.events(job_id, after_sequence=retained[-2].sequence) == (retained[-1],)
    assert snapshot.progress == 1.0
    assert snapshot.is_terminal


def test_job_runner_turns_operation_exception_into_failed_snapshot() -> None:
    runner = InMemoryJobRunner(max_workers=1)

    def operation(_context):
        raise ValueError("bad input")

    job_id = runner.submit("failure", operation)
    snapshot = runner.wait(job_id, timeout=3.0)
    runner.shutdown()

    assert snapshot.status is JobStatus.FAILED
    assert snapshot.error_type == "ValueError"
    assert snapshot.error_message == "bad input"


def test_job_runner_cancels_cooperative_operation() -> None:
    runner = InMemoryJobRunner(max_workers=1)
    started = Event()

    def operation(context):
        started.set()
        while True:
            context.raise_if_cancelled()
            time.sleep(0.005)

    job_id = runner.submit("cancel", operation)
    assert started.wait(timeout=2.0)
    assert runner.cancel(job_id)
    snapshot = runner.wait(job_id, timeout=3.0)
    runner.shutdown()

    assert snapshot.status is JobStatus.CANCELLED


def test_job_failure_exposes_stable_error_contract() -> None:
    runner = InMemoryJobRunner(max_workers=1)
    job_id = runner.submit("failure-contract", lambda _context: (_ for _ in ()).throw(ValueError("bad")))
    snapshot = runner.wait(job_id, timeout=3)
    assert snapshot.error_code == "MYGPR_JOB_ERROR"
    assert snapshot.error_details["schema"] == "mygpr.error_info.v1"
    failed = [event for event in runner.events(job_id) if event.event_type is JobEventType.FAILED]
    assert len(failed) == 1
    assert failed[0].payload["error_code"] == snapshot.error_code
    runner.shutdown()
