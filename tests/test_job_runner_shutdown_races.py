"""Deterministic submission and event delivery at the shutdown boundary."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from mygpr.application.jobs.models import JobEventType, JobStatus
from mygpr.application.jobs.runner import InMemoryJobRunner, JobRunnerClosedError
from mygpr.application.jobs.cancellation import JobCancelledError


def test_shutdown_wins_before_job_registration(monkeypatch):
    runner = InMemoryJobRunner(max_workers=1)
    entered, release, ran = Event(), Event(), Event()
    original = runner.prune

    def paused_prune():
        entered.set()
        assert release.wait(5)
        return original()

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(runner, "prune", paused_prune)
        submission = pool.submit(runner.submit, "late", lambda _ctx: ran.set())
        assert entered.wait(5)
        monkeypatch.setattr(runner, "prune", original)
        try:
            assert runner.shutdown() == ()
        finally:
            release.set()
        with pytest.raises(JobRunnerClosedError):
            submission.result(timeout=5)
    assert not ran.is_set()
    assert runner.list_snapshots() == ()


def test_shutdown_waits_for_submission_to_bind_future(monkeypatch):
    runner = InMemoryJobRunner(max_workers=1)
    entered, release, stopped, ran = Event(), Event(), Event(), Event()
    original = runner._executor.submit

    def paused_submit(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    def stop():
        result = runner.shutdown()
        stopped.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(runner._executor, "submit", paused_submit)
        submission = pool.submit(runner.submit, "admitted", lambda _ctx: ran.set())
        assert entered.wait(5)
        shutdown = pool.submit(stop)
        try:
            assert not stopped.wait(0.1)
        finally:
            release.set()
        job_id = submission.result(timeout=5)
        assert shutdown.result(timeout=5) == ()
    assert ran.is_set()
    assert runner.snapshot(job_id).status is JobStatus.COMPLETED


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
def test_nonblocking_shutdown_preserves_late_progress_and_terminal_event(outcome):
    runner = InMemoryJobRunner(max_workers=1)
    started, release, completed = Event(), Event(), Event()
    events = []

    def operation(context):
        started.set()
        assert release.wait(5)
        context.report_progress(1, 1, "late progress")
        if outcome == "failed":
            raise ValueError("late operation failure")
        if outcome == "cancelled":
            raise JobCancelledError("late operation cancellation")
        return "success"

    def receive(event):
        events.append(event)
        if event.event_type in {JobEventType.COMPLETED, JobEventType.FAILED, JobEventType.CANCELLED}:
            completed.set()

    job_id = runner.submit("late events", operation)
    runner.subscribe(job_id, receive)
    assert started.wait(5)
    try:
        assert runner.shutdown(wait=False) == (job_id,)
    finally:
        release.set()
    result = runner.wait(job_id, timeout=5)
    runner.shutdown()
    assert result.status.value == outcome
    if outcome == "completed":
        assert result.result == "success"
    elif outcome == "failed":
        assert result.error_type == "ValueError"
        assert result.error_message == "late operation failure"
    assert completed.wait(5)
    assert JobEventType.PROGRESS in {event.event_type for event in events}


def test_queued_cancel_and_failure_leave_runner_reusable():
    runner = InMemoryJobRunner(max_workers=1)
    started, release, queued_ran, finalized = Event(), Event(), Event(), Event()

    def blocked(_context):
        started.set()
        assert release.wait(5)

    try:
        first = runner.submit("block", blocked)
        assert started.wait(5)
        queued = runner.submit("queued", lambda _context: queued_ran.set(), finalizer=finalized.set)
        assert runner.cancel(queued)
        assert runner.wait(queued, timeout=5).status is JobStatus.CANCELLED
        assert finalized.wait(5)
        assert not queued_ran.is_set()
        release.set()
        assert runner.wait(first, timeout=5).status is JobStatus.COMPLETED
        failure = runner.submit("failure", lambda _ctx: (_ for _ in ()).throw(ValueError("fault")))
        assert runner.wait(failure, timeout=5).status is JobStatus.FAILED
        next_job = runner.submit("reused", lambda _ctx: 42)
        assert runner.wait(next_job, timeout=5).result == 42
    finally:
        release.set()
        runner.shutdown()
