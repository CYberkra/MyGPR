from __future__ import annotations

from core.job_manager import JobContext, JobProgress, JobRegistry, JobState


def test_job_registry_tracks_progress_success_and_snapshot() -> None:
    registry = JobRegistry(history_limit=20)
    record = registry.create("生成正式报告", category="报告", metadata={"line_id": "L01"})

    assert record.state == JobState.QUEUED
    registry.mark_running(record.job_id)
    registry.update_progress(record.job_id, JobProgress(3, 10, "生成图件", "L01"))
    done = registry.mark_succeeded(record.job_id, summary="报告包已生成")

    assert done.state == JobState.SUCCEEDED
    assert done.progress.percent == 100
    assert done.result_summary == "报告包已生成"
    snapshot = registry.snapshot()
    assert snapshot[0]["metadata"]["line_id"] == "L01"
    assert snapshot[0]["progress"]["percent"] == 100


def test_job_context_observes_cooperative_cancellation() -> None:
    registry = JobRegistry()
    record = registry.create("大文件处理", cancellable=True)
    updates: list[tuple[str, int]] = []
    context = JobContext(
        record.job_id,
        registry.event(record.job_id),
        lambda _job_id, progress: updates.append((progress.message, progress.percent)),
    )

    context.report(1, 4, "读取分块")
    assert updates == [("读取分块", 25)]
    assert registry.request_cancel(record.job_id) is True
    assert context.cancel_requested is True

    try:
        context.check_cancelled()
    except Exception as exc:
        assert exc.__class__.__name__ == "JobCancelled"
    else:  # pragma: no cover
        raise AssertionError("cancel must interrupt a cooperative worker")

    registry.mark_cancelled(record.job_id)
    assert registry.get(record.job_id).state == JobState.CANCELLED
