from __future__ import annotations

from pathlib import Path

import numpy as np

from core.field_project_operations import create_project, import_line_data
from core.field_project_status import build_project_status_snapshot
from core.project_events import ProjectEvent, ProjectEventType
from core.project_state_tracker import ProjectStateTracker, load_project_state


def test_project_event_marks_spatial_and_report_stale(tmp_path: Path) -> None:
    store = create_project(tmp_path, name="linkage-state")
    event = ProjectEvent.create(
        ProjectEventType.TARGETS_CHANGED,
        project_root=store.root,
        line_id="L01",
        reason="L01 目标标注已变化",
    )

    state = ProjectStateTracker(store.root).record_event(event)

    assert (store.root / "metadata" / "project_state.json").exists()
    assert state["dirty"]["spatial"] is True
    assert state["dirty"]["report"] is True
    assert "L01 目标标注已变化" in state["stale_reasons"]["spatial"]
    assert "L01 目标标注已变化" in state["stale_reasons"]["report"]


def test_spatial_refresh_clears_spatial_but_keeps_report_stale(tmp_path: Path) -> None:
    store = create_project(tmp_path, name="linkage-spatial")
    tracker = ProjectStateTracker(store.root)
    tracker.record_event(ProjectEvent.create(ProjectEventType.TARGETS_CHANGED, project_root=store.root, line_id="L01", reason="目标变化"))
    refreshed = tracker.record_event(ProjectEvent.create(ProjectEventType.SPATIAL_RESULTS_REFRESHED, project_root=store.root, reason="空间成果已刷新"))

    assert refreshed["dirty"]["spatial"] is False
    assert refreshed["stale_reasons"]["spatial"] == []
    assert refreshed["dirty"]["report"] is True
    assert "空间成果已刷新" in refreshed["stale_reasons"]["report"]


def test_report_generation_clears_report_stale(tmp_path: Path) -> None:
    store = create_project(tmp_path, name="linkage-report")
    tracker = ProjectStateTracker(store.root)
    tracker.record_event(ProjectEvent.create(ProjectEventType.PROCESSING_RESULT_SAVED, project_root=store.root, line_id="L01", reason="处理结果已保存"))
    state = tracker.record_event(ProjectEvent.create(ProjectEventType.REPORT_GENERATED, project_root=store.root, reason="报告已生成"))

    assert state["dirty"]["report"] is False
    assert state["stale_reasons"]["report"] == []


def test_project_status_exposes_dirty_report_and_attention(tmp_path: Path) -> None:
    store = create_project(tmp_path, name="linkage-status")
    ProjectStateTracker(store.root).record_event(
        ProjectEvent.create(ProjectEventType.REPORT_MARKED_STALE, project_root=store.root, reason="项目数据已变化")
    )

    snapshot = build_project_status_snapshot(store)

    assert snapshot.dirty_modules["report"] is True
    assert snapshot.report_status == "需重新生成" or snapshot.report_status != "已生成"
    assert any("成果报告需重新生成" in item[1] for item in snapshot.attention_items)


def test_line_import_event_keeps_source_and_report_state(tmp_path: Path) -> None:
    store = create_project(tmp_path, name="linkage-import")
    source = tmp_path / "external" / "line.npy"
    source.parent.mkdir()
    np.save(source, np.arange(24 * 16, dtype=np.float32).reshape(24, 16))
    line = import_line_data(store, source, name="导入测线")

    state = ProjectStateTracker(store.root).record_event(
        ProjectEvent.create(ProjectEventType.LINE_IMPORTED, project_root=store.root, line_id=line.line_id, reason=f"{line.line_id} 测线数据已导入")
    )

    assert state["dirty"]["report"] is True
    assert load_project_state(store.root)["last_events"][-1]["event_type"] == ProjectEventType.LINE_IMPORTED
