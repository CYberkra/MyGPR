from __future__ import annotations

from pathlib import Path

import numpy as np

from core.field_project_operations import RecentProjectsStore, create_project, import_line_data
from core.field_project_status import build_project_status_snapshot


def test_project_status_snapshot_uses_project_files(tmp_path: Path) -> None:
    recent = RecentProjectsStore(tmp_path / "recent.json")
    store = create_project(tmp_path, name="真实指标测试", recent_store=recent)
    matrix_path = tmp_path / "line.npy"
    np.save(matrix_path, np.ones((64, 32), dtype=np.float32))
    line = import_line_data(store, matrix_path, name="L01 测试线")
    store.save_processed_line(line.line_id, np.ones((64, 32), dtype=np.float32), {"method": "dewow"})
    store.save_targets(line.line_id, [
        {"target_id": "T-01", "line_id": line.line_id, "distance_m": 1.0, "depth_m": 0.5, "status": "已确认"},
        {"target_id": "T-02", "line_id": line.line_id, "distance_m": 2.0, "depth_m": 0.7, "status": "待复核"},
    ])

    snapshot = build_project_status_snapshot(store)

    assert snapshot.line_count == 1
    assert snapshot.imported_line_count == 1
    assert snapshot.processed_line_count == 1
    assert snapshot.target_count == 2
    assert snapshot.confirmed_target_count == 1
    assert snapshot.pending_target_count == 1
    assert snapshot.spatial_point_count == 2
    assert snapshot.raw_size_mb > 0
    assert snapshot.task_rows
    assert snapshot.activity_rows


def test_empty_project_status_is_not_demo_metrics(tmp_path: Path) -> None:
    store = create_project(tmp_path, name="空项目", recent_store=RecentProjectsStore(tmp_path / "recent.json"))
    snapshot = build_project_status_snapshot(store)
    assert snapshot.line_count == 0
    assert snapshot.imported_line_count == 0
    assert snapshot.report_status == "未生成"
    assert any("暂无测线" in item[1] for item in snapshot.attention_items)
