from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.field_data_quality import DataOrientation, DataQualityStatus, evaluate_line_data_quality
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.trajectory_model import TrajectoryModel, TrajectoryPoint


def test_import_writes_quality_report_and_updates_line_quality(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="qc-test")
    line = store.import_line_file("L01", Path("sample_data/gui_sidecar_all_data_main.csv"), name="qc-line", copy_into_project=True)

    report_path = store.quality_report_path("L01")
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "mygpr.line_data_quality.v1"
    assert payload["sample_count"] == 10
    assert payload["trace_count"] == 12
    assert payload["orientation"] in {DataOrientation.FEW_SAMPLES, DataOrientation.SAMPLES_BY_TRACES}

    updated = store.get_line(line.line_id)
    assert updated.data_quality in {"通过", "警告"}


def test_transposed_matrix_risk_is_flagged() -> None:
    matrix = np.random.default_rng(12).normal(size=(600, 40)).astype("float32")
    dataset = GPRDataSet.from_matrix("LX", matrix, length_m=20.0, time_window_ns=700.0)
    trajectory = TrajectoryModel([TrajectoryPoint(distance_m=float(i), x=float(i), y=0.0) for i in range(40)])

    report = evaluate_line_data_quality(dataset, trajectory)

    assert report.status == DataQualityStatus.WARNING
    assert report.orientation == DataOrientation.TRANSPOSE_RISK
    assert any(issue.code == "transpose_risk" for issue in report.issues)


def test_project_quality_check_reports_all_imported_lines(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="project-qc")
    store.import_line_file("L01", Path("sample_data/gui_sidecar_all_data_main.csv"), name="line-a", copy_into_project=True)
    store.import_line_file("L02", Path("sample_data/uav_gpr_motion_demo_v1/main.csv"), name="line-b", copy_into_project=True)

    reports = store.run_project_quality_check()

    assert len(reports) == 2
    assert {report.line_id for report in reports} == {"L01", "L02"}
    assert all(store.quality_report_path(report.line_id).exists() for report in reports)
