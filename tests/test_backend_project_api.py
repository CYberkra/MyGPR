from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.field_project_models import FieldProjectManifest
from mygpr.application.jobs.cancellation import CancellationTokenSource, JobCancelledError
from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.jobs.models import JobStatus
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend
from tests.qt_import_isolation import assert_qt_imports_unchanged, qt_module_snapshot


def _data(rows: int = 72, cols: int = 48) -> np.ndarray:
    y = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    return (0.08 * y + np.exp(-((y - 0.55 - 0.03 * np.sin(x * 6.28)) ** 2) / 0.003)).astype(np.float32)


def test_project_api_create_window_process_audit_backup_restore(tmp_path: Path) -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "project", name="Backend Project")
        line = backend.projects.save_line_dataset(
            summary.project_id,
            "L01",
            _data(),
            name="Line 01",
            length_m=23.5,
            time_window_ns=500.0,
        )
        assert line.sample_count == 72
        assert line.trace_count == 48
        info = backend.projects.get_dataset_info(summary.project_id, "L01")
        assert info.shape == (72, 48)
        window, sample_indices, trace_indices = backend.projects.read_window(
            summary.project_id,
            "L01",
            sample_start=10,
            sample_end=30,
            trace_start=5,
            trace_end=20,
        )
        assert window.shape == (20, 15)
        assert sample_indices[0] == 10
        assert trace_indices[0] == 5

        pipeline = PipelineDefinition(
            name="Backend persisted pipeline",
            steps=(
                PipelineStep("dewow", {"window": 11}),
                PipelineStep("agcGain", {"window": 9}),
            ),
        )
        job_id = backend.submit_project_pipeline(summary.project_id, "L01", pipeline)
        snapshot = backend.jobs.wait(job_id, timeout=30)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        artifact = snapshot.result
        assert artifact.line_id == "L01"
        assert artifact.shape == (72, 48)
        assert artifact.data_reference.startswith("h5://")
        assert backend.projects.list_artifacts(summary.project_id, "L01")[0].artifact_id == artifact.artifact_id
        assert any(event.event_type.value == "artifact_created" for event in backend.jobs.events(job_id))

        report = backend.projects.audit_project(summary.project_id, clean_staging=True, staging_min_age_s=0)
        assert report.healthy

        backup = backend.projects.backup_project(summary.project_id, tmp_path / "backups")
        assert backup.verified
        assert Path(backup.archive_path).is_file()
        backend.projects.close_project(summary.project_id)
        restored = backend.projects.restore_project(
            backup.archive_path,
            tmp_path / "restored",
            project_dir_name="restored_project",
        )
        assert restored.verified
        reopened = backend.projects.list_open_projects()[0]
        assert reopened.project_id == summary.project_id
        assert backend.projects.get_dataset_info(reopened.project_id, "L01").shape == (72, 48)
    finally:
        backend.shutdown()


def test_invalid_manifest_line_id_is_rejected_at_deserialization_boundary() -> None:
    payload = {
        "schema": "mygpr.field_project.v3",
        "project_id": "project-id",
        "name": "unsafe",
        "lines": [{"line_id": "../escape", "name": "bad"}],
    }
    with pytest.raises(ValueError, match="非法测线编号"):
        FieldProjectManifest.from_dict(payload)


def test_backend_project_import_does_not_require_qt(tmp_path: Path) -> None:
    qt_before = qt_module_snapshot()
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "headless", name="Headless")
        assert summary.project_id
        assert_qt_imports_unchanged(qt_before)
    finally:
        backend.shutdown()


def test_reporting_service_generates_project_relative_package(tmp_path: Path) -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "report-project", name="Report Project")
        backend.projects.save_line_dataset(
            summary.project_id,
            "L01",
            _data(32, 24),
            length_m=12.0,
        )
        result = backend.reporting.generate_package(
            summary.project_id,
            package_name="backend_report",
        )
        root = Path(summary.root_path)
        assert result.package_dir == "reports/backend_report"
        assert (root / result.manifest_path).is_file()
        assert (root / result.pdf_path).is_file()
        assert (root / result.delivery_zip_path).is_file()
    finally:
        backend.shutdown()


def test_cancelled_project_artifact_commit_leaves_no_artifact(tmp_path: Path) -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "cancel-project", name="Cancel Project")
        raw = _data(32, 20)
        backend.projects.save_line_dataset(summary.project_id, "L01", raw, length_m=8.0)
        token_source = CancellationTokenSource()
        token_source.cancel()
        context = ExecutionContext(cancellation_token=token_source.token)
        with pytest.raises(JobCancelledError):
            backend.projects.save_processing_artifact(
                summary.project_id,
                "L01",
                raw,
                name="cancelled",
                method_id="noop",
                method_name="noop",
                params={},
                pipeline=[],
                branch_id="L01:main",
                input_dataset={},
                context=context,
            )
        assert backend.projects.list_artifacts(summary.project_id, "L01") == ()
    finally:
        backend.shutdown()


def test_project_api_chained_processing_records_parent(tmp_path: Path) -> None:
    """回归：从成果继续处理（input_artifact_id）须在成果上记录血缘 parent_artifact_id。

    P1-2：此前 GUI 保存路径 payload 顶层无 parent_artifact_id → _base_manifest 恒取 ""
    → 无处理谱系可追溯。
    """
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "chained", name="Chained")
        backend.projects.save_line_dataset(
            summary.project_id, "L01", _data(), name="L01",
            length_m=23.5, time_window_ns=500.0,
        )
        p1 = PipelineDefinition(name="p1", steps=(PipelineStep("dewow", {"window": 11}),))
        job1 = backend.submit_project_pipeline(summary.project_id, "L01", p1, result_name="成果A")
        snap1 = backend.jobs.wait(job1, timeout=30)
        assert snap1.status is JobStatus.COMPLETED, snap1.error_message
        artifact_a = snap1.result
        assert artifact_a is not None and artifact_a.artifact_id
        assert artifact_a.parent_artifact_id == ""  # 从原始数据处理，无父

        # 从成果 A 继续处理 → 成果 B 必须记录 parent=A
        p2 = PipelineDefinition(name="p2", steps=(PipelineStep("agcGain", {"window": 9}),))
        job2 = backend.submit_project_pipeline(
            summary.project_id, "L01", p2, result_name="成果B",
            input_artifact_id=artifact_a.artifact_id,
        )
        snap2 = backend.jobs.wait(job2, timeout=30)
        assert snap2.status is JobStatus.COMPLETED, snap2.error_message
        artifact_b = snap2.result
        assert artifact_b is not None and artifact_b.artifact_id
        assert artifact_b.parent_artifact_id == artifact_a.artifact_id, (
            f"预期血缘 {artifact_a.artifact_id}，实际 {artifact_b.parent_artifact_id!r}")

        # 列表读取同样保留血缘
        by_id = {a.artifact_id: a for a in backend.projects.list_artifacts(summary.project_id, "L01")}
        assert by_id[artifact_b.artifact_id].parent_artifact_id == artifact_a.artifact_id
    finally:
        backend.shutdown()


def test_project_api_processing_artifact_persists_output_axes(tmp_path: Path) -> None:
    """回归：P1-1 成果持久化输出轴——time_cut 改变时窗/零点后，读回成果物理轴按新时窗重建。

    此前 manifest 只存 input_dataset，加载恒用原始时窗 → 二次处理/成像按错误物理轴计算。
    keep_range [250, 500) ns：72 采样截成 36，时窗 500→250ns，零点偏移 250ns。
    """
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "axes", name="Axes")
        backend.projects.save_line_dataset(
            summary.project_id, "L01", _data(72, 48), name="L01",
            length_m=23.5, time_window_ns=500.0,
        )
        pipeline = PipelineDefinition(
            name="cut", steps=(PipelineStep("time_cut", {
                "mode": "keep_range", "time_start_ns": 250.0, "time_end_ns": 500.0,
            }),),
        )
        job = backend.submit_project_pipeline(
            summary.project_id, "L01", pipeline, result_name="cutA")
        snap = backend.jobs.wait(job, timeout=30)
        assert snap.status is JobStatus.COMPLETED, snap.error_message
        artifact = snap.result
        assert artifact is not None and artifact.artifact_id

        info = backend.projects.get_artifact_dataset_info(
            summary.project_id, "L01", artifact.artifact_id)
        assert tuple(info.shape) == (36, 48)  # 采样数减半

        dataset = backend.projects.read_artifact_dataset(
            summary.project_id, "L01", artifact.artifact_id)
        header = dataset.header_info
        time_axis = header.get("time_axis_ns")
        time_axis = (np.asarray(time_axis, dtype=np.float32)
                     if time_axis is not None else np.array([], dtype=np.float32))
        assert time_axis.size == 36, f"时间轴长度应为 36，实际 {time_axis.size}"
        assert float(header.get("time_window_ns") or 0.0) == pytest.approx(250.0, abs=1.0)
        assert float(time_axis[0]) == pytest.approx(250.0, abs=1.0)   # 零点偏移
        assert float(time_axis[-1]) == pytest.approx(500.0, abs=1.0)  # 终点=新时窗终点
    finally:
        backend.shutdown()
