from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.ingest_service import IngestService
from core.processing_session import ProcessingSessionService
from core.qc_service import QcService


def _formal_project(tmp_path: Path):
    source = tmp_path / "line.csv"
    raw = np.arange(120, dtype=np.float32).reshape(20, 6)
    np.savetxt(source, raw, delimiter=",")
    temporary = IngestService.open_temporary(source)
    formal = IngestService.formalize(temporary, tmp_path / "formal", name="Formal")
    temporary.close()
    _acknowledge_all_warnings(formal, formal.list_lines()[0].line_id)
    return formal, raw


def _formal_project_pending_integrity(tmp_path: Path):
    source = tmp_path / "line_pending.csv"
    raw = np.arange(120, dtype=np.float32).reshape(20, 6)
    np.savetxt(source, raw, delimiter=",")
    temporary = IngestService.open_temporary(source)
    formal = IngestService.formalize(
        temporary,
        tmp_path / "formal_pending",
        name="Pending",
        verify_hashes=False,
    )
    temporary.close()
    return formal, raw


def _acknowledge_all_warnings(project, line_id: str) -> None:
    qc = QcService(project)
    report = qc.run_line_qc(line_id)
    for item in report.items:
        if item.severity == "warning" and not item.acknowledged:
            qc.acknowledge_warning(line_id, item.code, "测试夹具确认该警告不阻断处理")


def test_processing_session_refuses_temporary_preview_projects(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(120, dtype=np.float32).reshape(20, 6), delimiter=",")
    project = IngestService.open_temporary(source)
    try:
        line_id = project.list_lines()[0].line_id
        with pytest.raises(PermissionError, match="临时项目"):
            ProcessingSessionService.open_line(project, line_id)
    finally:
        project.close()


def test_processing_session_refuses_unacknowledged_qc_warnings(
    tmp_path: Path,
) -> None:
    project, _raw = _formal_project_pending_integrity(tmp_path)
    try:
        line_id = project.list_lines()[0].line_id
        with pytest.raises(PermissionError, match="未确认质控警告"):
            ProcessingSessionService.open_line(project, line_id)

        _acknowledge_all_warnings(project, line_id)

        session = ProcessingSessionService.open_line(project, line_id)
        assert np.array_equal(session.original_data, _raw)
    finally:
        project.close()


def test_processing_session_refuses_tampered_verified_raw(
    tmp_path: Path,
) -> None:
    project, _raw = _formal_project(tmp_path)
    try:
        line = project.list_lines()[0]
        copied = project.resolve_relative_path(line.raw_files[0].path)
        copied.chmod(0o666)
        copied.write_text("tampered", encoding="utf-8")

        with pytest.raises(PermissionError, match="阻断质控错误"):
            ProcessingSessionService.open_line(project, line.line_id)
    finally:
        project.close()


def test_processing_session_preview_does_not_modify_formal_state(tmp_path: Path) -> None:
    project, raw = _formal_project(tmp_path)
    try:
        line_id = project.list_lines()[0].line_id
        session = ProcessingSessionService.open_line(project, line_id)
        preview = session.preview_method("amplitude_scale", {"scale": 2.0})

        assert np.allclose(preview.data, raw * 2.0)
        assert np.array_equal(session.current_data, raw)
        assert np.array_equal(session.original_data, raw)
        assert session.steps == []
    finally:
        project.close()


def test_processing_session_apply_undo_redo_reset_and_save_version(tmp_path: Path) -> None:
    project, raw = _formal_project(tmp_path)
    try:
        line_id = project.list_lines()[0].line_id
        session = ProcessingSessionService.open_line(project, line_id)

        session.apply_method("amplitude_scale", {"scale": 2.0})
        assert np.allclose(session.current_data, raw * 2.0)
        assert [step.method_id for step in session.steps] == ["amplitude_scale"]

        assert session.undo() is True
        assert np.array_equal(session.current_data, raw)
        assert session.can_redo is True

        assert session.redo() is True
        assert np.allclose(session.current_data, raw * 2.0)

        result = session.save_version("Scaled")
        loaded = project.load_processing_result(result.result_id, line_id=line_id)
        assert np.allclose(loaded["data"], raw * 2.0)
        assert loaded["record"].processing_chain[0]["method_id"] == "amplitude_scale"

        session.reset()
        assert np.array_equal(session.current_data, raw)
        assert session.steps == []
    finally:
        project.close()


def test_processing_session_pipeline_injects_runtime_context_and_preserves_chain(
    tmp_path: Path,
) -> None:
    project, raw = _formal_project(tmp_path)
    try:
        line_id = project.list_lines()[0].line_id
        session = ProcessingSessionService.open_line(project, line_id)
        session.apply_pipeline(
            [
                {"method_id": "amplitude_scale", "params": {"scale": 2.0}},
                {"method_id": "compensatingGain", "params": {"gain_min": 6.0, "gain_max": 6.0}},
            ]
        )

        assert np.allclose(session.current_data, raw * 2.0 * (10.0 ** (6.0 / 20.0)))
        assert [step.method_id for step in session.steps] == [
            "amplitude_scale",
            "compensatingGain",
        ]
        assert len(session.compare_snapshots()) >= 3
    finally:
        project.close()


def test_processing_session_autotune_recommends_without_applying(tmp_path: Path) -> None:
    project, raw = _formal_project(tmp_path)
    try:
        line_id = project.list_lines()[0].line_id
        session = ProcessingSessionService.open_line(project, line_id)
        recommendation = session.recommend_method(
            "compensatingGain",
            candidate_params=[
                {"gain_min": 0.0, "gain_max": 0.0},
                {"gain_min": 1.0, "gain_max": 3.0},
                {"gain_min": 1.0, "gain_max": 6.0},
            ],
            search_mode="fast",
        )

        assert recommendation["method_key"] == "compensatingGain"
        assert recommendation["recommended_params"]
        assert np.array_equal(session.current_data, raw)
        assert session.steps == []
    finally:
        project.close()


def test_processing_session_runs_and_exports_manual_auto_comparison(
    tmp_path: Path,
) -> None:
    project, raw = _formal_project(tmp_path)
    try:
        line_id = project.list_lines()[0].line_id
        session = ProcessingSessionService.open_line(project, line_id)

        comparison = session.run_manual_auto_comparison(
            pipeline=["dewow"],
            manual_params_by_method={"dewow": {"window": 1}},
            search_mode="fast",
        )

        assert comparison.manual.source == "current_ui_params"
        assert comparison.manual.pipeline == ["dewow"]
        assert comparison.automatic.pipeline == ["dewow"]
        assert comparison.automatic.auto_tune_results
        assert np.array_equal(session.current_data, raw)

        bundle = session.export_last_manual_auto_comparison(bundle_name="session_cmp")
        output_dir = Path(bundle["output_dir"])
        assert output_dir.is_relative_to(project.root)
        assert (output_dir / "comparison_report.md").exists()
        assert (output_dir / "evidence_bundle.zip").exists()
    finally:
        project.close()


def test_processing_session_can_edit_and_replay_chain_from_original(tmp_path: Path) -> None:
    project, raw = _formal_project(tmp_path)
    try:
        line_id = project.list_lines()[0].line_id
        session = ProcessingSessionService.open_line(project, line_id)
        session.apply_pipeline(
            [
                {"method_id": "amplitude_scale", "params": {"scale": 2.0}},
                {"method_id": "amplitude_scale", "params": {"scale": 3.0}},
            ]
        )
        assert np.allclose(session.current_data, raw * 6.0)

        session.move_step(1, 0)
        assert [step.params["scale"] for step in session.steps] == [3.0, 2.0]
        assert np.allclose(session.current_data, raw * 6.0)

        session.set_step_enabled(0, False)
        assert np.allclose(session.current_data, raw * 2.0)

        session.update_step_params(1, {"scale": 4.0})
        assert np.allclose(session.current_data, raw * 4.0)

        removed = session.remove_step(0)
        assert removed.enabled is False
        assert len(session.steps) == 1
        assert np.allclose(session.current_data, raw * 4.0)
    finally:
        project.close()
