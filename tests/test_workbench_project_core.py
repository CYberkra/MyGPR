from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.command_contract import assert_command_allowed
from core.ingest_service import IngestService
from core.project_service import ProjectService
from core.qc_service import QcService
from core.workspace_session import WorkspaceSessionService


def _write_matrix_csv(path: Path) -> None:
    np.savetxt(path, np.arange(24, dtype=np.float32).reshape(6, 4), delimiter=",")


def test_formal_project_uses_versioned_readable_relative_paths(tmp_path: Path) -> None:
    service = ProjectService.create(tmp_path / "field_project", name="Field Project")
    try:
        manifest = service.manifest
        assert manifest.schema == "mygpr.project.v1"
        assert manifest.temporary is False
        assert (service.root / "project.mygpr.json").exists()
        payload = json.loads((service.root / "project.mygpr.json").read_text(encoding="utf-8"))
        assert payload["name"] == "Field Project"
        assert service.resolve_relative_path("lines/L001/line.json") == (
            service.root / "lines" / "L001" / "line.json"
        )
        with pytest.raises(ValueError):
            service.resolve_relative_path("../outside.json")
    finally:
        service.close()


def test_temporary_ingest_can_be_formalized_and_detects_raw_tampering(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    _write_matrix_csv(source)

    temporary = IngestService.open_temporary(source)
    assert temporary.manifest.temporary is True
    assert temporary.list_lines()[0].raw_files[0].path == str(source.resolve())

    formal = IngestService.formalize(temporary, tmp_path / "formal", name="Formal")
    try:
        line = formal.list_lines()[0]
        raw_ref = line.raw_files[0]
        assert not Path(raw_ref.path).is_absolute()
        copied = formal.resolve_relative_path(raw_ref.path)
        assert copied.exists()
        assert raw_ref.integrity_status == "verified"
        assert raw_ref.sha256

        copied.chmod(0o666)
        copied.write_text("changed", encoding="utf-8")
        report = QcService(formal).run_line_qc(line.line_id)
        assert report.can_process is False
        assert any(item.code == "raw_integrity_mismatch" for item in report.items)
    finally:
        formal.close()
        temporary.close()


def test_qc_warnings_do_not_block_but_errors_do(tmp_path: Path) -> None:
    source = tmp_path / "line.csv"
    _write_matrix_csv(source)
    temporary = IngestService.open_temporary(source)
    try:
        line = temporary.list_lines()[0]
        report = QcService(temporary).run_line_qc(line.line_id)
        assert report.can_process is True
        assert any(item.severity == "warning" for item in report.items)

        source.unlink()
        missing_report = QcService(temporary).run_line_qc(line.line_id)
        assert missing_report.can_process is False
        assert any(item.severity == "error" for item in missing_report.items)
    finally:
        temporary.close()


def test_workspace_session_keeps_project_documents_separate_from_global_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = tmp_path / "settings"
    monkeypatch.setenv("LOCALAPPDATA", str(settings))
    project = ProjectService.create(tmp_path / "project", name="Project")
    try:
        sessions = WorkspaceSessionService(project)
        sessions.save_global_layout("data_management", {"left": 280, "right": 320})
        sessions.save_project_session(
            open_documents=["line:L001", "qc:L001"],
            selected_line_id="L001",
            active_workspace="data_management",
        )
        assert sessions.load_global_layout("data_management") == {"left": 280, "right": 320}
        restored = sessions.load_project_session()
        assert restored["open_documents"] == ["line:L001", "qc:L001"]
        assert restored["selected_line_id"] == "L001"
    finally:
        project.close()


def test_command_contract_is_capability_based() -> None:
    assert_command_allowed("formal_ready", "processing")
    assert_command_allowed("temporary_preview", "display")
    with pytest.raises(PermissionError):
        assert_command_allowed("temporary_preview", "processing")
    with pytest.raises(PermissionError):
        assert_command_allowed("qc_review_required", "processing")
    with pytest.raises(PermissionError):
        assert_command_allowed("qc_blocked", "processing")


def test_ingest_discovers_legacy_manifest_sidecars_and_qc_parses_them(tmp_path: Path) -> None:
    source = tmp_path / "main.csv"
    _write_matrix_csv(source)
    (tmp_path / "rtk_custom.csv").write_text(
        "timestamp_s,longitude,latitude\n0,104,30\n1,104.1,30.1\n",
        encoding="utf-8",
    )
    (tmp_path / "imu_custom.csv").write_text(
        "timestamp_s,roll_deg,pitch_deg,yaw_deg\n0,0,0,0\n1,1,1,1\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "primary_data_file": "main.csv",
                "sidecars": {"rtk": "rtk_custom.csv", "imu": "imu_custom.csv"},
            }
        ),
        encoding="utf-8",
    )

    project = IngestService.open_temporary(source)
    try:
        line = project.list_lines()[0]
        assert set(line.sidecars) >= {"rtk", "imu"}
        report = QcService(project).run_line_qc(line.line_id)
        assert any(item.code == "rtk_valid" for item in report.items)
        assert any(item.code == "imu_valid" for item in report.items)
    finally:
        project.close()


def test_invalid_discovered_sidecar_is_a_warning_not_a_blocker(tmp_path: Path) -> None:
    source = tmp_path / "main.csv"
    _write_matrix_csv(source)
    (tmp_path / "rtk.csv").write_text("wrong,columns\n1,2\n", encoding="utf-8")
    project = IngestService.open_temporary(source)
    try:
        line = project.list_lines()[0]
        report = QcService(project).run_line_qc(line.line_id)
        assert report.can_process is True
        assert any(item.code == "rtk_invalid" and item.severity == "warning" for item in report.items)
    finally:
        project.close()


def test_qc_warning_can_record_manual_acknowledgement(tmp_path: Path) -> None:
    source = tmp_path / "main.csv"
    _write_matrix_csv(source)
    project = IngestService.open_temporary(source)
    try:
        line = project.list_lines()[0]
        qc = QcService(project)
        report = qc.run_line_qc(line.line_id)
        warning = next(item for item in report.items if item.severity == "warning")
        updated = qc.acknowledge_warning(line.line_id, warning.code, "现场确认可继续")
        acknowledged = next(item for item in updated.items if item.code == warning.code)
        assert acknowledged.acknowledged is True
        assert acknowledged.acknowledgement_note == "现场确认可继续"
        rerun = qc.run_line_qc(line.line_id)
        rerun_acknowledged = next(item for item in rerun.items if item.code == warning.code)
        assert rerun_acknowledged.acknowledged is True
        assert rerun_acknowledged.acknowledgement_note == "现场确认可继续"
    finally:
        project.close()


def test_formal_project_sidecar_assignment_copies_into_project(tmp_path: Path) -> None:
    source = tmp_path / "main.csv"
    _write_matrix_csv(source)
    sidecar = tmp_path / "manual_rtk.csv"
    sidecar.write_text(
        "timestamp_s,longitude,latitude\n0,104,30\n1,104.1,30.1\n",
        encoding="utf-8",
    )
    temporary = IngestService.open_temporary(source)
    project = IngestService.formalize(temporary, tmp_path / "formal_sidecar", name="Formal")
    temporary.close()
    try:
        line_id = project.list_lines()[0].line_id
        updated = IngestService.assign_sidecar(project, line_id, "rtk", sidecar)
        stored = Path(updated.sidecars["rtk"])
        assert stored.is_absolute() is False
        assert project.resolve_relative_path(stored).exists()
        assert project.resolve_relative_path(stored).is_relative_to(project.root)
        sidecar_ref = next(ref for ref in updated.raw_files if ref.role == "sidecar:rtk")
        assert sidecar_ref.integrity_status == "pending"
        IngestService.verify_project_integrity(project, [line_id])
        verified = project.get_line(line_id)
        verified_sidecar = next(ref for ref in verified.raw_files if ref.role == "sidecar:rtk")
        assert verified_sidecar.sha256

        sidecar_copy = project.resolve_relative_path(verified_sidecar.path)
        sidecar_copy.chmod(0o666)
        sidecar_copy.write_text("tampered", encoding="utf-8")
        report = QcService(project).run_line_qc(line_id)
        assert report.can_process is False
        assert any(
            item.code == "raw_integrity_mismatch"
            and item.evidence.get("role") == "sidecar:rtk"
            for item in report.items
        )
    finally:
        project.close()


def test_processing_result_persists_numpy_metadata_outside_json(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "result_project", name="Results")
    try:
        result = project.save_processing_result(
            "L001",
            np.ones((4, 3), dtype=np.float32),
            name="Processed",
            processing_chain=[{"method": "dewow"}],
            header_info={"samples": np.int64(4), "large_axis": np.arange(4)},
            trace_metadata={"trace_distance_m": np.arange(3, dtype=np.float32)},
        )
        result_dir = project.root / "results" / "L001" / result.result_id
        payload = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
        assert payload["header_info"]["samples"] == 4
        assert payload["header_info"]["large_axis"]["kind"] == "ndarray_summary"
        assert payload["trace_metadata_path"].endswith("trace_metadata.npz")
        assert project.resolve_relative_path(payload["trace_metadata_path"]).exists()
    finally:
        project.close()


def test_formal_copy_can_remain_pending_until_background_integrity_verification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "main.csv"
    _write_matrix_csv(source)
    temporary = IngestService.open_temporary(source)
    project = IngestService.formalize(
        temporary,
        tmp_path / "pending_integrity",
        name="Pending",
        verify_hashes=False,
    )
    temporary.close()
    try:
        pending = project.list_lines()[0]
        assert pending.raw_files[0].integrity_status == "pending"
        assert pending.raw_files[0].sha256 is None
        verified_ids = IngestService.verify_project_integrity(project, [pending.line_id])
        verified = project.get_line(pending.line_id)
        assert verified_ids == [pending.line_id]
        assert verified.raw_files[0].integrity_status == "verified"
        assert verified.raw_files[0].sha256
    finally:
        project.close()


def test_integrity_verification_never_rebaselines_tampered_verified_raw(
    tmp_path: Path,
) -> None:
    source = tmp_path / "main.csv"
    _write_matrix_csv(source)
    temporary = IngestService.open_temporary(source)
    project = IngestService.formalize(temporary, tmp_path / "formal_tamper", name="Formal")
    temporary.close()
    try:
        line = project.list_lines()[0]
        expected_hash = line.raw_files[0].sha256
        copied = project.resolve_relative_path(line.raw_files[0].path)
        copied.chmod(0o666)
        copied.write_text("tampered", encoding="utf-8")

        IngestService.verify_project_integrity(project, [line.line_id])
        updated = project.get_line(line.line_id)
        assert updated.raw_files[0].sha256 == expected_hash
        assert updated.raw_files[0].integrity_status == "mismatch"
        assert QcService(project).run_line_qc(line.line_id).can_process is False
    finally:
        project.close()


def test_project_lock_allows_only_one_writer(tmp_path: Path) -> None:
    first = ProjectService.create(tmp_path / "locked_project", name="Locked")
    try:
        with pytest.raises(RuntimeError, match="already open"):
            ProjectService.open(first.root)
    finally:
        first.close()
    reopened = ProjectService.open(tmp_path / "locked_project")
    reopened.close()


def test_project_service_can_recover_a_stale_writer_lock(tmp_path: Path) -> None:
    root = tmp_path / "recoverable_project"
    project = ProjectService.create(root, name="Recoverable")
    project.close()
    (root / ".mygpr.lock").write_text(
        '{"pid": 99999999, "opened_at": "2026-06-08T00:00:00Z"}',
        encoding="utf-8",
    )

    recovered = ProjectService.recover(root)
    try:
        assert recovered.manifest.name == "Recoverable"
        assert (root / ".mygpr.lock").exists()
    finally:
        recovered.close()


def test_project_service_never_recovers_an_active_writer_lock(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "active_project", name="Active")
    try:
        with pytest.raises(RuntimeError, match="already open"):
            ProjectService.recover(project.root)
    finally:
        project.close()


def test_project_service_refuses_to_overwrite_existing_project(tmp_path: Path) -> None:
    root = tmp_path / "existing_project"
    project = ProjectService.create(root, name="Existing")
    project.close()
    with pytest.raises(FileExistsError):
        ProjectService.create(root, name="Replacement")
