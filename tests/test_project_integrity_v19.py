from __future__ import annotations

import json
import os
from pathlib import Path

from core.basal_interface_annotations import BasalInterfaceAnnotation
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.manual_processing_chain import ManualProcessingSession
from core.project_integrity import ProjectIntegrityAuditor
from core.project_runtime_guard import ProjectRuntimeGuard
from core.workspace_context import WorkspaceContextStore
from core.storage_uri import resolve_h5_uri


def _project_with_processing_and_annotation(root: Path):
    store = FieldProjectStore.create_empty(root, name="integrity-v19")
    dataset = GPRDataSet.synthetic_basal_interface("L01", rows=64, cols=96, length_m=48.0)
    store.save_gpr_dataset("L01", dataset)
    session = ManualProcessingSession(dataset)
    session.append_step("dewow", {"window": 9})
    payload = session.build_save_payload("dewow", {"window": 9})
    data_path, _params_path = store.save_processed_line("L01", session.current_dataset.matrix, payload)
    annotation = BasalInterfaceAnnotation(
        "L01",
        dataset.trace_count,
        dataset.sample_count,
        source_result_id=data_path.stem,
        source_mode="processed",
        status="confirmed",
    )
    annotation.set_keypoint(0, 30.0)
    annotation.set_keypoint(dataset.trace_count - 1, 35.0)
    store.save_basal_interface_annotation("L01", annotation, export_labels=False)
    return store, data_path.stem


def test_integrity_audit_repairs_only_workspace_pointers_and_staging(tmp_path: Path) -> None:
    store, artifact_id = _project_with_processing_and_annotation(tmp_path / "project")
    try:
        context = WorkspaceContextStore(store.root)
        payload = context.load()
        payload.update(
            {
                "selected_line_id": "L99",
                "processing_source_by_line": {"L01": "missing-processing", "L99": "ghost"},
                "annotation_by_line": {
                    "L01": {"version": "old", "status": "draft", "source_result_id": "missing-processing"},
                    "L99": {"version": "ghost", "status": "draft", "source_result_id": "ghost"},
                },
                "selected_spatial_result_id": "basal_surface_v999",
                "selected_report_id": "report_v999",
            }
        )
        context.save(payload)
        orphan = store.root / "spatial" / "results" / ".orphan.staging"
        orphan.mkdir(parents=True)
        (orphan / "partial.txt").write_text("partial", encoding="utf-8")
        os.utime(orphan, (1, 1))

        report = ProjectIntegrityAuditor(store).audit(
            repair_context=True,
            clean_staging=True,
            staging_min_age_s=0,
        )
        repaired = context.load()
        assert repaired["selected_line_id"] == "L01"
        assert repaired["processing_source_by_line"] == {"L01": artifact_id}
        assert repaired["annotation_by_line"]["L01"]["source_result_id"] == artifact_id
        assert "L99" not in repaired["annotation_by_line"]
        assert repaired["selected_spatial_result_id"] == ""
        assert repaired["selected_report_id"] == ""
        assert not orphan.exists()
        assert report.repairs
        assert (store.root / "metadata" / "project_integrity.json").exists()
    finally:
        store.close()


def test_integrity_audit_detects_missing_processing_data_without_mutating_annotation(tmp_path: Path) -> None:
    store, artifact_id = _project_with_processing_and_annotation(tmp_path / "project")
    try:
        annotation_path = store.interface_annotation_path("L01")
        before = annotation_path.read_bytes()
        records = __import__("core.processing_artifact_index", fromlist=["index_processing_artifacts"]).index_processing_artifacts(store.root)
        assert records and records[0].artifact_id == artifact_id
        h5_path, dataset_path = resolve_h5_uri(store.root, records[0].data_path)
        import h5py
        with h5py.File(h5_path, "r+") as handle:
            del handle[dataset_path]

        report = ProjectIntegrityAuditor(store).audit(repair_context=True)
        codes = {issue.code for issue in report.issues}
        assert "processing.missing_registered_dataset" in codes
        assert annotation_path.read_bytes() == before
        assert report.error_count >= 1
    finally:
        store.close()


def test_runtime_guard_records_unclean_and_clean_sessions(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    first = ProjectRuntimeGuard(root)
    first.mark_open(active_workspace="processing_lab", line_id="L01")

    second = ProjectRuntimeGuard(root)
    assert second.previous_unclean() is True
    payload = second.mark_open(active_workspace="interpretation", line_id="L01")
    assert payload["previous_unclean"] is True
    second.checkpoint(active_workspace="spatial", line_id="L01")
    second.mark_clean_close(active_workspace="delivery", line_id="L01")

    saved = json.loads((root / "metadata" / "runtime_session.json").read_text(encoding="utf-8"))
    assert saved["state"] == "closed"
    assert saved["active_workspace"] == "delivery"
    assert ProjectRuntimeGuard(root).previous_unclean() is False


def test_chunked_copy_cancellation_removes_partial_destination(tmp_path: Path) -> None:
    from core.chunked_gpr_io import ImportCancelled, copy_file_chunked

    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * (3 * 1024 * 1024))
    destination = tmp_path / "destination.bin"
    calls = {"count": 0}

    def cancel() -> bool:
        calls["count"] += 1
        return calls["count"] >= 3

    try:
        copy_file_chunked(source, destination, cancel_requested=cancel, chunk_bytes=256 * 1024)
    except ImportCancelled:
        pass
    else:
        raise AssertionError("copy should have been cancelled")
    assert not destination.exists()
