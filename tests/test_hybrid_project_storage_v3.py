from __future__ import annotations

import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from core.field_project_backup import backup_project_archive
from core.field_project_models import FieldLineRecord
from core.field_project_operations import delete_project_line
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.processing_artifact_index import index_processing_artifacts
from core.project_integrity import ProjectIntegrityAuditor
from core.project_storage_backend import HYBRID_STORAGE_BACKEND
from core.storage_uri import is_h5_uri
from core.hdf5_line_container import list_processing_artifact_ids


def _hybrid_project(root: Path) -> FieldProjectStore:
    store = FieldProjectStore.create_empty(root, name="hybrid-v3")
    store.upsert_line(FieldLineRecord("L01", "Line 01"))
    dataset = GPRDataSet.from_matrix(
        "L01",
        np.arange(96 * 80, dtype=np.float32).reshape(96, 80),
        length_m=39.5,
        time_window_ns=600.0,
    )
    store.save_gpr_dataset("L01", dataset)
    return store


def test_new_project_uses_catalog_and_per_line_hdf5(tmp_path: Path) -> None:
    store = _hybrid_project(tmp_path / "project")
    try:
        assert store.manifest.storage_backend == HYBRID_STORAGE_BACKEND
        assert store.manifest.schema == "mygpr.field_project.v3"
        assert (store.root / "catalog.sqlite").is_file()
        line_h5 = store.root / "data" / "lines" / "L01.h5"
        assert line_h5.is_file()
        with h5py.File(line_h5, "r") as handle:
            dataset = handle["/raw/bscan"]
            assert dataset.shape == (96, 80)
            assert dataset.dtype == np.dtype("float32")
            assert dataset.chunks is not None
            assert dataset.compression == "gzip"
            assert not bool(handle["/raw"].attrs["immutable"])
            assert handle["/raw"].attrs["write_policy"] == "controlled_replace_with_backup"
        loaded = store.load_gpr_dataset("L01")
        assert type(loaded.matrix).__name__ == "HDF5ArrayProxy"
        window, _, _ = loaded.preview_window(
            sample_start=10, sample_end=20, trace_start=5, trace_end=15
        )
        assert window.shape == (10, 10)
    finally:
        store.close()


def test_processing_branch_lineage_and_catalog_manifest(tmp_path: Path) -> None:
    store = _hybrid_project(tmp_path / "project")
    try:
        raw = store.load_gpr_dataset("L01")
        first, _ = store.save_processed_line(
            "L01", np.asarray(raw.matrix) * 2,
            {"method": "gain", "params": {"factor": 2}, "branch_id": "L01:main"},
        )
        second, _ = store.save_processed_line(
            "L01", np.asarray(raw.matrix) * 3,
            {"method": "gain", "params": {"factor": 3}, "branch_id": "L01:main"},
        )
        records = sorted(index_processing_artifacts(store.root, "L01"), key=lambda row: row.created_at)
        assert [row.artifact_id for row in records] == [first.stem, second.stem]
        assert records[1].parent_artifact_id == records[0].artifact_id
        assert records[1].branch_id == "L01:main"
        assert is_h5_uri(records[1].data_path)
        branch = store.list_processing_branches("L01")[0]
        assert branch["head_artifact_id"] == second.stem
        catalog_row = store.storage.catalog.get_artifact(second.stem)
        assert catalog_row is not None
        assert catalog_row["manifest"]["manifest_sha256"]
        assert catalog_row["manifest"]["params_sha256"]
    finally:
        store.close()


def test_named_branch_is_seeded_from_selected_artifact(tmp_path: Path) -> None:
    store = _hybrid_project(tmp_path / "project")
    try:
        raw = store.load_gpr_dataset("L01")
        first, _ = store.save_processed_line(
            "L01", np.asarray(raw.matrix), {"method": "noop", "params": {}, "branch_id": "L01:main"}
        )
        branch = store.create_processing_branch(
            "L01", "方案 B", from_artifact_id=first.stem, parent_branch_id="L01:main"
        )
        saved, _ = store.save_processed_line(
            "L01", np.asarray(raw.matrix) + 1,
            {"method": "offset", "params": {"value": 1}, "branch_id": branch["branch_id"]},
        )
        record = next(row for row in index_processing_artifacts(store.root, "L01") if row.artifact_id == saved.stem)
        assert record.parent_artifact_id == first.stem
        assert record.branch_id == branch["branch_id"]
    finally:
        store.close()


def test_export_catalog_and_backup_snapshot_are_consistent(tmp_path: Path) -> None:
    store = _hybrid_project(tmp_path / "project")
    try:
        export = store.root / "exports" / "result.csv"
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text("x,y\n1,2\n", encoding="utf-8")
        registered = store.register_project_export(export, export_kind="test_csv")
        assert registered is not None
        rows = store.list_project_exports(export_kind="test_csv")
        assert rows and rows[0]["sha256"] == registered["sha256"]

        raw = store.load_gpr_dataset("L01")
        store.save_processed_line("L01", np.asarray(raw.matrix), {"method": "noop", "params": {}})

        result = backup_project_archive(store, tmp_path / "backups")
        assert result.verified
        with zipfile.ZipFile(result.archive_path) as archive:
            names = set(archive.namelist())
        missing = {"catalog.sqlite", "data/lines/L01.h5"} - names
        sidecars = {name for name in names if name.startswith("data/lines/L01.artifacts/") and name.endswith(".h5")}
        assert sidecars, f"backup snapshot flaked: no sidecar, names={sorted(names)}"
        assert not missing, (
            f"backup snapshot flaked: missing={sorted(missing)}, names={sorted(names)}"
        )
        assert "catalog.sqlite-wal" not in names
        assert "catalog.sqlite-shm" not in names
    finally:
        store.close()


def test_line_delete_moves_hdf5_and_cascades_catalog(tmp_path: Path) -> None:
    store = _hybrid_project(tmp_path / "project")
    try:
        raw = store.load_gpr_dataset("L01")
        store.save_processed_line("L01", np.asarray(raw.matrix), {"method": "noop", "params": {}})
        line_h5 = store.storage.line_container_path("L01")
        sidecar_dir = store.storage.line_artifacts_dir("L01")
        sidecars = list(sidecar_dir.glob("*.h5"))
        assert sidecars, "save_processed_line 未产生 sidecar 文件"
        result = delete_project_line(store, "L01")
        assert result.remaining_line_count == 0
        assert not line_h5.exists()
        assert not sidecar_dir.exists()
        assert store.storage.catalog.list_lines() == []
        assert store.storage.catalog.list_artifacts(line_id="L01") == []
        assert any((store.root / ".trash" / "lines").rglob("L01.h5"))
        trashed = list((store.root / ".trash" / "lines").rglob("*.h5"))
        assert len(trashed) >= 2, f"sidecar 未随容器入回收站: {trashed}"
    finally:
        store.close()


def test_catalog_failure_rolls_back_hdf5_processing_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _hybrid_project(tmp_path / "project")
    try:
        raw = store.load_gpr_dataset("L01")

        def fail_registration(_payload):
            raise RuntimeError("simulated catalog failure")

        monkeypatch.setattr(store.storage.catalog, "register_artifact", fail_registration)
        with pytest.raises(RuntimeError, match="simulated catalog failure"):
            store.save_processed_line(
                "L01", np.asarray(raw.matrix), {"method": "noop", "params": {}}
            )
        assert list_processing_artifact_ids(store.storage.line_container_path("L01")) == []
        assert list((store.root / "processed" / "L01").glob("*.artifact")) == []
    finally:
        store.close()


def test_integrity_reports_unindexed_hdf5_processing_group(tmp_path: Path) -> None:
    store = _hybrid_project(tmp_path / "project")
    try:
        raw = store.load_gpr_dataset("L01")
        saved, _ = store.save_processed_line(
            "L01", np.asarray(raw.matrix), {"method": "noop", "params": {}}
        )
        artifact_id = saved.stem
        with store.storage.catalog.transaction() as db:
            db.execute(
                "UPDATE processing_branches SET head_artifact_id='' WHERE head_artifact_id=?",
                (artifact_id,),
            )
            db.execute("DELETE FROM artifacts WHERE artifact_id=?", (artifact_id,))
        report = ProjectIntegrityAuditor(store).audit(persist=False)
        issue = next(
            item for item in report.issues
            if item.code == "storage.unindexed_hdf5_artifact"
        )
        assert issue.object_id == artifact_id
        assert issue.severity == "warning"
    finally:
        store.close()


def test_final_catalog_manifest_failure_restores_branch_and_removes_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _hybrid_project(tmp_path / "project")
    try:
        raw = store.load_gpr_dataset("L01")
        original_register = store.storage.catalog.register_artifact
        call_count = 0

        def fail_second_registration(payload):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated final catalog failure")
            return original_register(payload)

        monkeypatch.setattr(store.storage.catalog, "register_artifact", fail_second_registration)
        with pytest.raises(RuntimeError, match="simulated final catalog failure"):
            store.save_processed_line(
                "L01", np.asarray(raw.matrix), {"method": "noop", "params": {}}
            )
        assert list_processing_artifact_ids(store.storage.line_container_path("L01")) == []
        assert store.storage.catalog.list_artifacts(line_id="L01") == []
        branch = store.storage.catalog.list_branches(line_id="L01")[0]
        assert branch["head_artifact_id"] == ""
    finally:
        store.close()


def test_controlled_raw_replacement_preserves_processing_lineage(tmp_path: Path) -> None:
    store = _hybrid_project(tmp_path / "project")
    try:
        raw = store.load_gpr_dataset("L01")
        saved, _ = store.save_processed_line(
            "L01", np.asarray(raw.matrix) * 2, {"method": "gain", "params": {"factor": 2}}
        )
        artifact_id = saved.stem
        replacement = GPRDataSet.from_matrix(
            "L01", np.full((96, 80), 7, dtype=np.float32), length_m=39.5, time_window_ns=600.0
        )
        store.save_gpr_dataset("L01", replacement)
        loaded = store.load_gpr_dataset("L01")
        assert float(np.asarray(loaded.matrix[0:1, 0:1])[0, 0]) == 7.0
        assert artifact_id in list_processing_artifact_ids(store.storage.line_container_path("L01"))
        assert store.storage.catalog.get_artifact(artifact_id) is not None
        policy = store.manifest.storage_policy
        assert policy["immutable_source_files"] is True
        assert policy["normalized_raw_write_policy"] == "controlled_replace_with_backup"
    finally:
        store.close()
