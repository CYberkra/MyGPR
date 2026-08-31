from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.basal_interface_annotations import BasalInterfaceAnnotation
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.job_manager import JobCancelled
from core.spatial_result_versions import SPATIAL_RESULT_SCHEMA, SpatialResultVersionService
from core.trajectory_model import TrajectoryModel, TrajectoryPoint


def _project_with_spatial_lines(root: Path, count: int = 3) -> FieldProjectStore:
    store = FieldProjectStore.create_empty(root, name="Spatial V1.6")
    store.manifest.coordinate_system = "EPSG:4547"
    store.manifest.vertical_datum = "1985 国家高程基准"
    store.save_manifest()
    for line_index in range(count):
        line_id = f"L{line_index + 1:02d}"
        dataset = GPRDataSet.from_matrix(
            line_id,
            np.zeros((80, 120), dtype=np.float32),
            length_m=60.0,
        )
        store.save_gpr_dataset(line_id, dataset)
        annotation = BasalInterfaceAnnotation(
            line_id,
            trace_count=120,
            sample_count=80,
            status="confirmed",
            source_result_id=f"{line_id}_processed_v1",
            source_mode="processed",
        )
        annotation.set_keypoint(0, 42.0 + line_index)
        annotation.set_keypoint(119, 51.0 + line_index)
        annotation.set_segment(40, 54, "weak")
        store.save_basal_interface_annotation(line_id, annotation, export_labels=True)
        store.save_trajectory(
            line_id,
            TrajectoryModel(
                [
                    TrajectoryPoint(
                        distance_m=float(trace) * 0.5,
                        x=500000.0 + float(trace) * 0.5,
                        y=3400000.0 + line_index * 20.0,
                        z=125.0 + line_index,
                        coordinate_system="EPSG:4547",
                    )
                    for trace in range(120)
                ]
            ),
        )
    return store


def test_spatial_result_version_is_immutable_and_traceable(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / "project")
    try:
        service = SpatialResultVersionService(store)
        record = service.create_result(name="basal_surface", generate_surface=True)

        assert record.result_id == "basal_surface_v001"
        assert record.summary["line_count"] == 3
        assert record.options["surface_generated"] is True
        assert service.current_result_id() == record.result_id
        result_dir = store.root / "spatial" / "results" / record.result_id
        manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema"] == SPATIAL_RESULT_SCHEMA
        assert (result_dir / "profiles.csv").exists()
        assert (result_dir / "interfaces.geojson").exists()
        assert (result_dir / "trajectories.geojson").exists()
        assert (result_dir / "surface_control_points.geojson").exists()
        assert len(manifest["sources"]["line_sources"]) == 3

        second = service.create_result(name="basal_surface", generate_surface=True)
        assert second.result_id == "basal_surface_v002"
        assert (result_dir / "manifest.json").read_text(encoding="utf-8") == json.dumps(manifest, ensure_ascii=False, indent=2)
    finally:
        store.close()


def test_spatial_result_reports_stale_after_annotation_changes(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / "project")
    try:
        service = SpatialResultVersionService(store)
        record = service.create_result(name="basal_surface")
        assert service.load_result(record.result_id).stale is False

        annotation = store.load_basal_interface_annotation("L01")
        assert annotation is not None
        annotation.set_keypoint(60, 47.5)
        store.save_basal_interface_annotation("L01", annotation, export_labels=True)

        assert service.load_result(record.result_id).stale is True
    finally:
        store.close()


def test_spatial_result_exports_zip_geojson_csv_and_kml(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / "project")
    try:
        service = SpatialResultVersionService(store)
        record = service.create_result(name="basal_surface")
        zip_path = service.export_result(record.result_id, tmp_path / "delivery", format_name="zip")
        geojson_path = service.export_result(record.result_id, tmp_path / "interfaces", format_name="geojson")
        csv_path = service.export_result(record.result_id, tmp_path / "profiles", format_name="csv")
        kml_path = service.export_result(record.result_id, tmp_path / "interfaces", format_name="kml")
        assert zip_path.suffix == ".zip" and zip_path.exists()
        assert geojson_path.suffix == ".geojson" and geojson_path.exists()
        assert csv_path.suffix == ".csv" and csv_path.exists()
        assert kml_path.suffix == ".kml" and "<kml" in kml_path.read_text(encoding="utf-8")
    finally:
        store.close()


def test_spatial_result_cancellation_leaves_no_partial_version(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / "project")
    try:
        service = SpatialResultVersionService(store)
        with pytest.raises(JobCancelled):
            service.create_result(name="cancelled", cancel_requested=lambda: True)
        results_root = store.root / "spatial" / "results"
        assert not (results_root / "cancelled_v001").exists()
        assert not list(results_root.glob(".cancelled_v001.*.staging"))
    finally:
        store.close()


def test_surface_generation_is_disabled_when_coverage_is_insufficient(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / "project", count=2)
    try:
        service = SpatialResultVersionService(store)
        preflight = service.preflight(generate_surface=True)
        assert preflight["passed"] is True
        assert preflight["surface_allowed"] is False
        record = service.create_result(name="lines_only", generate_surface=True)
        assert record.options["surface_generated"] is False
        assert "surface_control_points_geojson" not in record.files
    finally:
        store.close()
