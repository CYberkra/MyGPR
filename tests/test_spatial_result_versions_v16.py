from __future__ import annotations

import json
import csv
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
            source_result_id=f"{line_id}_raw",
            source_mode="raw",
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
        assert record.options["surface_generated"] is False
        assert record.options["surface_control_points_generated"] is True
        assert any('不生成连续曲面' in warning for warning in record.summary['preflight_warnings'])
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


def test_spatial_preflight_rejects_unknown_physical_axes(tmp_path):
    store = _project_with_spatial_lines(tmp_path / 'project', count=1)
    try:
        dataset = store.load_gpr_dataset('L01')
        dataset.metadata.update(time_axis_status='unknown', distance_axis_status='unknown')
        store.save_gpr_dataset('L01', dataset)
        result = SpatialResultVersionService(store).preflight(line_ids=['L01'])
        assert not result['passed']
        assert any('距离未知' in warning for warning in result['warnings'])
    finally:
        store.close()


def test_spatial_preflight_rejects_draft_and_partial_selection(tmp_path):
    store = _project_with_spatial_lines(tmp_path / 'project', count=2)
    try:
        annotation = store.load_basal_interface_annotation('L02')
        annotation.status = 'draft'
        store.save_basal_interface_annotation('L02', annotation)
        service = SpatialResultVersionService(store)
        result = service.preflight(line_ids=['L01', 'L02'])
        assert not result['passed']
        assert result['usable_line_ids'] == ['L01']
        with pytest.raises(ValueError, match='L02.*草稿'):
            service.create_result(line_ids=['L01', 'L02'], generate_surface=False)
        assert not service.list_results()
        assert service.create_result(line_ids=['L01'], generate_surface=False).summary['line_count'] == 1
    finally:
        store.close()


def test_spatial_result_becomes_stale_after_time_calibration(tmp_path):
    store = _project_with_spatial_lines(tmp_path / 'project', count=1)
    try:
        service = SpatialResultVersionService(store)
        result = service.create_result(line_ids=['L01'], generate_surface=False)
        assert not service.load_result(result.result_id).stale
        line = store.get_line('L01')
        line.time_window_override_ns = 100.0
        store.upsert_line(line)
        assert service.load_result(result.result_id).stale
        newer = service.create_result(line_ids=['L01'], generate_surface=False)
        assert not service.load_result(newer.result_id).stale
    finally:
        store.close()


def test_permanent_spatial_delete_preserves_line_inputs_and_selects_remaining(tmp_path):
    store = _project_with_spatial_lines(tmp_path / 'project', count=1)
    try:
        service = SpatialResultVersionService(store)
        first = service.create_result(name='surface', generate_surface=False)
        second = service.create_result(name='surface', generate_surface=False)
        before = np.asarray(store.load_gpr_dataset('L01').matrix).copy()
        source = store.root / 'spatial' / 'results' / second.result_id
        assert service.current_result_id() == second.result_id
        service.delete_permanently(second.result_id)
        assert not source.exists()
        assert service.current_result_id() == first.result_id
        assert not (store.root / 'spatial' / 'recycle_bin').exists()
        np.testing.assert_array_equal(np.asarray(store.load_gpr_dataset('L01').matrix), before)
        service.delete_permanently(first.result_id)
        assert service.current_result_id() == ''
    finally:
        store.close()


def test_ignored_traces_do_not_become_surface_control_points(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / 'project')
    try:
        annotation = store.load_basal_interface_annotation('L01')
        annotation.set_segment(10, 20, 'ignore')
        store.save_basal_interface_annotation('L01', annotation)
        result = SpatialResultVersionService(store).create_result(name='ignore', generate_surface=True)
        path = store.root / 'spatial' / 'results' / result.result_id / 'surface_control_points.geojson'
        features = json.loads(path.read_text(encoding='utf-8'))['features']
        assert not any(f['properties']['line_id'] == 'L01' and 10 <= f['properties']['trace_index'] <= 20 for f in features)
    finally:
        store.close()


def test_interface_export_uses_ground_elevation(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / 'project', count=1)
    try:
        trajectory = store.load_trajectory('L01')
        for point in trajectory.points:
            point.flight_height_m = 2.0
        store.save_trajectory('L01', trajectory)
        with store.export_spatial_interface_curve('L01').open(encoding='utf-8-sig', newline='') as fh:
            row = next(csv.DictReader(fh))
        assert float(row['surface_z']) == pytest.approx(123.0)
        assert float(row['interface_z']) == pytest.approx(123.0 - float(row['depth_m']), abs=1e-5)
    finally:
        store.close()


def test_annotation_rejects_invalid_processed_source_id(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / 'project', count=1)
    try:
        annotation = store.load_basal_interface_annotation('L01')
        annotation.source_result_id = '../outside'
        annotation.source_mode = 'processed'
        with pytest.raises(ValueError, match='来源'):
            store.save_basal_interface_annotation('L01', annotation)
        annotation.source_result_id = ''
        with pytest.raises(ValueError, match='来源'):
            store.save_basal_interface_annotation('L01', annotation)
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


def test_spatial_result_uses_actual_track_crs_for_generic_project(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / 'project', count=1)
    try:
        store.manifest.coordinate_system = 'CGCS2000 / 3-degree GK'
        store.save_manifest()
        record = SpatialResultVersionService(store).create_result(name='actual_crs', generate_surface=False)
        assert record.coordinate_system == 'EPSG:4547'
        path = store.root / 'spatial' / 'results' / record.result_id / 'interfaces.geojson'
        assert json.loads(path.read_text(encoding='utf-8'))['crs']['properties']['name'] == 'EPSG:4547'
    finally:
        store.close()


def test_spatial_preflight_rejects_mixed_track_crs(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / 'project', count=2)
    try:
        trajectory = store.load_trajectory('L02')
        for point in trajectory.points:
            point.coordinate_system = 'EPSG:4523'
        store.save_trajectory('L02', trajectory)
        result = SpatialResultVersionService(store).preflight(generate_surface=False)
        assert not result['passed']
        assert any('坐标系' in message for message in result['errors'])
    finally:
        store.close()


def test_spatial_preflight_recognizes_equivalent_projection_labels(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / 'project', count=1)
    try:
        trajectory = store.load_trajectory('L01')
        for point in trajectory.points:
            point.coordinate_system = 'CGCS2000 / 3-degree GK CM 114E (EPSG:4547)'
        store.save_trajectory('L01', trajectory)
        result = SpatialResultVersionService(store).preflight(generate_surface=False)
        assert result['passed'], result['errors']
        assert result['coordinate_system'] == 'EPSG:4547'
    finally:
        store.close()


@pytest.mark.parametrize('coordinate_system', ['EPSG:4547', 'EPSG:32648'])
def test_spatial_preflight_accepts_synchronized_wkt_and_epsg_tracks(
    tmp_path: Path, coordinate_system: str,
) -> None:
    from pyproj import CRS

    store = _project_with_spatial_lines(tmp_path / 'project', count=2)
    try:
        store.manifest.coordinate_system = coordinate_system
        store.save_manifest()
        for line_id, label in (
            ('L01', coordinate_system),
            ('L02', CRS.from_user_input(coordinate_system).to_wkt()),
        ):
            trajectory = store.load_trajectory(line_id)
            for point in trajectory.points:
                point.coordinate_system = label
            store.save_trajectory(line_id, trajectory)
        result = SpatialResultVersionService(store).preflight(generate_surface=False)
        assert result['passed'], result['errors']
        assert result['coordinate_system'] == coordinate_system
    finally:
        store.close()


def test_spatial_result_does_not_invent_unknown_track_crs(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / 'project', count=1)
    try:
        store.manifest.coordinate_system = 'CGCS2000 / 3-degree GK'
        store.save_manifest()
        trajectory = store.load_trajectory('L01')
        for point in trajectory.points:
            point.coordinate_system = ''
        store.save_trajectory('L01', trajectory)
        record = SpatialResultVersionService(store).create_result(name='unknown_crs', generate_surface=False)
        assert record.coordinate_system == ''
        assert record.status == 'generated_with_warnings'
        path = store.root / 'spatial' / 'results' / record.result_id / 'interfaces.geojson'
        assert 'crs' not in json.loads(path.read_text(encoding='utf-8'))
        with pytest.raises(ValueError, match='坐标系'):
            SpatialResultVersionService(store).export_result(record.result_id, tmp_path / 'unknown', format_name='kml')
    finally:
        store.close()


def test_kml_export_converts_projected_coordinates_to_lon_lat(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET
    from pyproj import Transformer

    store = _project_with_spatial_lines(tmp_path / 'project', count=1)
    try:
        service = SpatialResultVersionService(store)
        record = service.create_result(name='kml_crs', generate_surface=False)
        path = store.root / 'spatial' / 'results' / record.result_id / 'interfaces.geojson'
        feature = json.loads(path.read_text(encoding='utf-8'))['features'][0]
        x, y, z = feature['geometry']['coordinates'][0]
        expected_lon, expected_lat = Transformer.from_crs('EPSG:4547', 'EPSG:4326', always_xy=True).transform(x, y)
        output = service.export_result(record.result_id, tmp_path / 'result', format_name='kml')
        element = ET.parse(output).find('.//{http://www.opengis.net/kml/2.2}coordinates')
        lon, lat, altitude = map(float, element.text.split()[0].split(','))
        assert lon == pytest.approx(expected_lon)
        assert lat == pytest.approx(expected_lat)
        assert altitude == pytest.approx(z)
    finally:
        store.close()


def test_spatial_provenance_distinguishes_annotation_source_from_latest_result(tmp_path: Path) -> None:
    store = _project_with_spatial_lines(tmp_path / 'project', count=1)
    try:
        annotation = store.load_basal_interface_annotation('L01')
        annotation.source_result_id = 'L01_raw'
        annotation.source_mode = 'raw'
        store.save_basal_interface_annotation('L01', annotation, export_labels=True)
        line = store.get_line('L01')
        line.processed_result = 'processing/unrelated_latest.npz'
        (store.root / line.processed_result).parent.mkdir(parents=True, exist_ok=True)
        (store.root / line.processed_result).write_bytes(b'unrelated result')
        store.upsert_line(line)
        record = SpatialResultVersionService(store).create_result(name='raw_source')
        result_dir = store.root / 'spatial' / 'results' / record.result_id
        manifest = json.loads((result_dir / 'manifest.json').read_text(encoding='utf-8'))
        source = manifest['sources']['line_sources']['L01']
        assert source['source_result_id'] == 'L01_raw'
        assert source['processing_result_role'] == 'latest_line_result_context'
        with (result_dir / 'profiles.csv').open(encoding='utf-8-sig', newline='') as fh:
            row = next(csv.DictReader(fh))
        assert row['source_result_id'] == 'L01_raw'
        assert row['latest_processing_result'] == 'processing/unrelated_latest.npz'
    finally:
        store.close()
