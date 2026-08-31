from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mygpr.domain.acquisition import (
    MotionCompensationProfile,
    SensorKind,
    SensorSyncSettings,
)
from mygpr.interfaces.backend import MyGPRBackend
from mygpr.application.jobs.models import JobStatus
from tests.qt_import_isolation import assert_qt_imports_unchanged, qt_module_snapshot


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sensor_files(root: Path, trace_count: int = 4) -> tuple[Path, Path, Path]:
    timestamps = np.arange(trace_count, dtype=np.float64) + 100.0
    trace_path = _write_csv(
        root / "trace_times.csv",
        [{"trace_timestamp_s": value} for value in timestamps],
    )
    rtk_path = _write_csv(
        root / "rtk.csv",
        [
            {
                "timestamp_s": value,
                "longitude": 104.0 + index * 0.00001,
                "latitude": 30.0,
                "local_x_m": float(index),
                "local_y_m": 0.0,
                "local_z_m": 100.0,
                "flight_height_m": 2.0,
                "rtk_fix_type": 4,
            }
            for index, value in enumerate(timestamps)
        ],
    )
    imu_path = _write_csv(
        root / "imu.csv",
        [
            {
                "timestamp_s": value,
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
                "yaw_deg": 90.0,
            }
            for value in timestamps
        ],
    )
    return trace_path, rtk_path, imu_path


def test_headless_acquisition_preflight_import_and_sensor_sync(tmp_path: Path) -> None:
    qt_before = qt_module_snapshot()
    source = tmp_path / "line.npy"
    np.save(source, np.arange(48 * 4, dtype=np.float32).reshape(48, 4))
    trace_path, rtk_path, imu_path = _sensor_files(tmp_path)

    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(
            tmp_path / "project",
            name="Acquisition Project",
            coordinate_system="EPSG:32648",
        )
        preflight = backend.acquisition.preflight(source, line_id="L01")
        assert preflight.can_import
        assert preflight.shape == (48, 4)

        imported = backend.acquisition.import_line(
            summary.project_id,
            source,
            line_id="L01",
            name="Line 01",
        )
        assert imported.shape == (48, 4)
        assert backend.projects.get_dataset_info(summary.project_id, "L01").shape == (48, 4)

        synced = backend.acquisition.synchronize_project_line(
            summary.project_id,
            "L01",
            rtk_path=rtk_path,
            trace_timestamps_path=trace_path,
            imu_path=imu_path,
            settings=SensorSyncSettings(project_crs="EPSG:32648"),
        )
        root = Path(summary.root_path)
        assert synced.diagnostics["rtk"]["coverage_ratio"] == 1.0
        assert (root / synced.manifest_path).is_file()
        assert (root / synced.trajectory_path).is_file()
        assert (root / synced.trace_metadata_path).is_file()
        loaded = backend.projects.read_dataset(summary.project_id, "L01")
        assert loaded.trace_metadata["alignment_status"].tolist() == ["aligned"] * 4
        assert np.allclose(loaded.trace_metadata["flight_height_m"], 2.0)
        motion_job = backend.submit_project_pipeline(
            summary.project_id,
            "L01",
            backend.acquisition.motion_pipeline(),
            result_name="motion-v2",
        )
        motion = backend.jobs.wait(motion_job, timeout=30)
        assert motion.status is JobStatus.COMPLETED, motion.error_message
        assert motion.result.method_id == "motion_compensation_v2"
        assert_qt_imports_unchanged(qt_before)
    finally:
        backend.shutdown()


def test_sidecar_parser_and_array_sync_return_domain_contracts(tmp_path: Path) -> None:
    _, rtk_path, imu_path = _sensor_files(tmp_path, trace_count=3)
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        rtk = backend.acquisition.parse_sidecar(rtk_path, kind=SensorKind.RTK)
        imu = backend.acquisition.parse_sidecar(imu_path, kind="imu")
        assert rtk.sample_count == 3
        assert imu.sample_count == 3
        result = backend.acquisition.synchronize_streams(
            trace_timestamps_s=np.array([100.0, 101.0, 102.0]),
            rtk=rtk,
            imu=imu,
            line_id="L02",
            trace_distance_hint_m=np.array([0.0, 1.0, 2.0]),
        )
        assert result.diagnostics["rtk"]["coverage_ratio"] == 1.0
        assert result.trace_metadata["alignment_status"].tolist() == ["aligned"] * 3
        assert len(result.trajectory) == 3
    finally:
        backend.shutdown()


def test_acquisition_jobs_and_motion_pipeline_contract(tmp_path: Path) -> None:
    source = tmp_path / "line.npy"
    np.save(source, np.ones((40, 4), dtype=np.float32))
    trace_path, rtk_path, imu_path = _sensor_files(tmp_path)
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "job-project", name="Jobs")
        import_job = backend.submit_line_import(
            project.project_id,
            str(source),
            line_id="L01",
            dielectric_constant=12.5,
        )
        imported = backend.jobs.wait(import_job, timeout=30)
        assert imported.status is JobStatus.COMPLETED, imported.error_message
        assert backend.projects.get_dataset_info(
            project.project_id, "L01"
        ).dielectric_constant == 12.5

        sync_job = backend.submit_sensor_sync(
            project.project_id,
            "L01",
            rtk_path=str(rtk_path),
            trace_timestamps_path=str(trace_path),
            imu_path=str(imu_path),
        )
        synced = backend.jobs.wait(sync_job, timeout=30)
        assert synced.status is JobStatus.COMPLETED, (
            f"sensor sync flaked: {synced.status}, stage={synced.message!r}, "
            f"errors={synced.error_details!r}"
        )

        integrated = backend.acquisition.motion_pipeline()
        assert [step.method_id for step in integrated.steps] == ["motion_compensation_v2"]
        atomic = backend.acquisition.motion_pipeline(
            MotionCompensationProfile(mode="atomic", include_vibration_cleanup=True)
        )
        assert [step.method_id for step in atomic.steps] == [
            "motion_compensation_speed",
            "motion_compensation_attitude",
            "motion_compensation_height",
            "motion_compensation_vibration",
        ]
    finally:
        backend.shutdown()
