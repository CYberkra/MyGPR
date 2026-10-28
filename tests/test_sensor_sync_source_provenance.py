"""Source archival regressions; numerical stubs establish no scientific validity."""
import json
from dataclasses import asdict
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pytest

from mygpr.interfaces.backend import MyGPRBackend
from mygpr.application.jobs.cancellation import CancellationTokenSource
from mygpr.application.jobs.context import ExecutionContext
from core.job_manager import JobCancelled


@pytest.fixture
def sync_project(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    project = backend.projects.create_project(tmp_path / "project", name="Source audit", coordinate_system="EPSG:32648")
    backend.projects.save_line_dataset(project.project_id, "L01", np.zeros((8, 3), dtype=np.float32))
    timestamps = tmp_path / "timestamps.csv"
    timestamps.write_text("timestamp_s\n0\n1\n2\n", encoding="utf-8")
    rtk = tmp_path / "rtk" / "sensor.csv"
    imu = tmp_path / "imu" / "sensor.csv"
    rtk.parent.mkdir()
    imu.parent.mkdir()
    rtk.write_text("timestamp_s,longitude,latitude,roll_deg,pitch_deg,yaw_deg\n0,104,30,0,0,90\n1,104.00001,30,0,0,90\n2,104.00002,30,0,0,90\n", encoding="utf-8")
    imu.write_text("timestamp_s,roll_deg,pitch_deg,yaw_deg\n0,0,0,90\n1,0,0,90\n2,0,0,90\n", encoding="utf-8")
    try:
        yield backend, project, {"rtk_path": rtk, "imu_path": imu, "trace_timestamps_path": timestamps}
    finally:
        backend.shutdown()


def _snapshot(sensors):
    return {path.relative_to(sensors).as_posix(): path.read_bytes() for path in sensors.rglob("*") if path.is_file()}


@pytest.mark.parametrize("same_source", [False, True])
def test_sensor_sources_keep_distinct_role_references_and_original_bytes(sync_project, same_source):
    backend, project, paths = sync_project
    if same_source:
        paths["imu_path"] = paths["rtk_path"]
    backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    sensors = Path(project.root_path) / "raw" / "L01" / "sensors"
    manifest = json.loads((sensors / "L01_sensor_sync_manifest.json").read_text(encoding="utf-8"))
    sources = manifest["source_files"]
    assert len(set(sources.values())) == len(paths)
    for role, relative in sources.items():
        assert (sensors / relative).read_bytes() == paths[f"{role}_path"].read_bytes()


def test_failed_source_copy_keeps_previous_valid_sync_manifest_and_sources(sync_project):
    backend, project, paths = sync_project
    backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    sensors = Path(project.root_path) / "raw" / "L01" / "sensors"
    previous = _snapshot(sensors)
    original_copy = shutil.copy2

    def fail_imu_copy(source, destination):
        if Path(source) == paths["imu_path"]:
            raise OSError("injected source-copy failure")
        return original_copy(source, destination)

    with patch("core.sensor_sync_service.shutil.copy2", side_effect=fail_imu_copy):
        with pytest.raises(OSError, match="injected source-copy failure"):
            backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    assert _snapshot(sensors) == previous
    assert not list(sensors.parent.glob(".sensor_sync_staging_*"))


def test_cancellation_during_source_copy_keeps_previous_valid_sync(sync_project):
    backend, project, paths = sync_project
    backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    sensors = Path(project.root_path) / "raw" / "L01" / "sensors"
    previous = _snapshot(sensors)
    cancellation = CancellationTokenSource()
    context = ExecutionContext(cancellation_token=cancellation.token)
    original_copy = shutil.copy2

    def cancel_during_copy(source, destination):
        result = original_copy(source, destination)
        cancellation.cancel()
        return result

    with patch("core.sensor_sync_service.shutil.copy2", side_effect=cancel_during_copy):
        with pytest.raises(JobCancelled):
            backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths, context=context)
    assert _snapshot(sensors) == previous
    assert not list(sensors.parent.glob(".sensor_sync_staging_*"))


@pytest.mark.parametrize("failure", ["cancel_after_swap", "metadata_write"])
def test_failure_after_directory_swap_keeps_last_valid_sync(sync_project, failure):
    backend, project, paths = sync_project
    backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    sensors = Path(project.root_path) / "raw" / "L01" / "sensors"
    previous = _snapshot(sensors)
    paths["rtk_path"].write_text(paths["rtk_path"].read_text(encoding="utf-8").replace("104", "105"), encoding="utf-8")
    store = backend.projects._session(project.project_id)._store
    if failure == "metadata_write":
        with patch.object(store, "upsert_line", side_effect=OSError("injected metadata write failure")):
            with pytest.raises(OSError, match="injected metadata write failure"):
                backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    else:
        cancellation = CancellationTokenSource()
        context = ExecutionContext(cancellation_token=cancellation.token)
        original_replace = Path.replace

        def cancel_after_swap(path, destination):
            result = original_replace(path, destination)
            if Path(destination) == sensors and path.name.startswith(".sensor_sync_staging_"):
                cancellation.cancel()
            return result

        with patch("core.sensor_sync_service.Path.replace", autospec=True, side_effect=cancel_after_swap):
            with pytest.raises(JobCancelled):
                backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths, context=context)
    assert _snapshot(sensors) == previous


@pytest.mark.parametrize("failure", ["save_manifest", "catalog"])
def test_one_time_metadata_write_failure_restores_manifest_catalog_and_reopened_project(sync_project, failure):
    backend, project, paths = sync_project
    backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    store = backend.projects._session(project.project_id)._store
    sensors = Path(project.root_path) / "raw" / "L01" / "sensors"
    previous = _snapshot(sensors)
    previous_line = asdict(store.get_line("L01"))
    previous_manifest = (store.root / "project.json").read_bytes()
    previous_catalog = store.storage.catalog.list_lines()
    paths["rtk_path"].write_text(paths["rtk_path"].read_text(encoding="utf-8").replace("104", "105"), encoding="utf-8")
    target = store if failure == "save_manifest" else store.storage.catalog
    name = "save_manifest" if failure == "save_manifest" else "upsert_line"
    original = getattr(target, name)
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected one-time metadata failure")
        return original(*args, **kwargs)

    with patch.object(target, name, side_effect=fail_once):
        with pytest.raises(OSError, match="injected one-time metadata failure"):
            backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    assert _snapshot(sensors) == previous
    assert (store.root / "project.json").read_bytes() == previous_manifest
    assert asdict(store.get_line("L01")) == previous_line
    assert store.storage.catalog.list_lines() == previous_catalog
    backend.projects.close_project(project.project_id)
    reopened = backend.projects.open_project(project.root_path)
    reopened_store = backend.projects._session(reopened.project_id)._store
    assert asdict(reopened_store.get_line("L01")) == previous_line
    assert _snapshot(sensors) == previous


def test_persistent_catalog_failure_preserves_recovery_versions_and_reports_incomplete_rollback(sync_project):
    backend, project, paths = sync_project
    backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    store = backend.projects._session(project.project_id)._store
    sensors = Path(project.root_path) / "raw" / "L01" / "sensors"
    previous = _snapshot(sensors)
    paths["rtk_path"].write_text(paths["rtk_path"].read_text(encoding="utf-8").replace("104", "105"), encoding="utf-8")
    with patch.object(store.storage.catalog, "upsert_line", side_effect=OSError("persistent catalog failure")):
        with pytest.raises(RuntimeError, match="需要恢复；previous=.*；failed=") as error:
            backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    assert isinstance(error.value.__cause__, OSError)
    assert "persistent catalog failure" in str(error.value)
    assert _snapshot(sensors) == previous
    assert any(_snapshot(path) == previous for path in sensors.parent.glob("sensors_previous_*"))
    failed = list(sensors.parent.glob("sensors_failed_*"))
    assert failed and any(_snapshot(path) != previous for path in failed)


def test_local_commit_ignores_cancel_arriving_after_metadata_commit(sync_project):
    backend, project, paths = sync_project
    store = backend.projects._session(project.project_id)._store
    cancellation = CancellationTokenSource()
    context = ExecutionContext(cancellation_token=cancellation.token)
    original = store.upsert_line

    def cancel_after_metadata(*args, **kwargs):
        result = original(*args, **kwargs)
        cancellation.cancel()
        return result

    with patch.object(store, "upsert_line", side_effect=cancel_after_metadata):
        result = backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths, context=context)
    assert context.is_cancelled()
    assert (Path(project.root_path) / result.manifest_path).is_file()


def test_directory_rollback_failure_preserves_previous_and_failed_source_versions(sync_project):
    backend, project, paths = sync_project
    backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    store = backend.projects._session(project.project_id)._store
    sensors = Path(project.root_path) / "raw" / "L01" / "sensors"
    previous = _snapshot(sensors)
    paths["rtk_path"].write_text(paths["rtk_path"].read_text(encoding="utf-8").replace("104", "105"), encoding="utf-8")
    with patch.object(store, "upsert_line", side_effect=OSError("metadata failed")), patch(
        "core.sensor_sync_service.shutil.copytree", side_effect=OSError("rollback directory failed")
    ):
        with pytest.raises(RuntimeError, match="需要恢复.*恢复上次同步目录失败") as error:
            backend.projects.synchronize_line_sensors(project.project_id, "L01", **paths)
    assert isinstance(error.value.__cause__, OSError)
    assert any(_snapshot(path) == previous for path in sensors.parent.glob("sensors_previous_*"))
    assert any(_snapshot(path) != previous for path in sensors.parent.glob("sensors_failed_*"))
