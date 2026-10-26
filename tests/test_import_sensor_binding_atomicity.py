"""A replacement waveform must not inherit another acquisition's navigation."""
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from core.field_project_store import FieldProjectStore
from core.sensor_sync_service import synchronize_project_line_sensors
from mygpr.infrastructure.persistence.field_project_adapter import _load_trace_metadata


@pytest.fixture
def synchronized_line(tmp_path):
    store = FieldProjectStore.create_empty(
        tmp_path / "project", name="sensor binding", coordinate_system="EPSG:32648",
    )
    source = tmp_path / "original.npy"
    np.save(source, np.ones((8, 3), dtype=np.float32))
    store.import_line_file("L01", source)
    timestamps = tmp_path / "timestamps.csv"
    timestamps.write_text("timestamp_s\n0\n1\n2\n", encoding="utf-8")
    rtk = tmp_path / "rtk.csv"
    rtk.write_text(
        "timestamp_s,longitude,latitude,altitude_m,flight_height_m\n"
        "0,104,30,100,5\n1,104.00001,30,100,5\n2,104.00002,30,100,5\n",
        encoding="utf-8",
    )
    imu = tmp_path / "imu.csv"
    imu.write_text(
        "timestamp_s,roll_deg,pitch_deg,yaw_deg\n"
        "0,0,0,90\n1,0,0,90\n2,0,0,90\n",
        encoding="utf-8",
    )
    synchronize_project_line_sensors(
        store, line_id="L01", rtk_path=rtk, imu_path=imu,
        trace_timestamps_path=timestamps,
    )
    return store


def _replacement(tmp_path, kind):
    if kind == "npy":
        source = tmp_path / "replacement.npy"
        np.save(source, np.full((8, 3), 7, dtype=np.float32))
    else:
        source = tmp_path / "replacement.h5"
        with h5py.File(source, "w") as handle:
            handle.attrs.update(samples_per_trace=8, time_window_ns=100., store_component="real")
            handle.create_dataset("traces", data=np.full((3, 8), 7, dtype=np.float32))
            handle.create_dataset("trace_meta/lon", data=[105., 105.1, 105.2])
            handle.create_dataset("trace_meta/lat", data=[31., 31.1, 31.2])
            handle.create_dataset("trace_meta/height", data=[10., 11., 12.])
    return source


@pytest.mark.parametrize("kind", ["npy", "hdf5"])
def test_successful_replacement_discards_previous_sensor_binding(synchronized_line, tmp_path, kind):
    store = synchronized_line
    store.import_line_file("L01", _replacement(tmp_path, kind))
    reopened = FieldProjectStore.open(store.root)
    line = reopened.get_line("L01")
    assert not line.trajectory_path
    assert not line.sensor_sync_manifest_path
    assert not line.trace_metadata_path
    assert line.rtk_status == "未定位"
    assert line.sensor_sync_status == "未同步"
    with pytest.raises(FileNotFoundError):
        reopened.load_trajectory("L01")
    metadata = _load_trace_metadata(reopened, "L01")
    if kind == "npy":
        assert metadata == {}
    else:
        np.testing.assert_allclose(metadata["longitude"], [105., 105.1, 105.2])
        np.testing.assert_allclose(metadata["flight_height_m"], [10., 11., 12.])
    np.testing.assert_array_equal(reopened.load_gpr_dataset("L01").matrix, np.full((8, 3), 7))


def test_replacement_discards_old_csv_trajectory_fallback(tmp_path):
    store = FieldProjectStore.create_empty(tmp_path / "project", name="csv binding")
    store.import_line_file("L01", Path("sample_data/gui_sidecar_all_data_main.csv"))
    assert store.trajectory_path("L01").exists()
    store.import_line_file("L01", _replacement(tmp_path, "npy"))
    reopened = FieldProjectStore.open(store.root)
    with pytest.raises(FileNotFoundError):
        reopened.load_trajectory("L01")
    assert not reopened.get_line("L01").trajectory_path


def test_import_source_named_like_default_trajectory_is_preserved(tmp_path):
    source = tmp_path / "L01_trajectory.csv"
    matrix = np.arange(32 * 16, dtype=np.float32).reshape(32, 16)
    np.savetxt(source, matrix, delimiter=",")
    source_bytes = source.read_bytes()
    with FieldProjectStore.create_empty(tmp_path / "project", name="source filename") as store:
        store.import_line_file("L01", source)
        np.testing.assert_array_equal(store.load_gpr_dataset("L01").matrix, matrix)
        assert (store.root / store.get_line("L01").raw_path).read_bytes() == source_bytes


def test_airborne_source_name_collision_keeps_original_and_generated_trajectory(tmp_path):
    source = tmp_path / "L01_trajectory.csv"
    source.write_bytes(Path("sample_data/gui_sidecar_all_data_main.csv").read_bytes())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    root = tmp_path / "project"
    with FieldProjectStore.create_empty(root, name="airborne source collision") as store:
        store.import_line_file("L01", source)
        line = store.get_line("L01")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        assert hashlib.sha256((root / line.raw_path).read_bytes()).hexdigest() == source_hash
        assert line.raw_path == "raw/L01/sources/L01_trajectory.csv"
        assert line.trajectory_path != line.raw_path
    with FieldProjectStore.open(root) as reopened:
        assert reopened.load_gpr_dataset("L01").matrix.shape == (10, 12)
        assert len(reopened.load_trajectory("L01").points) == 12


def test_failed_replacement_restores_sensor_binding_and_original_data(synchronized_line, tmp_path, monkeypatch):
    store = synchronized_line
    before = store.get_line("L01")
    metadata = _load_trace_metadata(store, "L01")
    sensors = store.root / "raw" / "L01" / "sensors"
    files = {path.relative_to(sensors): path.read_bytes() for path in sensors.rglob("*") if path.is_file()}

    def fail_manifest(*args):
        raise OSError("injected import manifest failure")

    monkeypatch.setattr(store, "_write_import_manifest", fail_manifest)
    with pytest.raises(OSError, match="injected import manifest failure"):
        store.import_line_file("L01", _replacement(tmp_path, "hdf5"))
    reopened = FieldProjectStore.open(store.root)
    assert reopened.get_line("L01") == before
    for key, values in metadata.items():
        np.testing.assert_array_equal(_load_trace_metadata(reopened, "L01")[key], values)
    assert {path.relative_to(sensors): path.read_bytes() for path in sensors.rglob("*") if path.is_file()} == files
    np.testing.assert_array_equal(reopened.load_gpr_dataset("L01").matrix, np.ones((8, 3)))
