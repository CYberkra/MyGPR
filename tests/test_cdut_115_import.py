"""Completed1.1.15 acquisition records; source row numbers are not trace IDs."""

import hashlib
import json
import zlib

import h5py
import numpy as np
import pytest

from core.gpr_data_model import load_gpr_dataset
from core.acquisition_hdf5 import acquisition_trace_metadata

IDENTITY = "d54b6f93-9d38-4d61-bd08-10882ef54a53"
FLOATS = ("lon", "lat", "alt", "height", "roll", "pitch", "yaw", "ax", "ay", "az", "gx", "gy", "gz", "timestamp_ms")


def markers():
    return [
        dict(
            marker_id=i,
            trace_no=n,
            marker_type=kind,
            label=label,
            marked_at="2026-10-30T17:00:00.000+08:00",
            lon=104.1,
            lat=30.6,
            alt=500.0,
            height=None,
        )
        for i, n, kind, label in [
            (1, 1, "start", "开始标记1"),
            (2, 3, "general", "一般标记2"),
            (3, 5, "end", "结束标记3"),
        ]
    ]


def h5_source(path):
    data = (np.arange(60, dtype=np.float32).reshape(5, 12) - 20) / 8
    with h5py.File(path, "w") as f:
        f.attrs.update(
            samples_per_trace=12,
            time_window_ns=44.0,
            store_component="real",
            time_axis_endpoint=True,
            acquisition_id=IDENTITY,
            committed_trace_count=5,
            prefix="line7",
            markers_schema_version=1,
            trace_number_base=1,
        )
        f.create_dataset("traces", data=data)
        f.create_dataset("trace_meta/trace_id", data=np.array([0, 20, 40, 60, 80], np.uint64))
        group = f.create_group("markers")
        group.attrs.update(schema_version=1, trace_number_base=1)
        for name in markers()[0]:
            values = [m[name] for m in markers()]
            if name in ("marker_id", "trace_no"):
                group.create_dataset(name, data=values, dtype="int64")
            elif name in ("lon", "lat", "alt", "height"):
                group.create_dataset(name, data=[np.nan if v is None else v for v in values], dtype="float64")
            else:
                group.create_dataset(name, data=values, dtype=h5py.string_dtype("utf-8"))
    return data.T


def csv_source(folder, *, journal=True):
    path = folder / f"line7_{IDENTITY}_ground.csv"
    data = (np.arange(60, dtype=np.float32).reshape(5, 12) - 20) / 8
    with path.open("w", encoding="utf-8") as f:
        f.write("Number of Samples = 12\nTime windows (ns) = 44\nNumber of Traces = 5\nTrace interval (m) = 0\n")
        for trace in data:
            for value in trace:
                f.write(f"104.1,30.6,500,{value:.9g},2\n")
    if journal:
        binding = {"schema": 1, "data_file": path.name, "acquisition_id": IDENTITY, "committed_trace_count": 5}
        Path = type(path)
        Path(str(path) + ".session.json").write_text(json.dumps(binding), encoding="utf-8")
        rows = []
        for i, trace in enumerate(data):
            row = dict.fromkeys(FLOATS, None)
            row.update(
                row=i + 1,
                trace_id=i * 20,
                acquisition_id=IDENTITY,
                timestamp_ms=1700000000000.0 + i * 100,
                lon=104.123456789 + i * 1e-7,
                lat=30.612345678,
                alt=500.0,
                height=None,
                samples_crc32=zlib.crc32(np.asarray(trace, dtype="<f4").tobytes()) & 0xFFFFFFFF,
            )
            rows.append(row)
        Path(str(path) + ".meta.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "file_prefix": "line7",
        "data_file": path.name,
        "trace_number_base": 1,
        "markers": markers(),
    }
    (folder / "line7_markers.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path, data.T


def test_h5_embedded_markers_preserve_source_rows_gaps_and_waveform(tmp_path):
    path = tmp_path / "line7.h5"
    expected = h5_source(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    ds = load_gpr_dataset(path)
    assert ds.metadata["acquisition_markers"] == markers()
    np.testing.assert_array_equal(ds.matrix, expected)
    np.testing.assert_array_equal(acquisition_trace_metadata(ds.metadata)["trace_id"], [0, 20, 40, 60, 80])
    assert ds.time_axis_ns[-1] == 44.0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_bound_csv_journal_retains_original_ids_full_precision_and_unknown_height(tmp_path):
    path, expected = csv_source(tmp_path)
    ds = load_gpr_dataset(path)
    assert ds.format_name == "cdut-acquisition-csv"
    assert ds.metadata["acquisition_id"] == IDENTITY
    assert ds.metadata["acquisition_markers"] == markers()
    meta = acquisition_trace_metadata(ds.metadata)
    np.testing.assert_array_equal(meta["trace_id"], [0, 20, 40, 60, 80])
    assert meta["longitude"][0] == 104.123456789
    assert np.isnan(meta["flight_height_m"]).all()
    assert meta["trace_timestamp_s"][0] == 1700000000.0
    assert ds.metadata["trajectory_rows"][0]["elevation_role"] == "platform"
    assert ds.metadata["trajectory_rows"][0]["elevation"] == 500.0
    np.testing.assert_array_equal(ds.matrix, expected)
    assert ds.metadata["distance_axis_status"] == "unknown"


@pytest.mark.parametrize("fault", ["crc", "identity", "committed", "missing_row", "duplicate_row", "filename"])
def test_invalid_bound_csv_metadata_cannot_be_applied_to_a_waveform(tmp_path, fault):
    path, _ = csv_source(tmp_path)
    session = type(path)(str(path) + ".session.json")
    journal = type(path)(str(path) + ".meta.jsonl")
    binding = json.loads(session.read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in journal.read_text(encoding='utf-8').splitlines()]
    if fault == "crc":
        rows[0]["samples_crc32"] ^= 1
    elif fault == "identity":
        rows[0]["acquisition_id"] = "ad5c4e42-45f6-4973-9b7e-74e724e2a0c5"
    elif fault == "committed":
        binding["committed_trace_count"] = 4
    elif fault == "missing_row":
        rows.pop()
    elif fault == "duplicate_row":
        rows[1]["row"] = 1
    else:
        binding["data_file"] = "foreign.csv"
    session.write_text(json.dumps(binding), encoding='utf-8')
    journal.write_text("\n".join(json.dumps(r) for r in rows), encoding='utf-8')
    with pytest.raises(ValueError):
        load_gpr_dataset(path)


@pytest.mark.parametrize("fault", ["out_of_range", "duplicate_start", "version", "length", "link"])
def test_bad_h5_marker_schema_does_not_silently_create_wrong_annotations(tmp_path, fault):
    path = tmp_path / "line7.h5"
    h5_source(path)
    with h5py.File(path, "r+") as f:
        if fault == "out_of_range":
            f["markers/trace_no"][1] = 6
        elif fault == "duplicate_start":
            f["markers/marker_type"][1] = "start"
        elif fault == "version":
            f["markers"].attrs["schema_version"] = 2
        elif fault == "length":
            del f["markers/height"]
            f["markers"].create_dataset("height", data=[1.0])
        else:
            del f["markers/trace_no"]
            f["markers/trace_no"] = h5py.SoftLink("/trace_meta/trace_id")
    with pytest.raises(ValueError):
        load_gpr_dataset(path)


def test_legacy_json_without_marker_type_is_general_and_foreign_filename_is_ignored(tmp_path):
    path, _ = csv_source(tmp_path, journal=False)
    side = tmp_path / "line7_markers.json"
    payload = json.loads(side.read_text(encoding='utf-8'))
    payload["markers"] = payload["markers"][1:2]
    payload["markers"][0]["marker_id"] = 1
    payload["markers"][0].pop("marker_type")
    side.write_text(json.dumps(payload), encoding='utf-8')
    assert load_gpr_dataset(path).metadata["acquisition_markers"][0]["marker_type"] == "general"
    payload["data_file"] = "another.csv"
    side.write_text(json.dumps(payload), encoding='utf-8')
    assert not load_gpr_dataset(path).metadata.get("acquisition_markers")


def unwrap(metadata):
    while isinstance(metadata.get("metadata"), dict):
        metadata = metadata["metadata"]
    return metadata


def test_project_copy_reopen_and_replacement_keep_only_own_companions(tmp_path):
    from core.field_project_operations import import_line_data
    from core.field_project_store import FieldProjectStore
    from mygpr.infrastructure.persistence.field_project_adapter import _load_trace_metadata

    source = tmp_path / "source"
    source.mkdir()
    path, expected = csv_source(source)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    store = FieldProjectStore.create_empty(tmp_path / "project", name="controller1.1.15")
    import_line_data(store, path, line_id="L01")
    copied = store.root / "raw/L01"
    for name, digest in hashes.items():
        assert hashlib.sha256((copied / name).read_bytes()).hexdigest() == digest
    reopened = FieldProjectStore.open(store.root)
    ds = reopened.load_gpr_dataset("L01")
    np.testing.assert_array_equal(ds.matrix, expected)
    assert unwrap(ds.metadata)["acquisition_markers"] == markers()
    np.testing.assert_array_equal(_load_trace_metadata(reopened, "L01")["trace_id"], [0, 20, 40, 60, 80])
    trajectory = reopened.load_trajectory("L01")
    assert trajectory.points[0].z == 500.0
    assert np.isnan(trajectory.points[0].flight_height_m)
    assert trajectory.points[0].timestamp_s == 1700000000.0
    assert np.isnan(trajectory.points[0].roll_deg)
    # A marker-free replacement must not inherit the old acquisition's sidecar.
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    other, _ = csv_source(replacement, journal=False)
    (replacement / "line7_markers.json").unlink()
    import_line_data(reopened, other, line_id="L01")
    assert not unwrap(reopened.load_gpr_dataset("L01").metadata).get("acquisition_markers")
    for name in ("line7_markers.json", path.name + ".session.json", path.name + ".meta.jsonl"):
        assert not (copied / name).exists()
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}


def test_invalid_replacement_rolls_back_source_companions_and_normalized_dataset(tmp_path):
    from core.field_project_operations import import_line_data
    from core.field_project_store import FieldProjectStore

    source = tmp_path / "source"
    source.mkdir()
    path, expected = csv_source(source)
    store = FieldProjectStore.create_empty(tmp_path / "project", name="rollback")
    import_line_data(store, path, line_id="L01")
    copied = store.root / "raw/L01"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in copied.iterdir() if p.is_file()}
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    bad, _ = csv_source(replacement)
    journal = type(path)(str(bad) + ".meta.jsonl")
    journal.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        import_line_data(store, bad, line_id="L01")
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in copied.iterdir() if p.is_file()}
    np.testing.assert_array_equal(store.load_gpr_dataset("L01").matrix, expected)


def test_preflight_is_explicitly_not_full_csv_integrity_verification(tmp_path):
    from core.field_import_preview import build_import_preflight

    path, _ = csv_source(tmp_path)
    result = build_import_preflight(path)
    assert result.can_import and any("3 个采集标记" in s for s in result.suggestions)
    assert any("并非完整性通过" in s for s in result.suggestions)
    journal = type(path)(str(path) + ".meta.jsonl")
    journal.unlink()
    assert not build_import_preflight(path).can_import


def test_external_reference_replacement_does_not_leave_owned_old_markers(tmp_path):
    from core.field_project_store import FieldProjectStore
    from core.field_project_operations import import_line_data

    first = tmp_path / "first"
    first.mkdir()
    path, _ = csv_source(first)
    store = FieldProjectStore.create_empty(tmp_path / "project", name="reference replacement")
    import_line_data(store, path, line_id="L01")
    external = tmp_path / "external"
    external.mkdir()
    other, _ = csv_source(external)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in external.iterdir()}
    store.import_line_file("L01", other, copy_into_project=False)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in external.iterdir()}
    assert not (store.root / "raw/L01/line7_markers.json").exists()
    third = tmp_path / "third"
    third.mkdir()
    plain, _ = csv_source(third, journal=False)
    (third / "line7_markers.json").unlink()
    import_line_data(store, plain, line_id="L01")
    assert not unwrap(store.load_gpr_dataset("L01").metadata)["acquisition_markers"]


def test_all_missing_navigation_stays_unknown_not_a_zero_trajectory(tmp_path):
    path, _ = csv_source(tmp_path)
    journal = type(path)(str(path) + ".meta.jsonl")
    rows = [json.loads(row) for row in journal.read_text(encoding='utf-8').splitlines()]
    for row in rows:
        for name in FLOATS:
            row[name] = None
    journal.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    ds = load_gpr_dataset(path)
    assert not ds.metadata["trajectory_rows"]
    meta = acquisition_trace_metadata(ds.metadata)
    assert np.isnan(meta["longitude"]).all() and np.isnan(meta["trace_timestamp_s"]).all()


def test_zero_window_csv_stays_unknown_after_save_and_auto_windows_use_samples(tmp_path):
    from core.field_project_store import FieldProjectStore
    from core.field_project_operations import import_line_data
    from mygpr.domain.processing.models import ProcessingRequest
    from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor

    path, _ = csv_source(tmp_path)
    path.write_text(path.read_text(encoding='utf-8').replace("Time windows (ns) = 44", "Time windows (ns) = 0"), encoding="utf-8")
    store = FieldProjectStore.create_empty(tmp_path / "unknown", name="unknown time")
    import_line_data(store, path, line_id="L01")
    ds = FieldProjectStore.open(store.root).load_gpr_dataset("L01")
    assert ds.to_metadata()["time_axis_status"] == "unknown"
    from core.gui_rendering import bundle_from_dataset

    preview = bundle_from_dataset(ds)
    assert preview.sample_axis is None and preview.depth_axis_m is None
    out = NativeProcessingExecutor().execute(
        ProcessingRequest(data=ds.matrix, method_id="short_trace_repair", header_info=ds.to_metadata(), params={})
    )
    assert all("actual_ns" not in gate for gate in out.metadata["gate_diagnostics"])


def test_markers_and_trace_identity_survive_processing_export_and_reimport(tmp_path):
    import zipfile
    from mygpr.interfaces.backend import MyGPRBackend
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

    path, _ = csv_source(tmp_path)
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="1.1.15 export")
        pid = project.project_id
        backend.projects.import_line_source(pid, "L01", path)
        artifact = backend.project_processing.execute_pipeline(
            pid, "L01", PipelineDefinition(steps=(PipelineStep("dewow", {"window": 5}),))
        )
        job = backend.submit_processing_result_package(
            pid, "L01", str(tmp_path / "results.zip"), artifact_id=artifact.artifact_id
        )
        result = backend.jobs.wait(job, timeout=60)
        assert result.status.value == "completed", result.error_message
        with zipfile.ZipFile(result.result.archive_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            exported = tmp_path / "exported.h5"
            exported.write_bytes(archive.read(manifest["steps"][0]["data"]))
        ds = load_gpr_dataset(exported)
        # Artifact columns may change: records remain source-row provenance in
        # the input manifest, rather than becoming current-column annotations.
        manifest = unwrap(ds.metadata)["processing_manifest"]
        assert unwrap(manifest["input_dataset"])["acquisition_markers"] == markers()
        with h5py.File(exported, "r") as handle:
            from mygpr.domain.project.trace_snapshot import read_trace_snapshot

            np.testing.assert_array_equal(read_trace_snapshot(handle)["trace_id"], [0, 20, 40, 60, 80])
    finally:
        backend.shutdown()
