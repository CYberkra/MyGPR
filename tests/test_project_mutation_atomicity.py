"""Failure paths must preserve the last complete project state."""

from pathlib import Path

import numpy as np
import pytest

from mygpr.interfaces.backend import MyGPRBackend


@pytest.fixture
def project(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    summary = backend.projects.create_project(tmp_path / "project", name="Atomic writes")
    try:
        yield backend, summary.project_id
    finally:
        backend.shutdown()


def test_invalid_import_leaves_no_line_or_catalog_row(project, tmp_path):
    backend, pid = project
    source = tmp_path / "invalid.npy"
    np.save(source, np.arange(4, dtype=np.float32))
    with pytest.raises(ValueError, match="2D"):
        backend.projects.import_line_source(pid, "L01", source)
    assert backend.projects.list_lines(pid) == ()
    store = backend.projects._session(pid)._store
    assert store.storage.catalog.list_lines() == []
    assert not (store.root / "raw" / "L01" / source.name).exists()


def test_failed_import_replacement_restores_waveform_and_metadata(project, tmp_path, monkeypatch):
    backend, pid = project
    source = tmp_path / "replacement.npy"
    np.save(source, np.full((16, 4), 2, dtype=np.float32))
    backend.projects.save_line_dataset(pid, "L01", np.ones((16, 4), dtype=np.float32), time_window_ns=900)
    store = backend.projects._session(pid)._store
    before_manifest = (store.root / "project.json").read_bytes()
    before_line = store.storage.catalog.list_lines()
    original = store.write_json

    def fail_metadata(path, payload):
        if Path(path) == store.gpr_metadata_path("L01"):
            raise OSError("metadata disk failure")
        return original(path, payload)

    monkeypatch.setattr(store, "write_json", fail_metadata)
    with pytest.raises(OSError, match="metadata disk failure"):
        backend.projects.import_line_source(pid, "L01", source)
    np.testing.assert_array_equal(backend.projects.read_dataset(pid, "L01").data, np.ones((16, 4)))
    assert (store.root / "project.json").read_bytes() == before_manifest
    assert store.storage.catalog.list_lines() == before_line
    assert not (store.root / "raw" / "L01" / source.name).exists()


def save_result(backend, pid, name):
    return backend.projects.save_processing_artifact(
        pid,
        "L01",
        np.ones((16, 4), dtype=np.float32),
        name=name,
        method_id="dewow",
        method_name=name,
        params={},
        pipeline=[],
        branch_id="L01:main",
        input_dataset={},
    )


@pytest.mark.parametrize("failure", ["manifest", "params", "latest", "descriptor", "project"])
def test_failed_result_save_restores_catalog_head_and_files(project, monkeypatch, failure):
    backend, pid = project
    backend.projects.save_line_dataset(pid, "L01", np.ones((16, 4), dtype=np.float32), time_window_ns=900)
    first = save_result(backend, pid, "existing")
    store = backend.projects._session(pid)._store
    before_rows = store.storage.catalog.list_artifacts(line_id="L01")
    before_lines = store.storage.catalog.list_lines()
    before_files = {
        p.relative_to(store.root): p.read_bytes()
        for folder in ("processed", "data")
        for p in (store.root / folder).rglob("*")
        if p.is_file()
    }
    original = store.write_json

    def fail_write(path, payload):
        name = Path(path).name
        matches = {
            "manifest": "_processing_manifest_" in name,
            "params": "_params_" in name,
            "latest": name == "L01_params.json",
            "descriptor": name.endswith(".artifact"),
        }
        if matches.get(failure):
            raise OSError("result metadata disk failure")
        return original(path, payload)

    monkeypatch.setattr(store, "write_json", fail_write)
    if failure == "project":

        def fail_manifest():
            raise OSError("result metadata disk failure")

        monkeypatch.setattr(store, "save_manifest", fail_manifest)
    with pytest.raises(OSError, match="result metadata disk failure"):
        save_result(backend, pid, "failed")
    assert store.storage.catalog.list_artifacts(line_id="L01") == before_rows
    assert store.storage.catalog.branch_head("L01:main") == first.artifact_id
    assert store.storage.catalog.list_lines() == before_lines
    after_files = {
        p.relative_to(store.root): p.read_bytes()
        for folder in ("processed", "data")
        for p in (store.root / folder).rglob("*")
        if p.is_file()
    }
    assert after_files == before_files


@pytest.mark.parametrize("failure", ["second_move", "catalog", "manifest"])
def test_failed_batch_delete_restores_whole_selection(project, monkeypatch, failure):
    from mygpr.infrastructure.persistence import artifact_delete

    backend, pid = project
    backend.projects.save_line_dataset(pid, "L01", np.ones((16, 4), dtype=np.float32), time_window_ns=900)
    first, second = save_result(backend, pid, "first"), save_result(backend, pid, "second")
    store = backend.projects._session(pid)._store
    before_rows = store.storage.catalog.list_artifacts(line_id="L01")
    head = store.storage.catalog.branch_head("L01:main")
    before_files = {p: p.read_bytes() for p in (store.root / "data").rglob("*.h5")}
    move = artifact_delete.shutil.move
    calls = []

    def fail_second(source, destination, *args, **kwargs):
        calls.append(source)
        if len(calls) == 2:
            raise PermissionError("result locked")
        return move(source, destination, *args, **kwargs)

    def fail(*args, **kwargs):
        raise PermissionError("result locked")

    if failure == "second_move":
        monkeypatch.setattr(artifact_delete.shutil, "move", fail_second)
    elif failure == "catalog":
        monkeypatch.setattr(store.storage.catalog, "delete_artifact", fail)
    else:
        monkeypatch.setattr(artifact_delete, "atomic_write_json", fail)
    with pytest.raises(PermissionError, match="result locked"):
        backend.projects.delete_artifacts(pid, "L01", [first.artifact_id, second.artifact_id])
    assert store.storage.catalog.list_artifacts(line_id="L01") == before_rows
    assert store.storage.catalog.branch_head("L01:main") == head
    assert {p: p.read_bytes() for p in (store.root / "data").rglob("*.h5")} == before_files
    assert not list((store.root / ".trash").rglob("*.h5"))


@pytest.mark.parametrize("endpoint", [True, False])
def test_h5_timebase_is_used_by_saved_project_processing(project, tmp_path, endpoint):
    import h5py
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

    backend, pid = project
    source = tmp_path / "stream.h5"
    with h5py.File(source, "w") as handle:
        handle.attrs.update(
            samples_per_trace=501,
            time_window_ns=900.0,
            committed_trace_count=4,
            store_component="real",
            time_axis_endpoint=endpoint,
        )
        handle.create_dataset("traces", data=np.tile(np.arange(501, dtype=np.float32), (4, 1)))
    backend.projects.import_line_source(pid, "L01", source)
    backend.projects.read_dataset(pid, "L01")
    pipeline = PipelineDefinition(name="Timebase", steps=(PipelineStep("set_zero_time", {"new_zero_time": 451.0}),))
    artifact = backend.project_processing.execute_pipeline(pid, "L01", pipeline)
    reopened = backend.projects.read_artifact_dataset(pid, "L01", artifact.artifact_id)
    expected_shift = int(451 / (900 / (500 if endpoint else 501)))
    assert reopened.data[0, 0] == expected_shift


def test_workbench_crop_save_preserves_output_axis(project):
    backend, pid = project
    backend.projects.save_line_dataset(pid, "L01", np.ones((64, 4), dtype=np.float32), time_window_ns=900)
    service = backend.processing_workbench
    session = service.open_session(pid, "L01", restore_draft=False)
    session = service.append_step(
        session.session_id, "time_cut", {"mode": "keep_range", "time_start_ns": 100, "time_end_ns": 500}
    )
    current_header = dict(service._session(session.session_id).current_header)
    artifact = service.save_session(session.session_id)
    reopened = backend.projects.read_artifact_dataset(pid, "L01", artifact.artifact_id)
    assert reopened.header_info["time_window_ns"] == pytest.approx(current_header["total_time_ns"])
    assert reopened.header_info["time_cut_offset_ns"] == pytest.approx(current_header["time_cut_offset_ns"])


def test_endpoint_crop_saved_axis_keeps_input_sampling_interval(project):
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

    backend, pid = project
    backend.projects.save_line_dataset(pid, "L01", np.ones((501, 4), dtype=np.float32), time_window_ns=900)
    store = backend.projects._session(pid)._store
    # Mark the normalized fact source explicitly, as acquisition H5 does.
    dataset = store.load_gpr_dataset("L01")
    dataset.metadata["time_axis_endpoint"] = True
    dataset.metadata["time_step_s"] = 1.8e-9
    dataset.time_axis_ns = np.linspace(0, 900, 501, dtype=np.float32)
    from core.gpr_data_model import time_to_depth_axis
    dataset.depth_axis_m = time_to_depth_axis(dataset.time_axis_ns, dataset.dielectric_constant)
    store.save_gpr_dataset("L01", dataset)
    artifact = backend.project_processing.execute_pipeline(
        pid,
        "L01",
        PipelineDefinition(
            name="Crop endpoint",
            steps=(PipelineStep("time_cut", {"mode": "keep_range", "time_start_ns": 100, "time_end_ns": 500}),),
        ),
    )
    reopened = backend.projects.read_artifact_dataset(pid, "L01", artifact.artifact_id)
    axis = reopened.header_info["time_axis_ns"]
    assert float(axis[1] - axis[0]) == pytest.approx(1.8, abs=1e-5)


def test_windows_lock_on_second_delete_target_preserves_selection(project):
    import os
    import h5py
    from core.hdf5_line_container import artifacts_dir_path

    if os.name != "nt":
        pytest.skip("Windows HDF5 file sharing contract")
    backend, pid = project
    backend.projects.save_line_dataset(pid, "L01", np.ones((16, 4), dtype=np.float32), time_window_ns=900)
    first, second = save_result(backend, pid, "first"), save_result(backend, pid, "second")
    store = backend.projects._session(pid)._store
    path = artifacts_dir_path(store.storage.line_container_path("L01")) / f"{first.artifact_id}.h5"
    before = store.storage.catalog.list_artifacts(line_id="L01")
    with h5py.File(path, "r"):
        with pytest.raises(PermissionError):
            backend.projects.delete_artifacts(pid, "L01", [first.artifact_id, second.artifact_id])
    assert store.storage.catalog.list_artifacts(line_id="L01") == before
    for artifact in (first, second):
        np.testing.assert_array_equal(
            backend.projects.read_artifact_dataset(pid, "L01", artifact.artifact_id).data, np.ones((16, 4))
        )
    assert not list((store.root / ".transactions").glob("*/transaction.json"))


@pytest.mark.parametrize("boundary", ["before_catalog_commit", "after_catalog_commit"])
def test_interrupted_import_reconciles_files_with_catalog(tmp_path, boundary):
    import os
    import subprocess
    import sys
    from core.field_project_store import FieldProjectStore

    root = tmp_path / "project"
    backend = MyGPRBackend.create_default(max_workers=1)
    summary = backend.projects.create_project(root, name="Interrupted import")
    backend.projects.save_line_dataset(
        summary.project_id, "L01", np.ones((16, 4), dtype=np.float32), time_window_ns=900
    )
    backend.shutdown()
    replacement = tmp_path / "replacement.npy"
    np.save(replacement, np.full((16, 4), 2, dtype=np.float32))
    child = """
import os, sys
from core.field_project_store import FieldProjectStore
from core.storage_primitives import FileTransaction
phase, root, source = sys.argv[1:]
original = FileTransaction.commit if phase == 'before_catalog_commit' else FileTransaction._cleanup
def interrupt(transaction):
    if transaction.label == 'import-line' and phase == 'after_catalog_commit':
        os._exit(17)
    original(transaction)
    if transaction.label == 'import-line':
        os._exit(17)
if phase == 'before_catalog_commit':
    FileTransaction.commit = interrupt
else:
    FileTransaction._cleanup = interrupt
store = FieldProjectStore.open(root)
store.import_line_file('L01', source)
raise AssertionError('interruption boundary not reached')
"""
    child_result = subprocess.run(
        [sys.executable, "-c", child, boundary, str(root), str(replacement)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child_result.returncode == 17, child_result.stderr
    with FieldProjectStore.open(root, recover_stale_lock=True) as reopened:
        expected = 1 if boundary == "before_catalog_commit" else 2
        np.testing.assert_array_equal(np.asarray(reopened.load_gpr_dataset("L01").matrix), np.full((16, 4), expected))
        assert len(reopened.storage.catalog.list_lines()) == 1


@pytest.mark.parametrize("damage", ["root_tree", "missing_backup", "transaction_tree"])
def test_invalid_recovery_journal_never_changes_project_files(tmp_path, damage):
    import json
    from core.storage_primitives import recover_file_transactions

    root = tmp_path / "project"
    journal = root / ".transactions" / "damaged" / "transaction.json"
    journal.parent.mkdir(parents=True)
    existing = root / "raw" / "L01" / "source.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("preserve me")
    payload = {
        "schema": "mygpr.file_transaction.v1",
        "state": "active",
        "files": [],
        "trees": [{"path": ".", "existed": True}],
    }
    if damage == "missing_backup":
        payload.update(files=[{"path": "saved.txt", "existed": True}], trees=[{"path": "raw/L01", "existed": False}])
    elif damage == "transaction_tree":
        payload["trees"] = [{"path": ".transactions", "existed": True}]
    journal.write_text(json.dumps(payload))
    records = recover_file_transactions(root)
    assert records and records[0]["success"] is False
    assert existing.read_text() == "preserve me"
    assert journal.exists()


def test_explicit_sample_interval_works_without_window(project):
    from mygpr.domain.processing.models import ProcessingRequest

    backend, _ = project
    data = np.tile(np.arange(16, dtype=np.float32)[:, None], (1, 4))
    result = backend.processing.execute_method(
        ProcessingRequest(
            data=data, method_id="set_zero_time", params={"new_zero_time": 5.0}, header_info={"time_step_s": 1e-9}
        )
    )
    assert result.data[0, 0] == 5


@pytest.mark.parametrize("legacy", [False, True])
def test_single_sample_crop_preserves_zero_span_and_interval(project, legacy):
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

    backend, pid = project
    store = backend.projects._session(pid)._store
    if legacy:
        from core.project_storage_backend import LEGACY_STORAGE_BACKEND, create_storage_backend

        store.manifest.storage_backend = LEGACY_STORAGE_BACKEND
        store.manifest.legacy_layout = True
        store.storage = create_storage_backend(store.root, store.manifest)
        store.save_manifest()
    backend.projects.save_line_dataset(pid, "L01", np.ones((501, 4), dtype=np.float32), time_window_ns=900)
    store = backend.projects._session(pid)._store
    dataset = store.load_gpr_dataset("L01")
    dataset.metadata["time_axis_endpoint"] = True
    dataset.metadata["time_step_s"] = 1.8e-9
    dataset.time_axis_ns = np.linspace(0, 900, 501, dtype=np.float32)
    from core.gpr_data_model import time_to_depth_axis
    dataset.depth_axis_m = time_to_depth_axis(dataset.time_axis_ns, dataset.dielectric_constant)
    store.save_gpr_dataset("L01", dataset)
    artifact = backend.project_processing.execute_pipeline(
        pid,
        "L01",
        PipelineDefinition(
            name="Single sample",
            steps=(PipelineStep("time_cut", {"mode": "keep_range", "time_start_ns": 898.2, "time_end_ns": 900}),),
        ),
    )
    reopened = backend.projects.read_artifact_dataset(pid, "L01", artifact.artifact_id)
    assert reopened.data.shape == (1, 4)
    assert reopened.header_info["time_window_ns"] == 0
    assert reopened.header_info["time_step_s"] == pytest.approx(1.8e-9)
    continued = backend.project_processing.execute_pipeline(
        pid,
        "L01",
        PipelineDefinition(
            name="Continue single sample", steps=(PipelineStep("set_zero_time", {"new_zero_time": 1.0}),)
        ),
        input_artifact_id=artifact.artifact_id,
    )
    assert backend.projects.read_artifact_dataset(pid, "L01", continued.artifact_id).data.shape == (1, 4)


def test_interrupted_nested_manifest_write_restores_original_snapshot(tmp_path):
    import os
    import subprocess
    import sys
    from core.field_project_store import FieldProjectStore

    root = tmp_path / "project"
    backend = MyGPRBackend.create_default(max_workers=1)
    project = backend.projects.create_project(root, name="Nested interruption")
    backend.projects.save_line_dataset(
        project.project_id, "L01", np.ones((16, 4), dtype=np.float32), time_window_ns=900
    )
    backend.shutdown()
    before = (root / "project.json").read_bytes()
    source = tmp_path / "replacement.npy"
    np.save(source, np.full((16, 4), 2, dtype=np.float32))
    child = """
import itertools, os, sys, uuid
from pathlib import Path
from core import field_project_store
store = field_project_store.FieldProjectStore.open(sys.argv[1])
sequence = itertools.count(1)
uuid.uuid4 = lambda: uuid.UUID(int=next(sequence))
original = field_project_store.atomic_write_json
calls = 0
def interrupt(path, payload):
    global calls
    original(path, payload)
    if Path(path).name == 'project.json':
        calls += 1
        if calls == 2:
            os._exit(17)
field_project_store.atomic_write_json = interrupt
store.import_line_file('L01', sys.argv[2])
raise AssertionError('interruption boundary not reached')
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(root), str(source)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 17, result.stderr
    with FieldProjectStore.open(root, recover_stale_lock=True) as reopened:
        assert (root / "project.json").read_bytes() == before
        np.testing.assert_array_equal(np.asarray(reopened.load_gpr_dataset("L01").matrix), np.ones((16, 4)))


@pytest.mark.parametrize(
    "start,end,expected_start", [(90, 180, 50), (99, 100.8, 55), (180, 360, 100), (89.999999, 180, 50)]
)
def test_crop_exact_sampling_boundaries_round_trip(project, start, end, expected_start):
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

    backend, pid = project
    data = np.tile(np.arange(501, dtype=np.float32)[:, None], (1, 4))
    backend.projects.save_line_dataset(pid, "L01", data, time_window_ns=900)
    store = backend.projects._session(pid)._store
    dataset = store.load_gpr_dataset("L01")
    dataset.metadata["time_axis_endpoint"] = True
    dataset.metadata["time_step_s"] = 1.8e-9
    dataset.time_axis_ns = np.linspace(0, 900, 501, dtype=np.float32)
    from core.gpr_data_model import time_to_depth_axis
    dataset.depth_axis_m = time_to_depth_axis(dataset.time_axis_ns, dataset.dielectric_constant)
    store.save_gpr_dataset("L01", dataset)
    artifact = backend.project_processing.execute_pipeline(
        pid,
        "L01",
        PipelineDefinition(
            name="Exact crop",
            steps=(PipelineStep("time_cut", {"mode": "keep_range", "time_start_ns": start, "time_end_ns": end}),),
        ),
    )
    reopened = backend.projects.read_artifact_dataset(pid, "L01", artifact.artifact_id)
    assert reopened.data[0, 0] == expected_start
    assert reopened.header_info["time_cut_offset_ns"] == pytest.approx(expected_start * 1.8)


@pytest.mark.parametrize("zero,expected", [(90, 50), (180, 100), (900, 500), (89.999999, 49)])
def test_zero_time_exact_sampling_boundary(project, zero, expected):
    from mygpr.domain.processing.models import ProcessingRequest

    backend, _ = project
    result = backend.processing.execute_method(
        ProcessingRequest(
            data=np.tile(np.arange(501, dtype=np.float32)[:, None], (1, 4)),
            method_id="set_zero_time",
            params={"new_zero_time": zero},
            header_info={"total_time_ns": 900.0, "time_axis_endpoint": True},
        )
    )
    assert result.metadata["shift_samples"] == expected
    assert result.data[0, 0] == expected


def test_time_cut_float32_grid_boundary_and_off_grid():
    from mygpr.infrastructure.processing.algorithms.extended.time_cut import method_time_cut
    axis = np.linspace(0, 900, 501, dtype=np.float32)
    data = np.arange(501, dtype=np.float32)[:, None]
    header = {'time_axis_ns': axis, 'time_axis_endpoint': True}
    rounded, _ = method_time_cut(data, mode='keep_range', time_start_ns=1.8, time_end_ns=5.4, header_info=header)
    np.testing.assert_array_equal(rounded[:, 0], [1, 2])
    snapped, _ = method_time_cut(data, mode='keep_range', time_start_ns=5.4, time_end_ns=9.0, header_info=header)
    np.testing.assert_array_equal(snapped[:, 0], [3, 4])
    exact, _ = method_time_cut(data, mode='keep_range', time_start_ns=100.8, time_end_ns=104.4, header_info=header)
    np.testing.assert_array_equal(exact[:, 0], [56, 57])
    offgrid, _ = method_time_cut(data, mode='keep_range', time_start_ns=100.801, time_end_ns=104.401, header_info=header)
    np.testing.assert_array_equal(offgrid[:, 0], [57, 58])
    quantized, _ = method_time_cut(data, mode='keep_range', time_start_ns=898.2, time_end_ns=900, header_info=header)
    np.testing.assert_array_equal(quantized[:, 0], [499])
    above, _ = method_time_cut(data, mode='keep_range', time_start_ns=898.201, time_end_ns=900.1, header_info=header)
    np.testing.assert_array_equal(above[:, 0], [500])
