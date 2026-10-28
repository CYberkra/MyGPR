import h5py
import numpy as np
import pytest

from core.chunked_gpr_io import ImportCancelled
from core.field_import_preview import build_import_preflight
from core.field_project_operations import import_line_data
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import load_gpr_dataset, load_gpr_dataset_for_import


def test_hdf5_is_visible_in_native_import_file_filter():
    from core.gpr_format_registry import get_format_spec, supported_file_dialog_filter

    assert get_format_spec("line.h5").support.startswith("native")
    assert "*.h5" in supported_file_dialog_filter()
    assert "*.hdf5" in supported_file_dialog_filter()


def write_acquisition(path, *, traces=376, samples=501, window=900.0, component="real"):
    values = np.arange(traces * samples, dtype=np.float32).reshape(traces, samples) - 100
    with h5py.File(path, "w") as handle:
        handle.attrs.update(samples_per_trace=samples, time_window_ns=window,
                            store_component=component, prefix="lineData", created="2026-10-14T17:18:27")
        handle.create_dataset("traces", data=values, chunks=(1, samples), maxshape=(None, samples))
        meta = handle.create_group("trace_meta")
        for name in ("lon", "lat", "alt", "height"):
            meta.create_dataset(name, data=np.full(traces, np.nan))
    return values


@pytest.mark.parametrize("trace_count", [376, 800])
@pytest.mark.parametrize("component", ["real", "magnitude"])
def test_acquisition_axes_component_and_import_parity(tmp_path, trace_count, component):
    source = tmp_path / "line.h5"
    expected = write_acquisition(source, traces=trace_count, component=component)
    preview = build_import_preflight(source)
    assert preview.can_import
    assert (preview.sample_count, preview.trace_count) == (501, trace_count)
    assert preview.time_window_ns == 900.0
    direct = load_gpr_dataset(source)
    staged = load_gpr_dataset_for_import(source, line_id="L01", staging_dir=tmp_path / "stage")
    assert isinstance(staged.matrix, np.memmap)
    for dataset in (direct, staged):
        assert dataset.matrix.shape == (501, trace_count)
        np.testing.assert_array_equal(dataset.matrix, expected.T)
        assert dataset.time_window_ns == 900.0
        np.testing.assert_allclose(dataset.time_axis_ns, np.arange(501) * (900.0 / 501), rtol=1e-6)
        assert dataset.metadata["store_component"] == component
        assert dataset.metadata["trace_meta"]["height"] == [None] * trace_count


def test_acquisition_chunked_import_and_header_only_preflight(tmp_path, monkeypatch):
    source = tmp_path / "line.h5"
    expected = write_acquisition(source, traces=600)
    original = h5py.Dataset.__getitem__
    phase = "preflight"

    def bounded_read(dataset, selection):
        if phase == "preflight":
            raise AssertionError("preflight must not read dataset payload")
        assert isinstance(selection, slice) and selection.stop - selection.start <= 256
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded_read)
    assert build_import_preflight(source).can_import
    phase = "import"
    result = load_gpr_dataset_for_import(source, line_id="L01", staging_dir=tmp_path / "stage")
    np.testing.assert_array_equal(result.matrix, expected.T)


def test_acquisition_cancel_cleans_staging_matrix(tmp_path):
    source = tmp_path / "line.h5"
    write_acquisition(source, traces=600)
    cancelled = False

    def progress(stage, current, total):
        nonlocal cancelled
        if stage == "copy_hdf5" and current:
            cancelled = True

    with pytest.raises(ImportCancelled):
        load_gpr_dataset_for_import(source, line_id="L01", staging_dir=tmp_path / "stage",
                                    cancel_requested=lambda: cancelled, progress_callback=progress)
    assert not (tmp_path / "stage" / "matrix.npy").exists()


def test_project_import_does_not_copy_staging_matrix_for_metadata(tmp_path, monkeypatch):
    source = tmp_path / "line.h5"
    write_acquisition(source)

    def forbid_copy(*args, **kwargs):
        raise AssertionError("metadata serialization must not copy the mmap matrix")

    monkeypatch.setattr(np.memmap, "__deepcopy__", forbid_copy)
    store = FieldProjectStore.create_empty(tmp_path / "project", name="bounded-import")
    import_line_data(store, source, line_id="L01")
    assert store.load_gpr_dataset("L01").matrix.shape == (501, 376)


@pytest.mark.parametrize("fault", ["samples", "meta_length", "component", "window"])
def test_acquisition_rejects_inconsistent_schema(tmp_path, fault):
    source = tmp_path / "bad.h5"
    write_acquisition(source)
    with h5py.File(source, "r+") as handle:
        if fault == "samples":
            handle.attrs["samples_per_trace"] = 500
        elif fault == "meta_length":
            del handle["trace_meta/height"]
            handle.create_dataset("trace_meta/height", data=[1.0])
        elif fault == "component":
            handle.attrs["store_component"] = "complex"
        else:
            handle.attrs["time_window_ns"] = -1.0
    assert not build_import_preflight(source).can_import
    with pytest.raises(ValueError):
        load_gpr_dataset(source)
    with pytest.raises(ValueError):
        load_gpr_dataset_for_import(source, line_id="L01", staging_dir=tmp_path / "stage")


def test_acquisition_unknown_time_remains_unknown(tmp_path):
    source = tmp_path / "line.h5"
    write_acquisition(source, window=0.0)
    store = FieldProjectStore.create_empty(tmp_path / "project", name="unknown-time")
    import_line_data(store, source, line_id="L01")
    result = store.load_gpr_dataset("L01")
    assert result.time_window_ns == 0.0
    assert np.all(result.time_axis_ns == 0.0)
    assert np.all(result.depth_axis_m == 0.0)


def test_project_reopen_preserves_trace_metadata_and_missing_values(tmp_path):
    from mygpr.infrastructure.persistence.field_project_adapter import _load_trace_metadata

    source = tmp_path / "line.h5"
    expected = write_acquisition(source)
    with h5py.File(source, "r+") as handle:
        handle["trace_meta/lon"][0] = 104.0
        handle["trace_meta/lat"][0] = 30.0
        handle["trace_meta/alt"][0] = 510.0
        handle["trace_meta/height"][0] = 10.0
    store = FieldProjectStore.create_empty(tmp_path / "project", name="hdf5")
    import_line_data(store, source, line_id="L01")
    assert not list((store.root / "raw" / "L01").glob(".import_staging_*"))
    reopened = FieldProjectStore.open(store.root)
    dataset = reopened.load_gpr_dataset("L01")
    np.testing.assert_array_equal(dataset.matrix, expected.T)
    assert dataset.time_window_ns == 900.0
    sensors = _load_trace_metadata(reopened, "L01")
    assert sensors["longitude"][0] == 104.0
    assert sensors["altitude_m"][0] == 510.0
    assert sensors["flight_height_m"][0] == 10.0
    assert np.isnan(sensors["flight_height_m"][1:]).all()


def test_generic_hdf5_matrix_keeps_existing_orientation(tmp_path):
    path = tmp_path / "generic.h5"
    expected = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("matrix", data=expected)
    assert build_import_preflight(path).shape_text == "50 × 32"
    np.testing.assert_array_equal(load_gpr_dataset(path).matrix, expected)


def test_backend_acquisition_exposes_aligned_sensors(tmp_path):
    from mygpr.interfaces.backend import MyGPRBackend

    source = tmp_path / "line.h5"
    expected = write_acquisition(source)
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        dataset = backend.acquisition.load_dataset(source, line_id="L01")
        np.testing.assert_array_equal(dataset.data, expected.T)
        assert np.isnan(dataset.trace_metadata["flight_height_m"]).all()
    finally:
        backend.shutdown()


@pytest.mark.parametrize("save_intermediates", [True, False])
def test_processing_preserves_acquisition_sampling_axis(tmp_path, save_intermediates):
    from mygpr.interfaces.backend import MyGPRBackend
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    from mygpr.application.jobs.models import JobStatus

    source = tmp_path / "line.h5"
    expected = write_acquisition(source)
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="sampling")
        backend.projects.import_line_source(project.project_id, "L01", source)
        original = backend.projects.read_dataset(project.project_id, "L01")
        pipeline = PipelineDefinition(name="scale", steps=(PipelineStep("amplitude_scale", {"scale": 2.0}),))
        job = backend.submit_project_pipeline(project.project_id, "L01", pipeline, save_intermediates=save_intermediates)
        snapshot = backend.jobs.wait(job, timeout=30)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        result = backend.projects.read_artifact_dataset(project.project_id, "L01", snapshot.result.artifact_id)
        np.testing.assert_array_equal(result.data, expected.T * 2)
        np.testing.assert_array_equal(result.header_info["time_axis_ns"], original.header_info["time_axis_ns"])
    finally:
        backend.shutdown()


def test_auto_loader_and_cli_accept_acquisition_hdf5(tmp_path, monkeypatch):
    import cli_batch
    from core.gpr_io import auto_load_data

    source = tmp_path / "line.h5"
    expected = write_acquisition(source)
    payload = auto_load_data(str(source))
    assert payload["header_info"]["total_time_ns"] == 900.0
    assert payload["header_info"]["time_step_s"] == pytest.approx(900e-9 / 501)
    assert payload["header_info"]["trace_interval_m"] == 0.0
    np.testing.assert_array_equal(payload["data"], expected.T)
    monkeypatch.setattr(cli_batch, "save_image", lambda *args, **kwargs: None)
    result = cli_batch.run_job({"id": "line", "input": str(source),
                               "methods": [{"key": "amplitude_scale", "params": {"mode": "constant", "scale": 2.0}}]},
                               str(tmp_path), str(tmp_path / "out"))
    assert result["status"] == "ok"
    np.testing.assert_allclose(np.loadtxt(tmp_path / result["final_csv"], delimiter=","), expected.T * 2)


def test_unknown_distance_is_not_exposed_as_meters(tmp_path):
    from core.gpr_io import auto_load_data
    from core.gui_rendering import bundle_from_dataset
    from mygpr.infrastructure.processing.algorithms.rtm import _resolve_geometry

    source = tmp_path / "line.h5"
    write_acquisition(source)
    dataset = load_gpr_dataset(source)
    assert dataset.length_m == 0.0
    assert bundle_from_dataset(dataset).trace_axis_m is None
    payload = auto_load_data(str(source))
    assert payload["header_info"]["length_m"] == 0.0
    with pytest.raises(ValueError, match="length_m"):
        _resolve_geometry(dataset.matrix.shape, {}, payload["header_info"])
    assert _resolve_geometry(dataset.matrix.shape, {"length_m": 75.0}, payload["header_info"])["length_m"] == 75.0
    calibrated = load_gpr_dataset(source, length_m=75.0)
    assert calibrated.length_m == 75.0
    assert bundle_from_dataset(calibrated).trace_axis_m is not None


@pytest.mark.parametrize("method,key", [("rtm_migration", "length_m"), ("kirchhoff_migration", "length_m"), ("stolt_migration", "dx")])
def test_native_migration_requires_explicit_unknown_geometry(method, key):
    from mygpr.domain.processing.models import ProcessingRequest
    from mygpr.infrastructure.processing.native_adapter import prepare_native_params

    data = np.zeros((32, 16), dtype=np.float32)
    header = {"distance_axis_status": "unknown", "length_m": 0.0, "time_window_ns": 900.0}
    with pytest.raises(ValueError, match=key):
        prepare_native_params(ProcessingRequest(data=data, method_id=method, header_info=header))
    assert prepare_native_params(ProcessingRequest(data=data, method_id=method, params={key: 2.0}, header_info=header))[key] == 2.0


@pytest.mark.parametrize("window", [0.0, 900.0])
def test_acquisition_npz_roundtrip_preserves_window_and_unknown_distance(tmp_path, window):
    source = tmp_path / "line.h5"
    write_acquisition(source, window=window)
    original = load_gpr_dataset(source)
    path = original.save_npz(tmp_path / "copy.npz")
    reloaded = load_gpr_dataset(path)
    assert reloaded.time_window_ns == window
    assert reloaded.length_m == 0.0
    np.testing.assert_array_equal(reloaded.time_axis_ns, original.time_axis_ns)


@pytest.mark.parametrize("save_intermediates", [True, False])
def test_project_migration_rejects_unknown_spacing_in_both_modes(tmp_path, save_intermediates):
    from mygpr.interfaces.backend import MyGPRBackend
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    from mygpr.application.jobs.models import JobStatus

    source = tmp_path / "line.h5"
    write_acquisition(source, traces=16, samples=32)
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="unknown geometry")
        backend.projects.import_line_source(project.project_id, "L01", source)
        backend.projects.close_project(project.project_id)
        project = backend.projects.open_project(tmp_path / "project")
        job = backend.submit_project_pipeline(project.project_id, "L01", PipelineDefinition(name="migration", steps=(
            PipelineStep("stolt_migration", {}),)), save_intermediates=save_intermediates)
        snapshot = backend.jobs.wait(job, timeout=30)
        assert snapshot.status is JobStatus.FAILED
        assert "dx" in snapshot.error_message
        assert not backend.projects.list_artifacts(project.project_id, "L01")
    finally:
        backend.shutdown()


@pytest.mark.parametrize("save_intermediates", [True, False])
def test_acquisition_time_cut_keeps_original_sample_interval(tmp_path, save_intermediates):
    from mygpr.interfaces.backend import MyGPRBackend
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    from mygpr.application.jobs.models import JobStatus

    source = tmp_path / "line.h5"
    expected = write_acquisition(source)
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="cut sampling")
        backend.projects.import_line_source(project.project_id, "L01", source)
        source_axis = backend.projects.read_dataset(project.project_id, "L01").header_info["time_axis_ns"]
        selected = (source_axis >= 180.0) & (source_axis < 450.0)
        steps = (PipelineStep("time_cut", {"mode": "keep_range", "time_start_ns": 180.0, "time_end_ns": 450.0}),
                 PipelineStep("amplitude_scale", {"scale": 2.0}))
        job = backend.submit_project_pipeline(project.project_id, "L01", PipelineDefinition(name="cut", steps=steps),
                                              save_intermediates=save_intermediates)
        snapshot = backend.jobs.wait(job, timeout=30)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        result = backend.projects.read_artifact_dataset(project.project_id, "L01", snapshot.result.artifact_id)
        np.testing.assert_array_equal(result.data, expected.T[selected] * 2)
        np.testing.assert_array_equal(result.header_info["time_axis_ns"], source_axis[selected])
        second_job = backend.submit_project_pipeline(
            project.project_id, "L01", PipelineDefinition(name="second cut", steps=(
                PipelineStep("time_cut", {"mode": "remove_above", "time_start_ns": 200.0}),)),
            input_artifact_id=snapshot.result.artifact_id, save_intermediates=save_intermediates,
        )
        second = backend.jobs.wait(second_job, timeout=30)
        assert second.status is JobStatus.COMPLETED, second.error_message
        reloaded = backend.projects.read_artifact_dataset(project.project_id, "L01", second.result.artifact_id)
        selected &= source_axis >= 200.0
        np.testing.assert_array_equal(reloaded.data, expected.T[selected] * 2)
        np.testing.assert_array_equal(reloaded.header_info["time_axis_ns"], source_axis[selected])
    finally:
        backend.shutdown()
