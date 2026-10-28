import numpy as np
import pytest
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def test_kirchhoff_elevation_public_pipeline(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='elevation')
        pid = project.project_id
        backend.projects.save_line_dataset(pid, 'L01', np.ones((48,16), np.float32), length_m=1.5, time_window_ns=12.)
        session = backend.projects._session(pid)
        folder = session._store.root / 'sensors'
        folder.mkdir(exist_ok=True)
        ground = np.linspace(100., 100.3, 16)
        np.savez(folder / 'ground.npz', ground_elevation_m=ground)
        rec = session._store.get_line('L01')
        rec.trace_metadata_path = 'sensors/ground.npz'
        session._store.upsert_line(rec)
        params = {'freq':5e7, 'depth':.4, 'v':.1, 'weight':0., 'length_m':1.5, 'time_window_ns':12., 'backend':'cpu', 'topo_cor':1}
        from mygpr.application.jobs.context import ExecutionContext
        warnings = []
        art = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('kirchhoff_migration', params),)), context=ExecutionContext(warning_callback=warnings.append))
        assert any(w.get('code') == 'kirchhoff_trace_grid_resampled' for w in warnings)
        data = backend.projects.read_artifact_dataset(pid, 'L01', art.artifact_id)
        axis = np.asarray(data.header_info['elevation_axis_m'])
        assert axis.shape == (48,) and np.all(np.diff(axis) < 0)
        assert axis[0] == pytest.approx(100.4)
        assert data.header_info['time_axis_status'] == 'unknown'
        assert data.trace_metadata == {}
        np.testing.assert_allclose(data.header_info['distance_axis_m'], np.linspace(0, 1.4, 16), atol=1e-6)
        info = backend.projects.get_artifact_dataset_info(pid, 'L01', art.artifact_id)
        np.testing.assert_allclose(info.metadata['elevation_axis_m'], axis)
        continued = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('amplitude_scale', {'scale':1.}),)), input_artifact_id=art.artifact_id)
        np.testing.assert_allclose(backend.projects.read_artifact_dataset(pid, 'L01', continued.artifact_id).header_info['elevation_axis_m'], axis)
        from mygpr.infrastructure.reporting.processing_package_export import _write_h5
        from mygpr.application.jobs.context import ExecutionContext
        from core.processing_result_hdf5 import load_processing_result
        import h5py
        dataset = session._artifact_dataset('L01', art.artifact_id)[2]
        path = tmp_path / 'result.h5'
        _write_h5(path, dataset, art, ExecutionContext.null())
        with h5py.File(path, 'r') as handle:
            loaded = load_processing_result(handle, source=path, line_id='L02')
        np.testing.assert_allclose(loaded.metadata['elevation_axis_m'], axis)
        backend.projects.import_line_source(pid, 'L02', path)
        np.testing.assert_allclose(backend.projects.read_dataset(pid, 'L02').header_info['elevation_axis_m'], axis)
        from tests.test_processing_result_package import export
        import json
        import zipfile
        package = export(backend, project, tmp_path / 'package.zip', art.artifact_id)
        with zipfile.ZipFile(package.archive_path) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            entry = next(item for item in manifest['steps'] if item['artifact_id'] == art.artifact_id)
            packaged = tmp_path / 'packaged.h5'
            packaged.write_bytes(archive.read(entry['data']))
        with h5py.File(packaged, 'r') as handle:
            np.testing.assert_allclose(handle['elevation_axis_m'][:], axis)
        with h5py.File(packaged, 'r+') as handle:
            handle['elevation_axis_m'][0] += 1
        with h5py.File(packaged, 'r') as handle, pytest.raises(ValueError, match='Conflicting'):
            load_processing_result(handle, source=packaged, line_id='bad')
        with pytest.raises(ValueError):
            backend.projects.export_artifact_segy(pid, 'L01', art.artifact_id, tmp_path / 'bad.sgy')
        with pytest.raises(ValueError, match='时间轴'):
            backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('time_cut', {'mode':'remove_above', 'time_start_ns':1}),)), input_artifact_id=art.artifact_id)
        with pytest.raises(ValueError, match='topo_cor=1'):
            backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('kirchhoff_migration', {**params, 'topo_cor':2}),)))
    finally:
        backend.shutdown()


def test_resampled_grid_spacing_follows_observed_ramp():
    from mygpr.infrastructure.processing.algorithms.kirchhoff.shared import _resample_to_input_grid
    ramp = np.repeat(np.arange(4, dtype=float)[:, None], 3, axis=1)
    out, dz, dx = _resample_to_input_grid(ramp, (7, 5), .2, .4)
    np.testing.assert_allclose(out[:, 0] * .2, np.arange(7) * dz)
    assert dx == pytest.approx(.2)
    _, one_dz, _ = _resample_to_input_grid(ramp[:1], (1, 5), .2, .4)
    assert one_dz == 0


def test_legacy_elevation_axis_roundtrip(tmp_path):
    import json
    from core.project_storage_backend import LegacyProjectStorageBackend
    folder = tmp_path / 'processed' / 'L01'
    folder.mkdir(parents=True)
    np.save(folder / 'L01_processed_1.npy', np.ones((3, 2), np.float32))
    (folder / 'L01_processing_manifest_1.json').write_text(json.dumps({'output_header': {'is_elevation': True, 'time_axis_status':'unknown', 'elevation_axis_m':[10, 9, 8]}}), encoding='utf-8')
    loaded = LegacyProjectStorageBackend(tmp_path, None).load_processing_artifact('L01', 'L01_processed_1')
    np.testing.assert_array_equal(loaded.to_metadata()['elevation_axis_m'], [10, 9, 8])
