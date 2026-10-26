"""Saved processing results are exported without recomputation or source edits."""
import hashlib
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
import pytest

from core.gpr_data_model import load_gpr_dataset, load_gpr_dataset_for_import
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


@pytest.fixture
def processed(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    project = backend.projects.create_project(tmp_path / 'project', name='Package')
    pid = project.project_id
    data = np.random.default_rng(41).normal(size=(72, 48)).astype(np.float32)
    backend.projects.save_line_dataset(pid, 'L01', data, time_window_ns=900.0)
    pipeline = PipelineDefinition(steps=(PipelineStep('dewow', {'window': 11}),
                                         PipelineStep('agcGain', {'window': 9})))
    final = backend.project_processing.execute_pipeline(pid, 'L01', pipeline)
    steps = list(backend.projects.list_artifacts(pid, 'L01'))
    first = next(a for a in steps if a.artifact_id != final.artifact_id)
    sibling = backend.project_processing.execute_pipeline(
        pid, 'L01', PipelineDefinition(steps=(PipelineStep('dewow', {'window': 7}),)),
        input_artifact_id=first.artifact_id)
    try:
        yield backend, project, first, final, sibling
    finally:
        backend.shutdown()


def export(backend, project, destination, artifact_id=''):
    job = backend.submit_processing_result_package(project.project_id, 'L01', str(destination),
                                                   artifact_id=artifact_id)
    result = backend.jobs.wait(job, timeout=60)
    assert result.status.value == 'completed', result.error_message
    return result.result


def test_line_package_has_every_saved_step_exact_data_axes_and_parameters(processed, tmp_path):
    backend, project, first, final, sibling = processed
    backend.projects.save_line_dataset(project.project_id, 'L02', np.ones((24, 20), np.float32))
    backend.project_processing.execute_pipeline(project.project_id, 'L02',
        PipelineDefinition(steps=(PipelineStep('dewow', {'window': 5}),)))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(project.root_path).rglob('*')
              if p.is_file() and p.suffix in {'.h5', '.json'}}
    result = export(backend, project, tmp_path / 'L01_处理结果.zip')
    assert set(result.artifact_ids) == {first.artifact_id, final.artifact_id, sibling.artifact_id}
    assert result.verified
    with zipfile.ZipFile(result.archive_path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['scope'] == 'line'
        assert {s['artifact_id'] for s in manifest['steps']} == set(result.artifact_ids)
        for step in manifest['steps']:
            expected = backend.projects.read_artifact_dataset(project.project_id, 'L01', step['artifact_id'])
            params = json.loads(archive.read(step['parameters']))
            assert params['params']['lineage_version'] == 2
            assert archive.read(step['preview']).startswith(b'\x89PNG')
            path = tmp_path / f"{step['artifact_id']}.h5"
            path.write_bytes(archive.read(step['data']))
            with h5py.File(path, 'r') as handle:
                np.testing.assert_array_equal(handle['matrix'][:], expected.data)
                for axis in ('time_axis_ns', 'distance_axis_m', 'depth_axis_m'):
                    np.testing.assert_array_equal(handle[axis][:], expected.header_info[axis])
            for loaded in (load_gpr_dataset(path, line_id='roundtrip'),
                           load_gpr_dataset_for_import(path, line_id='import',
                                                       staging_dir=tmp_path / f"stage-{step['artifact_id']}")):
                np.testing.assert_array_equal(loaded.matrix, expected.data)
                np.testing.assert_array_equal(loaded.time_axis_ns, expected.header_info['time_axis_ns'])
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_to_step_excludes_sibling_and_preserves_existing_destination(processed, tmp_path):
    backend, project, first, final, sibling = processed
    destination = tmp_path / 'result.zip'
    destination.write_bytes(b'previous export')
    result = export(backend, project, destination, final.artifact_id)
    assert result.artifact_ids == (first.artifact_id, final.artifact_id)
    assert destination.read_bytes() == b'previous export'
    assert Path(result.archive_path) != destination
    with zipfile.ZipFile(result.archive_path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['scope'] == 'to_step'
        assert manifest['target_artifact_id'] == final.artifact_id
        assert sibling.artifact_id not in result.artifact_ids


@pytest.mark.parametrize('scope', ['line', 'to_step'])
def test_package_includes_one_reimportable_unprocessed_line_snapshot(processed, tmp_path, scope):
    backend, project, _, final, _ = processed
    expected = backend.projects.read_dataset(project.project_id, 'L01')
    processed_data = backend.projects.read_artifact_dataset(project.project_id, 'L01', final.artifact_id)
    assert not np.array_equal(expected.data, processed_data.data)
    backend.projects.save_line_dataset(project.project_id, 'L02', np.ones((8, 7), np.float32))
    result = export(backend, project, tmp_path / 'raw.zip', final.artifact_id if scope == 'to_step' else '')
    with zipfile.ZipFile(result.archive_path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        raw = manifest['raw_input']
        assert raw['line_id'] == 'L01'
        assert raw['kind'] == 'project_raw_snapshot'
        assert raw['shape'] == list(expected.data.shape)
        assert raw['data'] == 'raw/data.h5'
        assert archive.namelist().count(raw['data']) == 1
        assert not any(name.startswith('steps/') and '/raw/' in name for name in archive.namelist())
        source = json.loads(archive.read(raw['source']))
        assert source['line_id'] == 'L01'
        assert source['kind'] == 'project_raw_snapshot'
        checksums = json.loads(archive.read('checksums.json'))
        for name in (raw['data'], raw['source']):
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksums[name]
        raw_path = tmp_path / 'raw.h5'
        raw_path.write_bytes(archive.read(raw['data']))
    with h5py.File(raw_path) as handle:
        assert handle.attrs['dataset_role'] == 'raw_input'
        assert handle.attrs['artifact_id'] == ''
        np.testing.assert_array_equal(handle['matrix'][:], expected.data)
        for axis in ('time_axis_ns', 'distance_axis_m', 'depth_axis_m'):
            np.testing.assert_array_equal(handle[axis][:], expected.header_info[axis])
    for data in (load_gpr_dataset(raw_path, line_id='raw'),
                 load_gpr_dataset_for_import(raw_path, line_id='raw-import', staging_dir=tmp_path / 'raw-stage')):
        np.testing.assert_array_equal(data.matrix, expected.data)
        np.testing.assert_array_equal(data.time_axis_ns, expected.header_info['time_axis_ns'])
    backend.projects.import_line_source(project.project_id, 'L03', raw_path)
    np.testing.assert_array_equal(backend.projects.read_dataset(project.project_id, 'L03').data, expected.data)


def test_missing_raw_input_fails_explicitly_and_keeps_existing_package(processed, tmp_path):
    backend, project, _, final, _ = processed
    container = Path(project.root_path) / 'data' / 'lines' / 'L01.h5'
    with h5py.File(container, 'r+') as handle:
        del handle['raw/bscan']
    destination = tmp_path / 'result.zip'
    destination.write_bytes(b'previous package')
    job = backend.submit_processing_result_package(project.project_id, 'L01', str(destination))
    snapshot = backend.jobs.wait(job, timeout=60)
    assert snapshot.status.value == 'failed'
    assert '原始测线' in snapshot.error_message
    assert destination.read_bytes() == b'previous package'
    assert list(tmp_path.glob('*.zip')) == [destination]
    assert not list(tmp_path.glob('.mygpr-package-*'))
    # Saved results remain readable; no artifact is deleted to repair missing raw data.
    assert backend.projects.read_artifact_dataset(project.project_id, 'L01', final.artifact_id).data.size
    backend.projects.close_project(project.project_id)


def test_raw_snapshot_uses_persisted_input_when_external_source_has_moved(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    project = backend.projects.create_project(tmp_path / 'project', name='External source')
    matrix = np.arange(240, dtype=np.float32).reshape(24, 10)
    source = tmp_path / 'source.npy'
    np.save(source, matrix)
    try:
        backend.projects.import_line_source(project.project_id, 'L01', source, copy_into_project=False)
        backend.project_processing.execute_pipeline(project.project_id, 'L01',
            PipelineDefinition(steps=(PipelineStep('dewow', {'window': 5}),)))
        source.unlink()
        result = export(backend, project, tmp_path / 'external.zip')
        with zipfile.ZipFile(result.archive_path) as archive:
            raw_path = tmp_path / 'raw.h5'
            raw_path.write_bytes(archive.read(json.loads(archive.read('manifest.json'))['raw_input']['data']))
        np.testing.assert_array_equal(load_gpr_dataset(raw_path, line_id='raw').matrix, matrix)
    finally:
        backend.shutdown()


@pytest.mark.parametrize('stage', ['导出原始测线', '导出矩阵', '压缩处理结果', '校验压缩包'])
def test_cancellation_cleans_staging_and_keeps_old_output(processed, tmp_path, stage):
    from mygpr.application.jobs.context import ExecutionContext
    from mygpr.application.jobs.cancellation import CancellationTokenSource, JobCancelledError
    backend, project, *_ = processed
    token = CancellationTokenSource()
    def progress(done, total, message):
        if message.startswith(stage):
            token.cancel()
    context = ExecutionContext(cancellation_token=token.token, progress_callback=progress)
    destination = tmp_path / 'result.zip'
    destination.write_bytes(b'old')
    with pytest.raises(JobCancelledError):
        backend.projects.export_processing_package(project.project_id, 'L01', destination, context=context)
    assert not context.is_committed
    assert destination.read_bytes() == b'old'
    assert not list(tmp_path.glob('.mygpr-package-*'))
    assert list(tmp_path.glob('*.zip')) == [destination]


@pytest.mark.parametrize('fault', ['write', 'verify'])
def test_disk_or_verification_failure_never_publishes_partial_package(processed, tmp_path, monkeypatch, fault):
    from mygpr.infrastructure.reporting import processing_package_export as module
    backend, project, *_ = processed
    if fault == 'write':
        def broken(*args, **kwargs):
            raise OSError('simulated disk full')
        monkeypatch.setattr(module, '_write_preview', broken)
    else:
        monkeypatch.setattr(module, '_sha256', lambda *args: 'corrupt digest')
    job = backend.submit_processing_result_package(project.project_id, 'L01', str(tmp_path / 'result.zip'))
    snapshot = backend.jobs.wait(job, timeout=60)
    assert snapshot.status.value == 'failed'
    assert not list(tmp_path.glob('*.zip'))
    assert not list(tmp_path.glob('.mygpr-package-*'))
    # The failed job releases its lease; the project can close normally.
    backend.projects.close_project(project.project_id)


def test_export_after_time_cut_preserves_axis_offset_and_project_reimport(processed, tmp_path):
    from core.field_import_preview import build_import_preflight
    backend, project, _, final, _ = processed
    # Persist a saved result with a shifted, inclusive physical time axis.
    original = backend.projects.read_artifact_dataset(project.project_id, 'L01', final.artifact_id)
    artifact = backend.projects.save_processing_artifact(
        project.project_id, 'L01', original.data[12:], name='cut', method_id='time_cut',
        method_name='cut', params={'lineage_version': 2, 'step_input_artifact_id': final.artifact_id,
                                  'output_header': {'time_cut_offset_ns': 150., 'total_time_ns': 750.,
                                                    'time_axis_endpoint': True}},
        pipeline=[], branch_id='cut', input_dataset={})
    expected = backend.projects.read_artifact_dataset(project.project_id, 'L01', artifact.artifact_id)
    result = export(backend, project, tmp_path / 'cut.zip', artifact.artifact_id)
    with zipfile.ZipFile(result.archive_path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        step = next(s for s in manifest['steps'] if s['artifact_id'] == artifact.artifact_id)
        path = tmp_path / 'cut.h5'
        path.write_bytes(archive.read(step['data']))
    assert build_import_preflight(path).can_import
    backend.projects.import_line_source(project.project_id, 'L03', path)
    data = backend.projects.read_dataset(project.project_id, 'L03')
    np.testing.assert_array_equal(data.data, expected.data)
    info = backend.projects.get_dataset_info(project.project_id, 'L03')
    # Importing an exported step must not reset its offset to zero.
    assert info.time_window_ns == 750.
    np.testing.assert_array_equal(data.header_info['time_axis_ns'], expected.header_info['time_axis_ns'])


def test_only_final_run_explains_missing_intermediates(processed, tmp_path):
    backend, project, *_ = processed
    final = backend.project_processing.execute_pipeline(
        project.project_id, 'L01', PipelineDefinition(steps=(
            PipelineStep('dewow', {'window': 11}), PipelineStep('agcGain', {'window': 9}))),
        save_intermediates=False)
    result = export(backend, project, tmp_path / 'only-final.zip', final.artifact_id)
    assert result.artifact_ids == (final.artifact_id,)
    assert any('中间步骤没有保存' in warning for warning in result.warnings)


def test_empty_line_and_wrong_line_target_fail_without_creating_files(processed, tmp_path):
    backend, project, *_ = processed
    backend.projects.save_line_dataset(project.project_id, 'L02', np.ones((8, 7), np.float32))
    for line, artifact in [('L02', ''), ('L01', 'missing')]:
        job = backend.submit_processing_result_package(project.project_id, line, str(tmp_path / 'bad.zip'),
                                                       artifact_id=artifact)
        assert backend.jobs.wait(job, timeout=30).status.value == 'failed'
    assert not list(tmp_path.glob('*.zip'))


def test_unknown_time_window_roundtrips_without_fabricating_a_physical_axis(tmp_path):
    from core.gpr_data_model import GPRDataSet
    from mygpr.application.jobs.context import ExecutionContext
    from mygpr.domain.project.models import ProjectArtifact
    from mygpr.infrastructure.reporting.processing_package_export import _write_h5
    data = GPRDataSet.from_matrix('unknown', np.ones((5, 4), np.float32), time_window_ns=0,
                                 metadata={'time_axis_status': 'unknown'})
    path = tmp_path / 'unknown.h5'
    _write_h5(path, data, ProjectArtifact('saved', 'unknown', 'saved', ''), ExecutionContext.null())
    loaded = load_gpr_dataset(path, line_id='unknown')
    assert loaded.time_window_ns == 0
    np.testing.assert_array_equal(loaded.time_axis_ns, np.zeros(5))


@pytest.mark.parametrize('role', ['processing_result', 'raw_input'])
def test_h5_export_reads_large_matrix_in_bounded_tiles(tmp_path, role):
    from core.gpr_data_model import GPRDataSet
    from mygpr.application.jobs.context import ExecutionContext
    from mygpr.domain.project.models import ProjectArtifact
    from mygpr.infrastructure.reporting.processing_package_export import _write_h5
    class Matrix:
        shape, dtype = (513, 2049), np.dtype('float32')
        def __array__(self, *args, **kwargs):
            raise AssertionError('full matrix materialized')
        def __getitem__(self, slices):
            rows, cols = slices
            height, width = rows.stop - rows.start, cols.stop - cols.start
            assert height <= 256 and width <= 1024
            return np.ones((height, width), np.float32)
    data = GPRDataSet('large', Matrix(), np.arange(2049), np.arange(513), np.arange(513))
    path = tmp_path / 'large.h5'
    artifact = ProjectArtifact('saved', 'large', 'saved', '') if role == 'processing_result' else None
    _write_h5(path, data, artifact, ExecutionContext.null())
    with h5py.File(path) as handle:
        assert handle.attrs['dataset_role'] == role
        assert handle['matrix'].shape == (513, 2049)
        assert np.all(handle['matrix'][512:, 2048:] == 1)


def test_png_places_measured_distance_at_actual_coordinates(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from core.gpr_data_model import GPRDataSet
    from mygpr.application.jobs.context import ExecutionContext
    from mygpr.domain.project.models import ProjectArtifact
    from mygpr.infrastructure.reporting.processing_package_export import _write_preview
    data = GPRDataSet.from_matrix('L01', np.arange(1000, dtype=np.float32).reshape(10, 100),
                                 length_m=.99, metadata={'distance_axis_status': 'known'})
    seen = []
    original = Axes.imshow
    def imshow(self, values, *args, **kwargs):
        if np.shape(values) == (10, 100):
            seen.append(tuple(kwargs['extent'][:2]))
        return original(self, values, *args, **kwargs)
    mesh_original = Axes.pcolormesh
    def mesh(self, x, y, values, *args, **kwargs):
        if np.shape(values) == (10, 100):
            seen.append((float(x[0]), float(x[-1])))
        return mesh_original(self, x, y, values, *args, **kwargs)
    monkeypatch.setattr(Axes, 'imshow', imshow)
    monkeypatch.setattr(Axes, 'pcolormesh', mesh)
    path = tmp_path / 'preview.png'
    _write_preview(path, data, ProjectArtifact('saved', 'L01', 'saved', ''), ExecutionContext.null())
    assert path.read_bytes().startswith(b'\x89PNG')
    assert seen
    np.testing.assert_allclose(seen[0], (-.005, .995), atol=1e-6)


def test_real_qt_delivery_bridge_completes_export_in_existing_job_system(processed, qapp, tmp_path):
    from types import SimpleNamespace
    from PyQt6.QtTest import QSignalSpy
    from ui.controllers.backend_controller import JobBridge
    from ui.controllers.delivery_controller import DeliveryController
    backend, project, *_ = processed
    bridge = JobBridge(backend)
    controller = DeliveryController()
    controller.set_backend(SimpleNamespace(backend=backend, job_bridge=bridge))
    controller.set_project_context(project.project_id)
    completed = QSignalSpy(controller.processing_package_generated)
    progress = QSignalSpy(bridge.progress_changed)
    try:
        job = controller.export_processing_package(project.project_id, 'L01', str(tmp_path / 'qt.zip'))
        assert job
        assert completed or completed.wait(30000)
        result = completed[0][0]
        assert result.verified and Path(result.archive_path).is_file()
        assert progress
    finally:
        bridge.shutdown()
