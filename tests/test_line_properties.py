"""Measured scale is project metadata and never rewrites B-scan samples."""
import numpy as np
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def test_line_properties_measure_restore_unknown_and_preserve_artifact(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='properties')
        pid = project.project_id
        matrix = np.arange(24, dtype=np.float32).reshape(6, 4)
        backend.projects.save_line_dataset(pid, 'L01', matrix, time_window_ns=60)
        artifact = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(
            steps=(PipelineStep('dewow', {'window': 3}),)))
        before = backend.projects.read_artifact_dataset(pid, 'L01', artifact.artifact_id)
        backend.projects.update_line_properties(pid, 'L01', length_m=12.0, view_reversed=True)
        info = backend.projects.get_dataset_info(pid, 'L01')
        assert info.metadata['distance_axis_status'] == 'measured'
        np.testing.assert_allclose(info.metadata['distance_axis_m'], [0, 4, 8, 12])
        assert info.metadata['view_reversed'] is True
        np.testing.assert_array_equal(backend.projects.read_dataset(pid, 'L01').data, matrix)
        after = backend.projects.read_artifact_dataset(pid, 'L01', artifact.artifact_id)
        np.testing.assert_array_equal(after.header_info['distance_axis_m'], before.header_info['distance_axis_m'])
        measured = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(
            steps=(PipelineStep('dewow', {'window': 3}),)))
        backend.projects.update_line_properties(pid, 'L01', length_m=None, view_reversed=False)
        measured_data = backend.projects.read_artifact_dataset(pid, 'L01', measured.artifact_id)
        np.testing.assert_allclose(measured_data.header_info['distance_axis_m'], [0, 4, 8, 12])
        info = backend.projects.get_dataset_info(pid, 'L01')
        assert info.length_m == 0
        assert info.metadata['distance_axis_status'] == 'unknown'
        assert info.metadata['view_reversed'] is False
    finally:
        backend.shutdown()


def test_properties_preserve_acquisition_ids_and_survive_reopen(tmp_path):
    from tests.test_cdut_streaming_hdf5_import import write_stream
    source = tmp_path / 'source.h5'
    waveform, _ = write_stream(source)
    original_bytes = source.read_bytes()
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='properties')
        pid = project.project_id
        backend.projects.import_line_source(pid, 'L01', source)
        before = backend.projects.read_trace_metadata(pid, 'L01')['trace_id'].copy()
        backend.projects.update_line_properties(pid, 'L01', 20., True)
        raw = backend.projects.read_dataset(pid, 'L01')
        np.testing.assert_array_equal(raw.data, waveform.T)
        np.testing.assert_array_equal(raw.trace_metadata['trace_id'], before)
        assert source.read_bytes() == original_bytes
        backend.projects.close_project(pid)
        reopened = backend.projects.open_project(project.root_path)
        info = backend.projects.get_dataset_info(reopened.project_id, 'L01')
        np.testing.assert_allclose(info.metadata['distance_axis_m'], [0, 5, 10, 15, 20])
        assert info.metadata['view_reversed'] is True
    finally:
        backend.shutdown()
