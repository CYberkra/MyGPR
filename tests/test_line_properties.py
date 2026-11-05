"""Measured scale is project metadata and never rewrites B-scan samples."""
import numpy as np
import pytest
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


def test_time_calibration_preserves_source_and_existing_result(tmp_path):
    source = tmp_path / 'source.npy'
    np.save(source, np.arange(1280, dtype=np.float32).reshape(64, 20))
    original = source.read_bytes()
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='time calibration')
        pid = project.project_id
        backend.projects.import_line_source(pid, 'L01', source)
        report = backend.maintenance.list_quality_reports(pid)[0]
        assert report.time_axis_status == 'unknown'
        assert report.distance_axis_status == 'unknown'
        assert backend.projects.read_dataset(pid, 'L01').header_info['time_axis_status'] == 'unknown'
        session = backend.processing_workbench.open_session(pid, 'L01', restore_draft=False)
        with pytest.raises(ValueError, match='时间轴未知'):
            backend.processing_workbench.analyze_trace(session.session_id, 0)
        artifact = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(
            steps=(PipelineStep('dewow', {'window': 3}),)))
        backend.projects.update_line_properties(pid, 'L01', 20.0, False, 100.0)
        report = backend.maintenance.list_quality_reports(pid)[0]
        assert report.time_window_ns == pytest.approx(100.0)
        assert report.length_m == pytest.approx(20.0)
        assert report.time_axis_status == 'known'
        assert report.distance_axis_status == 'known'
        raw = backend.projects.read_dataset(pid, 'L01')
        assert raw.header_info['time_axis_status'] == 'known'
        assert raw.header_info['time_step_s'] == pytest.approx(100e-9 / 63)
        session = backend.processing_workbench.open_session(pid, 'L01', restore_draft=False)
        analysis = backend.processing_workbench.analyze_trace(session.session_id, 0)
        assert analysis.sample_axis[-1] == pytest.approx(100.0)
        assert analysis.frequency_axis[1] == pytest.approx(1 / (64 * (100e-9 / 63)) / 1e6)
        assert backend.projects.read_artifact_dataset(pid, 'L01', artifact.artifact_id).header_info['time_axis_status'] == 'unknown'
        assert source.read_bytes() == original
        backend.projects.close_project(pid)
        reopened = backend.projects.open_project(project.root_path)
        pid = reopened.project_id
        assert backend.projects.get_line(pid, 'L01').time_window_override_ns == 100.0
        assert backend.projects.read_dataset(pid, 'L01').header_info['time_axis_status'] == 'known'
        backend.projects.update_line_properties(pid, 'L01', 20.0, False, 0.0)
        assert backend.projects.read_dataset(pid, 'L01').header_info['time_axis_status'] == 'unknown'
        assert source.read_bytes() == original
    finally:
        backend.shutdown()
