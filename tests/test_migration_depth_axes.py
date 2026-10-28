import numpy as np
import pytest
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


@pytest.mark.parametrize('method,params', [
    ('rtm_migration', {'v': .1, 'depth_m': .8, 'dx_m': .1, 'dz_m': .1, 'boundary_width': 2}),
    ('kirchhoff_migration', {'freq': 5e7, 'depth': .4, 'v': .1, 'weight': 0., 'length_m': 1.5, 'time_window_ns': 12., 'backend': 'cpu'}),
])
def test_migration_depth_axes_and_unknown_time(tmp_path, method, params):
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        pid = backend.projects.create_project(tmp_path / 'project', name='migration').project_id
        raw = np.random.default_rng(1).normal(size=(48, 16)).astype(np.float32)
        backend.projects.save_line_dataset(pid, 'L01', raw, length_m=1.5, time_window_ns=12.)
        art = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep(method, params),)))
        data = backend.projects.read_artifact_dataset(pid, 'L01', art.artifact_id)
        header = data.header_info
        step = params['dz_m'] if method == 'rtm_migration' else .3 / 47
        np.testing.assert_allclose(header['depth_axis_m'], np.arange(data.data.shape[0]) * step, atol=1e-6)
        assert header['time_axis_status'] == 'unknown'
        with pytest.raises(ValueError):
            backend.projects.export_artifact_segy(pid, 'L01', art.artifact_id, tmp_path / 'bad.sgy')
        with pytest.raises(ValueError):
            backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('time_cut', {'mode':'remove_above', 'time_start_ns': 1}),)), input_artifact_id=art.artifact_id)
    finally:
        backend.shutdown()

@pytest.mark.parametrize('save_intermediates', [False, True])
def test_unknown_migration_time_rejects_frequency_filter(tmp_path, save_intermediates):
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        pid = backend.projects.create_project(tmp_path / 'project', name='migration').project_id
        backend.projects.save_line_dataset(pid, 'L01', np.ones((48, 16), np.float32), length_m=1.5, time_window_ns=12.)
        art = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('rtm_migration', {'v': .1, 'depth_m': .8, 'dx_m': .1, 'dz_m': .1, 'boundary_width': 2}),)))
        with pytest.raises(ValueError, match='时间轴'):
            backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('frequency_filter_1d', {}),)), input_artifact_id=art.artifact_id, save_intermediates=save_intermediates)
    finally:
        backend.shutdown()


def test_elevation_output_refuses_unverified_axis():
    from mygpr.infrastructure.persistence.field_project_adapter import _validated_output_header
    with pytest.raises(ValueError, match='高程'):
        _validated_output_header({'is_elevation': True}, (4, 3), method_id='kirchhoff_migration')
