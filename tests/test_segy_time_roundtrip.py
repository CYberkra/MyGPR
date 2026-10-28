"""SEG-Y 2.1 sample times and coordinates survive export and reading."""
import struct

import numpy as np
import pytest

from core.gpr_vendor_readers import read_segy_fixed
from mygpr.application.project.segy_export import write_segy


@pytest.mark.parametrize('step_ns,start_ns', [(10.0, 100.0), (2000.0, 0.0)])
def test_segy_preserves_real_time_and_positions(tmp_path, step_ns, start_ns):
    matrix = np.arange(20, dtype=np.float32).reshape(5, 4)
    time = start_ns + np.arange(5) * step_ns
    distance = np.array([0, .25, 1.125, 3.])
    path = write_segy(tmp_path / 'result.sgy', matrix, time_axis_ns=time, trace_positions=distance)
    raw = path.read_bytes()
    assert raw[3500:3502] == bytes((2, 1))
    assert struct.unpack_from('>d', raw, 3272)[0] == pytest.approx(step_ns / 1000)
    assert struct.unpack_from('>H', raw, 3502)[0] == 1
    result = read_segy_fixed(path)
    np.testing.assert_array_equal(result['data'], matrix)
    np.testing.assert_allclose(result['header_info']['time_axis_ns'], time, atol=1e-9)
    np.testing.assert_allclose(result['header_info']['distance_axis_m'], distance, atol=1e-9)
    assert result['header_info']['sample_interval_us'] == pytest.approx(step_ns / 1000)


def test_segy_rejects_unrepresentable_time_axis_before_writing(tmp_path):
    for axis in ([0., 10., 21.], [1., 11., 21.]):
        path = tmp_path / 'invalid.sgy'
        with pytest.raises(ValueError, match='时间|时基'):
            write_segy(path, np.ones((3, 2)), time_axis_ns=axis)
        assert not path.exists()


def test_public_artifact_export_uses_saved_cropped_time_axis(tmp_path):
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    from mygpr.interfaces.backend import MyGPRBackend
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='SEG-Y')
        matrix = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
        backend.projects.save_line_dataset(project.project_id, 'L01', matrix,
                                           length_m=3.0, time_window_ns=200.0)
        artifact = backend.project_processing.execute_pipeline(project.project_id, 'L01',
            PipelineDefinition(steps=(PipelineStep('time_cut', {
                'mode': 'keep_range', 'time_start_ns': 100., 'time_end_ns': 200.}),)))
        source = backend.projects.read_artifact_dataset(project.project_id, 'L01', artifact.artifact_id)
        path = backend.projects.export_artifact_segy(project.project_id, 'L01', artifact.artifact_id,
                                                     tmp_path / 'crop.sgy')
        result = read_segy_fixed(path)
        np.testing.assert_array_equal(result['data'], source.data)
        np.testing.assert_allclose(result['header_info']['time_axis_ns'], source.header_info['time_axis_ns'])
        np.testing.assert_allclose(result['header_info']['distance_axis_m'], source.header_info['distance_axis_m'])
    finally:
        backend.shutdown()


def test_float32_gpr_time_axis_is_not_mistaken_for_nonuniform(tmp_path):
    axis = np.linspace(0, 900, 501, dtype=np.float32)
    matrix = np.ones((501, 2), dtype=np.float32)
    path = write_segy(tmp_path / 'gpr.sgy', matrix, time_axis_ns=axis)
    result = read_segy_fixed(path)
    np.testing.assert_allclose(result['header_info']['time_axis_ns'], axis, rtol=0, atol=0.00004)


def test_float32_millimetre_coordinates_roundtrip_without_rounding_arbitrary_values(tmp_path):
    axis = np.array([0., 10., 20.], dtype=np.float32)
    coords = np.array([10.1, 10.2], dtype=np.float32)
    path = write_segy(tmp_path / 'coords.sgy', np.ones((3, 2)), time_axis_ns=axis, trace_positions=coords)
    np.testing.assert_array_equal(read_segy_fixed(path)['header_info']['distance_axis_m'].astype('float32'), coords)
    with pytest.raises(ValueError, match='坐标'):
        write_segy(tmp_path / 'invalid.sgy', np.ones((3, 2)), time_axis_ns=axis,
                   trace_positions=np.array([10.1004, 10.2], dtype=np.float32))
