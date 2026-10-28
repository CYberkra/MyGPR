"""CSV processing uses the same inclusive axis displayed by its reader."""
from pathlib import Path

import numpy as np
import pytest

from core.gpr_data_model import load_gpr_dataset


def write_line(path: Path):
    indices = np.arange(501, dtype=np.float64)
    trace = np.sin(2 * np.pi * 50 * indices / 501) + .25 * np.cos(2 * np.pi * 100 * indices / 501)
    matrix = np.column_stack([trace, trace * 2, trace * 3]).astype(np.float32)
    lines = ['Number of Samples = 501', 'Time windows (ns) = 900',
             'Number of Traces = 3', 'Trace interval (m) = 0']
    lines.extend(f',,,{value:.9g},' for value in matrix.T.ravel())
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return matrix


def test_reader_declares_its_displayed_inclusive_axis(tmp_path):
    path = tmp_path / 'line.csv'
    write_line(path)
    data = load_gpr_dataset(path)
    assert data.metadata['time_axis_endpoint'] is True
    assert data.metadata['time_step_s'] == pytest.approx(1.8e-9)
    assert data.metadata['time_axis_convention'] == 'csv_reader_inclusive_interpretation'
    assert data.to_metadata()['time_axis_endpoint'] is True
    np.testing.assert_allclose(data.time_axis_ns, np.arange(501) * 1.8, rtol=1e-6)


@pytest.mark.parametrize('save_intermediates', [True, False])
@pytest.mark.parametrize('method', ['time_cut', 'frequency_filter_1d'])
def test_project_processing_uses_csv_reader_sampling(tmp_path, save_intermediates, method):
    from mygpr.application.jobs.models import JobStatus
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    from mygpr.interfaces.backend import MyGPRBackend

    source = tmp_path / 'line.csv'
    original = write_line(source)
    if method == 'time_cut':
        params = dict(mode='keep_range', time_start_ns=18., time_end_ns=720.)
        expected = original[10:400]
        expected_time = np.arange(10, 400) * 1.8
    else:
        # Correct inclusive dt keeps bin 50 at 55.4447 MHz; window/N rejects it.
        params = dict(filter_type='lowpass', high_freq_mhz=55.5, taper_ratio=0.)
        frequencies = np.fft.rfftfreq(501, d=1.8e-9)
        expected = np.fft.irfft(np.fft.rfft(original, axis=0) *
                                (frequencies <= 55.5e6)[:, None], n=501, axis=0)
        expected_time = np.arange(501) * 1.8
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='CSV sampling')
        backend.projects.import_line_source(project.project_id, 'L01', source)
        job = backend.submit_project_pipeline(project.project_id, 'L01',
            PipelineDefinition(name='sampling', steps=(PipelineStep(method, params),)),
            save_intermediates=save_intermediates)
        snapshot = backend.jobs.wait(job, timeout=30)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        artifact_id = snapshot.result.artifact_id
        result = backend.projects.read_artifact_dataset(project.project_id, 'L01', artifact_id)
        np.testing.assert_allclose(result.data, expected, rtol=1e-6, atol=5e-7)
        np.testing.assert_allclose(result.header_info['time_axis_ns'], expected_time, rtol=1e-6, atol=1e-4)
        assert result.header_info['time_step_s'] == pytest.approx(1.8e-9)
        backend.projects.close_project(project.project_id)
        reopened = backend.projects.open_project(project.root_path)
        restored = backend.projects.read_artifact_dataset(reopened.project_id, 'L01', artifact_id)
        np.testing.assert_array_equal(restored.data, result.data)
        np.testing.assert_array_equal(restored.header_info['time_axis_ns'], result.header_info['time_axis_ns'])
    finally:
        backend.shutdown()
