import builtins
import numpy as np
import pytest

from mygpr.infrastructure.processing.algorithms.extended import trace_qc
from mygpr.interfaces.backend import MyGPRBackend
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep


def test_saved_marks_continue_after_crop_and_remove(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'qc', name='qc')
        pid = project.project_id
        data = np.arange(100, dtype=np.float32).reshape(20, 5) + 1
        backend.projects.save_line_dataset(pid, 'L01', data, length_m=4., time_window_ns=200.)
        def run(method, params, parent=''):
            return backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep(method, params),)), input_artifact_id=parent)
        def qc(index, mode='mark'):
            return {'mode': mode, 'manual_trace_indices': index, 'spike_zscore': 0, 'empty_rms_threshold': 0}
        first = run('trace_qc', qc('1'))
        second = run('trace_qc', qc('3'), first.artifact_id)
        saved = backend.projects.read_artifact_dataset(pid, 'L01', second.artifact_id)
        np.testing.assert_array_equal(saved.trace_metadata['trace_qc_bad_mask'], [0, 1, 0, 1, 0])
        crop = run('trace_cut', {'trace_start': 2, 'trace_end': 5}, second.artifact_id)
        removed = run('trace_qc', qc('', 'remove'), crop.artifact_id)
        result = backend.projects.read_artifact_dataset(pid, 'L01', removed.artifact_id)
        np.testing.assert_array_equal(result.data, data[:, [2, 4]])
        np.testing.assert_array_equal(result.header_info['distance_axis_m'], [2, 4])
    finally:
        backend.shutdown()


def test_mute_and_remove_use_prior_mask_and_preserve_mapping():
    data = np.arange(20, dtype=np.float32).reshape(4, 5) + 1
    metadata = {'trace_qc_bad_mask': np.array([0, 1, 0, 0, 0]), 'trace_id': np.arange(10, 15), 'sensor': np.arange(5) * 2}
    opts = dict(trace_metadata=metadata, manual_trace_indices='3', spike_zscore=0, header_info={'distance_axis_m': np.arange(5.)})
    muted, info = trace_qc.method_trace_qc(data, mode='mute', **opts)
    np.testing.assert_array_equal(muted[:, [1, 3]], 0)
    np.testing.assert_array_equal(info['trace_metadata_updates']['trace_qc_bad_mask'], [0, 1, 0, 1, 0])
    removed, info = trace_qc.method_trace_qc(data, mode='remove', **opts)
    np.testing.assert_array_equal(removed, data[:, [0, 2, 4]])
    np.testing.assert_array_equal(info['trace_metadata_out']['trace_id'], [10, 12, 14])
    np.testing.assert_array_equal(info['trace_metadata_out']['sensor'], [0, 4, 8])
    np.testing.assert_array_equal(info['header_info_updates']['distance_axis_m'], [0, 2, 4])


@pytest.mark.parametrize('mask', [np.zeros(4), np.zeros((5, 1))])
def test_reject_misaligned_prior_mask(mask):
    with pytest.raises(ValueError, match='mask'):
        trace_qc.method_trace_qc(np.ones((3, 5)), spike_zscore=0, trace_metadata={'trace_qc_bad_mask': mask})


def test_large_range_is_clipped_before_iteration(monkeypatch):
    def bounded_range(*args):
        result = builtins.range(*args)
        assert len(result) <= 5
        return result
    monkeypatch.setattr(trace_qc, 'range', bounded_range, raising=False)
    assert trace_qc._parse_trace_indices('0-1000000000000,4-2', 5) == [0, 1, 2, 3, 4]
