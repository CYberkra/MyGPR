"""File-backed processing must use the same physical time basis as loaded processing."""
from pathlib import Path

import numpy as np
import pytest

from mygpr.domain.processing.models import PipelineDefinition, PipelineStep, ProcessingRequest
from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor


class _ArraySource:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.dtype = str(data.dtype)

    def iter_blocks(self, *, block_rows):
        for start in range(0, self.shape[0], block_rows):
            end = min(start + block_rows, self.shape[0])
            yield start, end, self.data[start:end]


def _execute_blocks(tmp_path, data, method, params, header):
    executor = FileBackedBlockPipelineExecutor(tmp_path / 'work', block_bytes=65536)
    return executor.execute(
        _ArraySource(data),
        PipelineDefinition(steps=(PipelineStep(method, params),)),
        header_info=header,
        trace_metadata={},
        consumer=lambda matrix, summary: (np.array(matrix, copy=True), summary),
    )


@pytest.mark.parametrize('header', [
    {'total_time_ns': 900., 'time_axis_endpoint': True},
    {'time_step_s': 1.8e-9},
    {'total_time_ns': 900., 'time_step_s': 1.8e-9},
], ids=['inclusive-endpoint', 'explicit-step-only', 'explicit-step-with-window'])
def test_block_zero_time_respects_physical_sampling_interval(tmp_path: Path, header):
    data = np.tile(np.arange(501, dtype=np.float32)[:, None], (1, 7))
    params = {'new_zero_time': 17.99}
    loaded = NativeProcessingExecutor().execute(ProcessingRequest(
        data, 'set_zero_time', params, header))
    blocked, summary = _execute_blocks(tmp_path, data, 'set_zero_time', params, header)
    expected = np.zeros_like(data)
    expected[:-9] = data[9:]
    np.testing.assert_array_equal(loaded.data, expected)
    np.testing.assert_array_equal(blocked, expected)
    assert summary.step_records[0].metadata['shift_samples'] == 9


@pytest.mark.parametrize('header', [
    {'total_time_ns': 900., 'time_axis_endpoint': True},
    {'time_step_s': 1.8e-9},
    {'total_time_ns': 900., 'time_step_s': 1.8e-9},
], ids=['inclusive-endpoint', 'explicit-step-only', 'explicit-step-with-window'])
def test_block_filter_respects_physical_sampling_interval(tmp_path: Path, header):
    # Bin 90 is 99.8004 MHz at 1.8 ns/point, below the 99.9 MHz high-pass cutoff.
    # Dividing the inclusive 900 ns window by N incorrectly puts it at 100 MHz.
    tone = np.sin(2 * np.pi * 90 * np.arange(501) / 501)
    data = np.tile(tone.astype(np.float32)[:, None], (1, 7))
    params = {'filter_type': 'highpass', 'low_freq_mhz': 99.9, 'taper_ratio': 0.}
    loaded = NativeProcessingExecutor().execute(ProcessingRequest(
        data, 'frequency_filter_1d', params, header))
    blocked, summary = _execute_blocks(tmp_path, data, 'frequency_filter_1d', params, header)
    np.testing.assert_allclose(loaded.data, 0., atol=1e-6)
    np.testing.assert_allclose(blocked, loaded.data, atol=1e-6)
    assert summary.step_records[0].metadata['sample_rate_hz'] == pytest.approx(1 / 1.8e-9)
