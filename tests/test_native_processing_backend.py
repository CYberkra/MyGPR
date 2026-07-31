from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from mygpr.application.jobs.cancellation import CancellationTokenSource, JobCancelledError
from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.processing.service import ProcessingService
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep, ProcessingRequest
from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor
from mygpr.infrastructure.processing.legacy_adapter import LegacyProcessingExecutor
from mygpr.infrastructure.processing.native_adapter import (
    NativeProcessingCatalog,
    NativeProcessingExecutor,
)


def _data(rows: int = 128, cols: int = 97) -> np.ndarray:
    rng = np.random.default_rng(20260722)
    y = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    return (rng.normal(0.0, 0.04, (rows, cols)) + 0.15 * y + np.sin(x * 9.0) * 0.05).astype(np.float32)


@pytest.mark.parametrize(
    ("method_id", "params", "atol"),
    [
        ("compensatingGain", {"gain_min": 1.0, "gain_max": 6.0}, 0.0),
        ("dewow", {"window": 23}, 0.0),
        ("set_zero_time", {"new_zero_time": 5.0}, 0.0),
        ("agcGain", {"window": 11}, 0.0),
        ("sec_gain", {"gain_min": 1.0, "gain_max": 6.0, "power": 1.2}, 3.0e-6),
        ("subtracting_average_2D", {"ntraces": 21}, 0.0),
        ("running_average_2D", {"ntraces": 9}, 0.0),
        ("sliding_avg", {"window_size": 10, "axis": 1}, 0.0),
        ("frequency_filter_1d", {"filter_type": "bandpass", "low_freq_mhz": 10.0, "high_freq_mhz": 80.0}, 1.0e-6),
        ("trace_median_filter", {"window_traces": 5}, 0.0),
        ("trace_savgol_filter", {"window_traces": 7, "polyorder": 2}, 3.0e-6),
    ],
)
def test_native_algorithms_match_verified_legacy_results(method_id: str, params: dict, atol: float) -> None:
    request = ProcessingRequest(
        data=_data(),
        method_id=method_id,
        params=params,
        header_info={"total_time_ns": 480.0},
    )
    native = NativeProcessingExecutor().execute(request).data
    legacy = LegacyProcessingExecutor().execute(request).data
    np.testing.assert_allclose(native, legacy, rtol=0.0, atol=atol)


class ArrayBlockSource:
    def __init__(self, data: np.ndarray, token_source: CancellationTokenSource | None = None) -> None:
        self._data = data
        self._token_source = token_source

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def dtype(self) -> str:
        return str(self._data.dtype)

    def iter_blocks(self, *, block_rows: int) -> Iterator[tuple[int, int, np.ndarray]]:
        for index, start in enumerate(range(0, self._data.shape[0], block_rows)):
            end = min(start + block_rows, self._data.shape[0])
            yield start, end, self._data[start:end]
            if index == 0 and self._token_source is not None:
                self._token_source.cancel()


def _pipeline() -> PipelineDefinition:
    return PipelineDefinition(
        name="Native block pipeline",
        steps=(
            PipelineStep("dewow", {"window": 17}),
            PipelineStep("running_average_2D", {"ntraces": 7}),
            PipelineStep("agcGain", {"window": 9}),
            PipelineStep("trace_median_filter", {"window_traces": 5}),
            PipelineStep("frequency_filter_1d", {"filter_type": "lowpass", "high_freq_mhz": 60.0}),
            PipelineStep("sec_gain", {"gain_min": 1.0, "gain_max": 4.0, "power": 1.1}),
        ),
    )


def test_file_backed_pipeline_matches_ndarray_pipeline(tmp_path: Path) -> None:
    data = _data(180, 133)
    pipeline = _pipeline()
    header = {"total_time_ns": 480.0}
    service = ProcessingService(NativeProcessingCatalog(), NativeProcessingExecutor())
    expected = service.execute_pipeline(data, pipeline, header_info=header).data
    executor = FileBackedBlockPipelineExecutor(tmp_path / "workspace", block_bytes=96 * 1024)

    def consume(matrix, summary):
        return np.array(matrix, copy=True), summary

    actual, summary = executor.execute(
        ArrayBlockSource(data),
        pipeline,
        header_info=header,
        trace_metadata={},
        consumer=consume,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=4.0e-6)
    assert len(summary.step_records) == len(pipeline.steps)
    assert all(record.implementation_version == "native-1.0" for record in summary.step_records)
    assert summary.input_sha256
    assert summary.output_sha256 == summary.step_records[-1].output_sha256
    assert list((tmp_path / "workspace").iterdir()) == []


def test_file_backed_pipeline_cancellation_removes_workspace(tmp_path: Path) -> None:
    data = _data(600, 180)
    pipeline = PipelineDefinition(steps=(PipelineStep("dewow", {"window": 15}),))
    token_source = CancellationTokenSource()
    executor = FileBackedBlockPipelineExecutor(tmp_path / "workspace", block_bytes=64 * 1024)
    context = ExecutionContext(cancellation_token=token_source.token)
    with pytest.raises(JobCancelledError):
        executor.execute(
            ArrayBlockSource(data, token_source),
            pipeline,
            header_info={"total_time_ns": 500.0},
            trace_metadata={},
            consumer=lambda matrix, summary: None,
            context=context,
        )
    assert list((tmp_path / "workspace").iterdir()) == []
