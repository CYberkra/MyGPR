from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from mygpr.application.processing.service import ProcessingService
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep, ProcessingRequest
from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor
from mygpr.infrastructure.processing.native_adapter import NativeProcessingCatalog, NativeProcessingExecutor

from core.processing_engine import (
    clone_header_info,
    clone_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)


def _legacy_kernel(request: ProcessingRequest) -> np.ndarray:
    """复刻旧 LegacyProcessingExecutor 的执行路径（core kernel 直调，等价证据用）。"""
    prepared = prepare_runtime_params(
        request.method_id,
        request.params,
        clone_header_info(request.header_info),
        clone_trace_metadata(request.trace_metadata),
        request.data.shape,
    )
    output, _ = run_processing_method(request.data, request.method_id, prepared)
    return np.asarray(output)


def _matrix(rows: int = 96, cols: int = 72) -> np.ndarray:
    rng = np.random.default_rng(2077)
    t = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, cols, dtype=np.float32)[None, :]
    background = 0.5 * np.sin(5.0 * t) @ np.ones((1, cols), dtype=np.float32)
    reflector = 0.3 * np.exp(-((t - (0.45 + 0.08 * x**2)) ** 2) / 0.002)
    return np.asarray(background + reflector + rng.normal(0.0, 0.02, (rows, cols)), dtype=np.float32)


@pytest.mark.parametrize(
    ("method_id", "params", "atol"),
    [
        ("svd_bg", {"rank": 2, "solver": "exact"}, 4.0e-6),
        ("svd_subspace", {"rank_start": 2, "rank_end": 8, "solver": "exact"}, 4.0e-6),
        ("fk_filter", {"angle_low": 10, "angle_high": 60, "taper_width": 5}, 5.0e-6),
        ("stolt_migration", {"dx": 0.05, "dt": 0.1, "v": 0.10, "pad_x": 0, "pad_t": 0}, 8.0e-5),
        ("rpca_background", {"lam": 0.08, "max_iter": 8, "tol": 1e-5, "svd_solver": "exact"}, 8.0e-5),
        ("hankel_svd", {"window_length": 24, "rank": 3, "aggressiveness": 0.5}, 5.0e-5),
    ],
)
def test_native_global_methods_match_historical_kernels(
    method_id: str,
    params: dict[str, object],
    atol: float,
) -> None:
    request = ProcessingRequest(data=_matrix(), method_id=method_id, params=params)
    native = NativeProcessingExecutor().execute(request)
    legacy = _legacy_kernel(request)
    np.testing.assert_allclose(native.data, legacy, rtol=0.0, atol=atol)
    assert native.metadata["implementation_version"] == "native-global-1.0"


def test_large_svd_uses_deterministic_randomized_solver() -> None:
    request = ProcessingRequest(
        data=_matrix(240, 180),
        method_id="svd_bg",
        params={
            "rank": 3,
            "solver": "auto",
            "exact_max_elements": 1_000,
            "random_seed": 11,
            "power_iterations": 2,
        },
    )
    executor = NativeProcessingExecutor()
    first = executor.execute(request)
    second = executor.execute(request)
    assert first.metadata["solver"] == "randomized"
    np.testing.assert_allclose(first.data, second.data, rtol=0.0, atol=0.0)


class ArraySource:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    @property
    def dtype(self) -> str:
        return str(self.matrix.dtype)

    def iter_blocks(self, *, block_rows: int) -> Iterator[tuple[int, int, np.ndarray]]:
        for start in range(0, self.matrix.shape[0], block_rows):
            end = min(start + block_rows, self.matrix.shape[0])
            yield start, end, self.matrix[start:end]


def test_global_pipeline_uses_file_backed_staging(tmp_path: Path) -> None:
    data = _matrix(140, 110)
    pipeline = PipelineDefinition(
        name="Global file-backed pipeline",
        steps=(
            PipelineStep("svd_bg", {"rank": 1, "solver": "exact"}),
            PipelineStep("fk_filter", {"angle_low": 12, "angle_high": 55, "taper_width": 4}),
        ),
    )
    executor = FileBackedBlockPipelineExecutor(tmp_path / "workspace", block_bytes=80 * 1024)
    assert executor.supports(pipeline)
    estimate = executor.estimate(data.shape, str(data.dtype), pipeline)
    assert estimate.supports_chunking
    assert estimate.relative_cost == "high"

    def consume(matrix, summary):
        return np.array(matrix, copy=True), summary

    actual, summary = executor.execute(
        ArraySource(data),
        pipeline,
        header_info={},
        trace_metadata={},
        consumer=consume,
    )
    service = ProcessingService(NativeProcessingCatalog(), NativeProcessingExecutor())
    expected = service.execute_pipeline(data, pipeline).data
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=8.0e-5)
    assert [record.implementation_version for record in summary.step_records] == [
        "native-global-1.0",
        "native-global-1.0",
    ]
    assert list((tmp_path / "workspace").iterdir()) == []
