from __future__ import annotations

import numpy as np

from core.gpr_data_model import GPRDataSet


def test_preview_window_slices_before_downsampling_and_returns_source_indices() -> None:
    matrix = np.arange(120 * 240, dtype=np.float32).reshape(120, 240)
    dataset = GPRDataSet.from_matrix("L01", matrix, length_m=239.0, time_window_ns=119.0)

    preview, sample_indices, trace_indices = dataset.preview_window(
        sample_start=20,
        sample_end=80,
        trace_start=50,
        trace_end=170,
        max_samples=30,
        max_traces=40,
    )

    assert sample_indices.tolist() == list(range(20, 80, 2))
    assert trace_indices.tolist() == list(range(50, 170, 3))
    assert preview.shape == (30, 40)
    assert preview[0, 0] == matrix[20, 50]
    assert preview[-1, -1] == matrix[78, 167]
