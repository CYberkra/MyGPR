from __future__ import annotations

import numpy as np

from core.io_performance import (
    choose_csv_read_dtype,
    csv_import_context,
    sanitize_float32_matrix,
    summarize_array_memory,
)


def test_choose_csv_read_dtype_matrix_only_uses_float32():
    assert choose_csv_read_dtype(header_info=None, has_sidecars=False) == "float32"


def test_choose_csv_read_dtype_preserves_airborne_or_sidecar_precision():
    assert choose_csv_read_dtype(header_info={"a_scan_length": 2}, has_sidecars=False) is None
    assert choose_csv_read_dtype(header_info=None, has_sidecars=True) is None


def test_sanitize_float32_matrix_replaces_nonfinite_and_reports_memory():
    arr, summary = sanitize_float32_matrix([[1.0, np.nan], [np.inf, 4.0]])
    assert arr.dtype == np.float32
    assert arr.shape == (2, 2)
    assert np.isfinite(arr).all()
    assert summary["nonfinite_replaced"] == 2
    assert summary["dtype"] == "float32"
    assert summary["nbytes"] == arr.nbytes


def test_summarize_array_memory_reports_contiguous_float32():
    arr = np.zeros((5, 7), dtype=np.float32)
    summary = summarize_array_memory(arr).to_dict()
    assert summary["shape"] == (5, 7)
    assert summary["is_float32"] is True
    assert summary["is_c_contiguous"] is True


def test_csv_import_context_is_serialisable(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("1,2\n3,4\n", encoding="utf-8")
    ctx = csv_import_context(str(path), header_info=None)
    assert ctx["has_header"] is False
    assert ctx["has_sidecars"] is False
    assert ctx["pandas_read_dtype"] == "float32"
    assert ctx["file_size_bytes"] > 0
