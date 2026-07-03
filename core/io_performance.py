# -*- coding: utf-8 -*-
"""Lightweight I/O and memory performance helpers.

The functions in this module are deliberately side-effect free.  They are used
by the GUI loading path to report memory pressure and by tests to keep import
behaviour deterministic.  They do not change processing arrays or numerical
algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayMemorySummary:
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    nbytes_mb: float
    is_float32: bool
    is_c_contiguous: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_array_memory(array: Any) -> ArrayMemorySummary:
    """Return a compact memory summary for a numpy-like array."""
    arr = np.asarray(array)
    return ArrayMemorySummary(
        shape=tuple(int(v) for v in arr.shape),
        dtype=str(arr.dtype),
        nbytes=int(arr.nbytes),
        nbytes_mb=float(arr.nbytes / (1024.0 * 1024.0)),
        is_float32=arr.dtype == np.float32,
        is_c_contiguous=bool(arr.flags.c_contiguous),
    )


def choose_csv_read_dtype(*, header_info: dict | None, has_sidecars: bool) -> str | None:
    """Choose a conservative pandas read dtype for the CSV import path.

    Matrix-only CSV files are safe to ingest as float32 because MyGPR processing
    arrays are float32.  Airborne stacked CSV files and sidecar-integrated CSV
    files can carry longitude/latitude/timestamp fields, so they remain on
    pandas' default numeric inference and are converted later by the existing
    parser.
    """
    if header_info is None and not has_sidecars:
        return "float32"
    return None


def csv_import_context(
    path: str,
    *,
    header_info: dict | None,
    trace_timestamps_s: Any = None,
    rtk_path: str | None = None,
    imu_path: str | None = None,
    altimeter_path: str | None = None,
) -> dict[str, Any]:
    """Return a small, serialisable import context for logs and perf reports."""
    has_sidecars = any(
        value is not None
        for value in (trace_timestamps_s, rtk_path, imu_path, altimeter_path)
    )
    dtype = choose_csv_read_dtype(header_info=header_info, has_sidecars=has_sidecars)
    try:
        file_size = int(os.path.getsize(path))
    except OSError:
        file_size = 0
    return {
        "path": str(path),
        "file_size_bytes": file_size,
        "file_size_mb": float(file_size / (1024.0 * 1024.0)),
        "has_header": bool(header_info),
        "has_sidecars": bool(has_sidecars),
        "pandas_read_dtype": dtype or "infer",
    }


def sanitize_float32_matrix(data: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert to float32 2-D matrix and replace non-finite values.

    Returns the converted array plus a compact audit dictionary.  This centralises
    repeated import-path sanitisation without changing the existing fill policy.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    replaced = 0
    fill_value = 0.0
    if not np.isfinite(arr).all():
        finite_mask = np.isfinite(arr)
        replaced = int(arr.size - int(np.count_nonzero(finite_mask)))
        fill_value = float(np.mean(arr[finite_mask])) if finite_mask.any() else 0.0
        arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    summary = summarize_array_memory(arr).to_dict()
    summary.update({"nonfinite_replaced": replaced, "fill_value": fill_value})
    return arr, summary
