from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from core.field_import_preview import build_import_preflight
from core.job_manager import JobCancelled


def test_npy_preflight_reads_shape_through_mmap_without_materialising_full_matrix(tmp_path: Path) -> None:
    source = tmp_path / "large.npy"
    matrix = np.lib.format.open_memmap(source, mode="w+", dtype=np.float32, shape=(12000, 900))
    matrix[0, 0] = -2.0
    matrix[-1, -1] = 3.0
    matrix.flush()
    del matrix

    updates: list[tuple[str, int, int]] = []
    result = build_import_preflight(
        source,
        progress_callback=lambda stage, current, total: updates.append((stage, current, total)),
    )

    assert result.can_import is True
    assert result.shape_text == "12000 × 900"
    assert "dtype=float32" in result.column_summary
    assert updates[-1][1] == updates[-1][2]


def test_large_csv_preflight_uses_bounded_sample_and_reports_shape_after_import(tmp_path: Path) -> None:
    source = tmp_path / "large.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in range(5000):
            writer.writerow([row + col / 10 for col in range(32)])

    result = build_import_preflight(source)

    assert result.can_import is True
    assert result.sample_count == 0
    assert result.trace_count == 0
    assert result.shape_text == "导入后确定"
    assert "已抽样 4096" in result.column_summary


def test_csv_preflight_honours_cooperative_cancellation(tmp_path: Path) -> None:
    source = tmp_path / "cancel.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in range(5000):
            writer.writerow([row + col for col in range(32)])

    cancelled = False

    def progress(_stage: str, current: int, _total: int) -> None:
        nonlocal cancelled
        if current > 0:
            cancelled = True

    with pytest.raises(JobCancelled):
        build_import_preflight(source, cancel_requested=lambda: cancelled, progress_callback=progress)
