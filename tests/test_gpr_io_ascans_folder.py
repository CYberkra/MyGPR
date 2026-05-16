#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for folder-based A-scan ingestion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.gpr_io import read_ascans_folder


def _write_ascan(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows), encoding="utf-8")


def test_read_ascans_folder_skips_malformed_rows_without_shifting_samples(tmp_path: Path):
    _write_ascan(
        tmp_path / "trace_0001.csv",
        [
            "time,amplitude",
            "0.0,10.0",
            "1.0,20.0",
            "2.0,30.0",
        ],
    )
    _write_ascan(
        tmp_path / "trace_0002.csv",
        [
            "time,amplitude",
            "0.0,40.0",
            "bad,not-a-number",
            "1.0,50.0",
            "2.0,60.0",
        ],
    )

    result = read_ascans_folder(str(tmp_path))

    assert result["data"].shape == (3, 2)
    assert np.array_equal(
        result["data"],
        np.array(
            [
                [10.0, 40.0],
                [20.0, 50.0],
                [30.0, 60.0],
            ],
            dtype=np.float32,
        ),
    )
