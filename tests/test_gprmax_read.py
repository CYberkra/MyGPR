#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for reading deterministic gprMax .out HDF5 files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from core.gpr_io import read_gprmax_out


def _write_gprmax_out(path: Path, data: np.ndarray, *, dt: float = 1e-10) -> None:
    with h5py.File(path, "w") as handle:
        rx_group = handle.create_group("rxs").create_group("rx1")
        rx_group.create_dataset("Ez", data=np.asarray(data, dtype=np.float32))
        handle.attrs["Iterations"] = int(np.asarray(data).shape[0])
        handle.attrs["dt"] = float(dt)
        handle.attrs["nx_ny_nz"] = [1, 1, 1]


def test_read_gprmax_out_merges_trace_files_in_numeric_order(tmp_path: Path):
    traces = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([4.0, 5.0, 6.0], dtype=np.float32),
        np.array([7.0, 8.0, 9.0], dtype=np.float32),
    ]
    _write_gprmax_out(tmp_path / "gpr_model_fixed10.out", traces[1])
    _write_gprmax_out(tmp_path / "gpr_model_fixed2.out", traces[0])
    _write_gprmax_out(tmp_path / "gpr_model_fixed30.out", traces[2])

    result = read_gprmax_out(str(tmp_path / "gpr_model_fixed2.out"))

    expected = np.column_stack(traces)
    assert result["data"].shape == (3, 3)
    assert np.array_equal(result["data"], expected)
    assert result["num_traces"] == 3
    assert result["samples_per_trace"] == 3
    assert result["time_step_s"] == 1e-10
    assert result["total_time_ns"] == 0.3
