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
    assert result["data"].dtype == np.float32
    assert np.array_equal(result["data"], expected)
    assert result["num_traces"] == 3
    assert result["samples_per_trace"] == 3
    assert result["time_step_s"] == 1e-10
    assert result["total_time_ns"] == 0.3


def test_read_gprmax_out_ignores_unrelated_trace_files(tmp_path: Path):
    _write_gprmax_out(tmp_path / "pipe_model1.out", np.array([1.0, 2.0, 3.0]))
    _write_gprmax_out(tmp_path / "pipe_model2.out", np.array([4.0, 5.0, 6.0]))
    _write_gprmax_out(tmp_path / "other_model1.out", np.array([9.0, 9.0, 9.0]))

    result = read_gprmax_out(str(tmp_path / "pipe_model1.out"))

    assert result["data"].shape == (3, 2)
    assert np.array_equal(
        result["data"],
        np.array(
            [
                [1.0, 4.0],
                [2.0, 5.0],
                [3.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )


def test_read_gprmax_out_matches_in_file_by_trace_prefix(tmp_path: Path):
    _write_gprmax_out(tmp_path / "pipe_model1.out", np.array([1.0, 2.0, 3.0]))
    _write_gprmax_out(tmp_path / "pipe_model2.out", np.array([4.0, 5.0, 6.0]))
    (tmp_path / "aaa_other.in").write_text(
        "\n".join(
            [
                "#title: other",
                "#waveform: impulse 1 1.0 my_impulse",
                "#src_steps: 0.200 0.000 0.000",
                "#rx_steps: 0.200 0.000 0.000",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "pipe_model.in").write_text(
        "\n".join(
            [
                "#title: pipe_model",
                "#waveform: impulse 1 1.0 my_impulse",
                "#src_steps: 0.070 0.000 0.000",
                "#rx_steps: 0.070 0.000 0.000",
            ]
        ),
        encoding="utf-8",
    )

    result = read_gprmax_out(str(tmp_path / "pipe_model1.out"))

    assert result["in_path"] == str(tmp_path / "pipe_model.in")
    assert result["header_info"]["trace_interval_m"] == 0.07
    assert np.allclose(result["trace_metadata"]["trace_distance_m"], [0.0, 0.07])


def test_read_gprmax_empty_merged_out_falls_back_to_related_trace_files(tmp_path: Path):
    with h5py.File(tmp_path / "pipe_model_merged.out", "w") as handle:
        handle.attrs["Iterations"] = 3
        handle.attrs["dt"] = 1e-10
        handle.attrs["nx_ny_nz"] = [1, 1, 1]
    _write_gprmax_out(tmp_path / "pipe_model1.out", np.array([1.0, 2.0, 3.0]))
    _write_gprmax_out(tmp_path / "pipe_model2.out", np.array([4.0, 5.0, 6.0]))
    _write_gprmax_out(tmp_path / "other_model1.out", np.array([9.0, 9.0, 9.0]))

    result = read_gprmax_out(str(tmp_path / "pipe_model_merged.out"))

    assert result["data"].shape == (3, 2)
    assert np.array_equal(
        result["data"],
        np.array(
            [
                [1.0, 4.0],
                [2.0, 5.0],
                [3.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )


def test_read_gprmax_out_attaches_impulse_context_from_matching_in(tmp_path: Path):
    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    _write_gprmax_out(tmp_path / "air_test_merged.out", data, dt=2e-10)
    (tmp_path / "air_test.in").write_text(
        "\n".join(
            [
                "#title: air_test",
                "#domain: 1.000 0.500 0.010",
                "#dx_dy_dz: 0.010 0.010 0.010",
                "#time_window: 8.000e-10",
                "#waveform: impulse 1 1.0 my_impulse",
                "#hertzian_dipole: z 0.100 0.200 0.100 my_impulse",
                "#rx: 0.100 0.300 0.100",
                "#src_steps: 0.050 0.000 0.000",
                "#rx_steps: 0.050 0.000 0.000",
            ]
        ),
        encoding="utf-8",
    )

    result = read_gprmax_out(str(tmp_path / "air_test_merged.out"))

    header = result["header_info"]
    assert header["source"] == "gprmax_out"
    assert header["data_context"] == "gprmax_impulse"
    assert header["frequency_filter_policy"] == "model_or_auto_tune_only"
    assert "frequency_filter_band_mhz" not in header
    assert header["default_processing_profile"] == "gprmax_impulse_validation"
    assert header["trace_interval_m"] == 0.05
    assert result["trace_metadata"] is not None
    assert np.allclose(result["trace_metadata"]["trace_distance_m"], [0.0, 0.05, 0.1])
