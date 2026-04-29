#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI batch recommended-profile contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import cli_batch
from core.preset_profiles import RECOMMENDED_RUN_PROFILES


def _write_small_csv(path: Path) -> Path:
    rows, cols = 48, 16
    t = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    data = np.repeat(np.sin(2.0 * np.pi * 3.0 * t), cols, axis=1)
    data[:, 5] += 0.05
    np.savetxt(path, data, delimiter=",")
    return path


def _write_airborne_csv(path: Path) -> Path:
    samples = 4
    traces = 8
    header_lines = [
        f"Number of Samples = {samples}",
        "Time windows (ns) = 120.0",
        f"Number of Traces = {traces}",
        "Trace interval (m) = 1.0",
    ]
    amplitudes = np.arange(samples * traces, dtype=np.float32).reshape(traces, samples)
    longitude = np.array(
        [0.0, 0.00001, 0.00002, 0.00003, 0.00008, 0.00014, 0.00021, 0.00029]
    )
    latitude = np.array(
        [30.0, 30.00015, 30.00005, 30.00020, 30.00008, 30.00025, 30.00010, 30.00028]
    )
    ground = np.linspace(100.0, 101.4, traces, dtype=np.float32)
    flight_height = np.linspace(12.0, 12.7, traces, dtype=np.float32)

    with path.open("w", encoding="utf-8", newline="") as handle:
        for line in header_lines:
            handle.write(f"{line}\n")
        for trace_idx in range(traces):
            for sample_idx in range(samples):
                handle.write(
                    "{:.8f},{:.8f},{:.3f},{:.6f},{:.3f}\n".format(
                        longitude[trace_idx],
                        latitude[trace_idx],
                        float(ground[trace_idx]),
                        float(amplitudes[trace_idx, sample_idx]),
                        float(flight_height[trace_idx]),
                    )
                )
    return path


def _write_motion_sidecars(tmp_path: Path) -> tuple[Path, Path]:
    rtk_path = tmp_path / "rtk.csv"
    rtk_path.write_text(
        "timestamp_s,longitude,latitude\n"
        "0.0,0.00000000,30.00000000\n"
        "0.7,0.00029000,30.00028000\n",
        encoding="utf-8",
    )
    imu_path = tmp_path / "imu.csv"
    imu_path.write_text(
        "timestamp_s,roll_deg,pitch_deg,yaw_deg\n"
        "0.0,0.0,0.0,180.0\n"
        "0.7,7.0,3.5,187.0\n",
        encoding="utf-8",
    )
    return rtk_path, imu_path


def test_resolve_job_methods_uses_recommended_profile_defaults():
    methods = cli_batch._resolve_job_methods(
        {
            "recommended_profile": "hankel_denoise",
        }
    )

    assert [step["key"] for step in methods] == RECOMMENDED_RUN_PROFILES[
        "hankel_denoise"
    ]["order"]
    assert methods[-1]["key"] == "hankel_svd"
    assert methods[-1]["params"] == {"window_length": 0, "rank": 0}


def test_validate_config_accepts_recommended_profile_job(tmp_path: Path):
    input_csv = _write_small_csv(tmp_path / "input.csv")
    cfg = {
        "jobs": [
            {
                "id": "wavelet-job",
                "input": str(input_csv),
                "recommended_profile": "wavelet_2d_denoise",
            }
        ]
    }

    result = cli_batch.validate_config(cfg, repo_root=str(tmp_path))

    assert result.ok is True
    assert result.errors == []


def test_validate_config_rejects_unknown_recommended_profile(tmp_path: Path):
    input_csv = _write_small_csv(tmp_path / "input.csv")
    cfg = {
        "jobs": [
            {
                "id": "bad-profile",
                "input": str(input_csv),
                "recommended_profile": "does_not_exist",
            }
        ]
    }

    result = cli_batch.validate_config(cfg, repo_root=str(tmp_path))

    assert result.ok is False
    assert any("unknown recommended_profile" in error for error in result.errors)


def test_run_job_expands_recommended_profile_into_steps(tmp_path: Path):
    input_csv = _write_small_csv(tmp_path / "input.csv")
    job = {
        "id": "wavelet-job",
        "input": str(input_csv),
        "recommended_profile": "wavelet_2d_denoise",
    }

    result = cli_batch.run_job(job, repo_root=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert [step["key"] for step in result["steps"]] == RECOMMENDED_RUN_PROFILES[
        "wavelet_2d_denoise"
    ]["order"]
    assert result["status"] == "ok"
    assert result["final_shape"] == [48, 16]


def test_run_job_uses_runtime_metadata_merge_for_motion_local_methods(tmp_path: Path):
    input_csv = _write_airborne_csv(tmp_path / "airborne.csv")
    job = {
        "id": "motion-local-runtime",
        "input": str(input_csv),
        "methods": [
            {
                "key": "trajectory_smoothing",
                "params": {"method": "savgol", "window_length": 5, "polyorder": 2},
            },
            {
                "key": "motion_compensation_speed",
                "params": {"spacing_m": 2.5},
            },
        ],
    }

    result = cli_batch.run_job(job, repo_root=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert result["status"] == "ok"
    assert [step["key"] for step in result["steps"]] == [
        "trajectory_smoothing",
        "motion_compensation_speed",
    ]
    assert result["steps"][0]["shape"] == [4, 8]
    assert result["steps"][1]["shape"][0] == 4
    assert result["steps"][1]["shape"][1] > result["steps"][0]["shape"][1]
    assert result["final_shape"] == result["steps"][1]["shape"]


def test_run_job_forwards_rtk_imu_sidecars_into_motion_runtime(monkeypatch, tmp_path: Path):
    input_csv = _write_airborne_csv(tmp_path / "airborne.csv")
    rtk_path, imu_path = _write_motion_sidecars(tmp_path)
    trace_timestamps_s = np.linspace(0.0, 0.7, 8, dtype=np.float64)
    seen: dict[str, np.ndarray] = {}

    def assert_sidecar_metadata(data, trace_metadata=None, **kwargs):
        assert trace_metadata is not None
        seen["roll_deg"] = np.asarray(trace_metadata["roll_deg"], dtype=np.float32)
        seen["local_x_m"] = np.asarray(trace_metadata["local_x_m"], dtype=np.float32)
        seen["trace_timestamp_s"] = np.asarray(
            trace_metadata["trace_timestamp_s"], dtype=np.float64
        )
        return data, {"method": "test_cli_sidecar_runtime"}

    monkeypatch.setitem(
        cli_batch.PROCESSING_METHODS,
        "test_cli_sidecar_runtime",
        {
            "name": "test_cli_sidecar_runtime",
            "type": "local",
            "func": assert_sidecar_metadata,
            "params": [],
            "auto_tune_family": "motion_comp",
        },
    )

    job = {
        "id": "motion-sidecar-runtime",
        "input": str(input_csv),
        "trace_timestamps_s": trace_timestamps_s.tolist(),
        "rtk_path": str(rtk_path),
        "imu_path": str(imu_path),
        "methods": [{"key": "test_cli_sidecar_runtime"}],
    }

    result = cli_batch.run_job(job, repo_root=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert result["status"] == "ok"
    assert np.array_equal(seen["trace_timestamp_s"], trace_timestamps_s)
    assert seen["roll_deg"].shape == (8,)
    assert seen["local_x_m"].shape == (8,)
