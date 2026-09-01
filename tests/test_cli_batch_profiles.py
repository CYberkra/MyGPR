#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI batch recommended-profile contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cli_batch
from mygpr.domain.processing.models import ProcessingResult


class _RecordingExecutor:
    """捕获 ProcessingRequest 并原样返回的最小执行器替身。"""

    def __init__(self) -> None:
        self.requests: list = []

    def execute(self, request, context=None):
        self.requests.append(request)
        return ProcessingResult(
            data=np.asarray(request.data, dtype=np.float32),
            method_id=request.method_id,
            params=dict(request.params),
            metadata={"method": request.method_id},
            header_info=dict(request.header_info or {}),
            trace_metadata=dict(request.trace_metadata or {}),
            runtime_warnings=[],
        )


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


def _write_altimeter_sidecar(tmp_path: Path) -> Path:
    altimeter_path = tmp_path / "altimeter.csv"
    altimeter_path.write_text(
        "timestamp_s,height_agl_m,height_source,snr,target_count,valid\n"
        "0.0,1.20,nar15,18.0,1,1\n"
        "0.7,1.40,nar15,20.0,1,1\n",
        encoding="utf-8",
    )
    return altimeter_path


def test_resolve_job_methods_requires_explicit_methods():
    """0.9.38 起预设档移除：job 必须显式给出 methods。"""
    import pytest

    with pytest.raises(ValueError, match="methods"):
        cli_batch._resolve_job_methods({})


def test_validate_config_rejects_removed_recommended_profile(tmp_path: Path):
    """预设档已移除：recommended_profile 字段必须报错，防止旧配置静默跑错链。"""
    input_csv = _write_small_csv(tmp_path / "input.csv")
    cfg = {
        "jobs": [
            {
                "id": "legacy-profile",
                "input": str(input_csv),
                "recommended_profile": "wavelet_2d_denoise",
            }
        ]
    }

    result = cli_batch.validate_config(cfg, repo_root=str(tmp_path))

    assert result.ok is False
    assert result.errors, "recommended_profile 应报错而非被忽略"


def test_bundled_cli_batch_mvp_example_validates():
    config_path = Path(cli_batch.BASE_DIR) / "config" / "cli_batch_mvp_example.json"
    cfg = cli_batch.load_config(str(config_path))

    result = cli_batch.validate_config(cfg, repo_root=str(Path(cli_batch.BASE_DIR)))

    assert result.ok is True
    assert result.errors == []


def test_summary_paths_are_unique_within_same_second(tmp_path: Path):
    first = cli_batch._build_summary_path(str(tmp_path))
    second = cli_batch._build_summary_path(str(tmp_path))

    assert first != second
    assert Path(first).parent == tmp_path
    assert Path(first).name.startswith("summary_")
    assert Path(first).suffix == ".json"


def test_resume_returns_ok_when_summary_has_no_failed_jobs(tmp_path: Path, capsys):
    summary = tmp_path / "summary.json"
    config = tmp_path / "config.json"
    config.write_text('{"jobs": []}', encoding="utf-8")
    summary.write_text(
        json.dumps(
            {
                "config": str(config),
                "output_dir": str(tmp_path / "out"),
                "results": [{"job_id": "job-ok", "status": "ok"}],
            }
        ),
        encoding="utf-8",
    )

    result = cli_batch.cmd_resume(SimpleNamespace(summary=str(summary), repo_root=str(tmp_path)))

    captured = capsys.readouterr()
    assert result == 0
    assert "no failed jobs" in captured.out


def test_resume_missing_summary_returns_nonzero(tmp_path: Path, capsys):
    result = cli_batch.cmd_resume(SimpleNamespace(summary=str(tmp_path / "missing.json"), repo_root=str(tmp_path)))

    captured = capsys.readouterr()
    assert result == 2
    assert "summary file not found" in captured.out


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
    assert any("recommended_profile" in error for error in result.errors)


def test_run_job_runs_explicit_wavelet_methods(tmp_path: Path):
    input_csv = _write_small_csv(tmp_path / "input.csv")
    job = {
        "id": "wavelet-job",
        "input": str(input_csv),
        "methods": [
            {"key": "dewow", "params": {"window": 3}},
            {"key": "wavelet_2d", "params": {"levels": 2}},
        ],
    }

    result = cli_batch.run_job(job, repo_root=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert [step["key"] for step in result["steps"]] == ["dewow", "wavelet_2d"]
    assert result["status"] == "ok"


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


def test_run_job_expands_motion_compensation_v2_profile(tmp_path: Path):
    input_csv = _write_airborne_csv(tmp_path / "airborne.csv")
    job = {
        "id": "motion-v2-runtime",
        "input": str(input_csv),
        "methods": [{"key": "motion_compensation_v2", "params": {}}],
    }

    result = cli_batch.run_job(job, repo_root=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert result["status"] == "ok"
    assert [step["key"] for step in result["steps"]] == ["motion_compensation_v2"]
    assert result["final_shape"] == [4, 8]


def test_run_job_forwards_rtk_imu_sidecars_into_motion_runtime(monkeypatch, tmp_path: Path):
    input_csv = _write_airborne_csv(tmp_path / "airborne.csv")
    rtk_path, imu_path = _write_motion_sidecars(tmp_path)
    trace_timestamps_s = np.linspace(0.0, 0.7, 8, dtype=np.float64)
    seen: dict[str, np.ndarray] = {}

    recording = _RecordingExecutor()
    monkeypatch.setattr(cli_batch, "_NATIVE_EXECUTOR", recording)
    monkeypatch.setitem(
        cli_batch.PROCESSING_METHODS,
        "test_cli_sidecar_runtime",
        {
            "name": "test_cli_sidecar_runtime",
            "type": "native",
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
    assert len(recording.requests) == 1
    trace_metadata = recording.requests[0].trace_metadata
    assert trace_metadata is not None
    seen["roll_deg"] = np.asarray(trace_metadata["roll_deg"], dtype=np.float32)
    seen["local_x_m"] = np.asarray(trace_metadata["local_x_m"], dtype=np.float32)
    seen["trace_timestamp_s"] = np.asarray(
        trace_metadata["trace_timestamp_s"], dtype=np.float64
    )
    assert np.array_equal(seen["trace_timestamp_s"], trace_timestamps_s)
    assert seen["roll_deg"].shape == (8,)
    assert seen["local_x_m"].shape == (8,)


def test_run_job_forwards_rtk_imu_altimeter_sidecars_into_motion_runtime(
    monkeypatch, tmp_path: Path
):
    input_csv = _write_airborne_csv(tmp_path / "airborne.csv")
    rtk_path, imu_path = _write_motion_sidecars(tmp_path)
    altimeter_path = _write_altimeter_sidecar(tmp_path)
    trace_timestamps_s = np.linspace(0.0, 0.7, 8, dtype=np.float64)

    recording = _RecordingExecutor()
    monkeypatch.setattr(cli_batch, "_NATIVE_EXECUTOR", recording)
    monkeypatch.setitem(
        cli_batch.PROCESSING_METHODS,
        "test_cli_altimeter_runtime",
        {
            "name": "test_cli_altimeter_runtime",
            "type": "native",
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
        "altimeter_path": str(altimeter_path),
        "methods": [{"key": "test_cli_altimeter_runtime"}],
    }

    validation = cli_batch.validate_config({"jobs": [job]}, repo_root=str(tmp_path))
    assert validation.ok is True

    result = cli_batch.run_job(job, repo_root=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert result["status"] == "ok"
    assert len(recording.requests) == 1
    trace_metadata = recording.requests[0].trace_metadata
    assert trace_metadata is not None
    assert np.array_equal(
        np.asarray(trace_metadata["trace_timestamp_s"], dtype=np.float64),
        trace_timestamps_s,
    )
    height_agl_m = np.asarray(trace_metadata["height_agl_m"], dtype=np.float32)
    assert height_agl_m.shape == (8,)
    assert height_agl_m[0] == np.float32(1.2)
    assert height_agl_m[-1] == np.float32(1.4)
    assert np.all(np.asarray(trace_metadata["height_confidence"]) > 0.0)
    assert set(np.asarray(trace_metadata["height_source"]).tolist()) == {"nar15"}
    assert np.asarray(trace_metadata["roll_deg"]).shape == (8,)
    assert np.asarray(trace_metadata["local_x_m"]).shape == (8,)
