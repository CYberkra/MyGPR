#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI batch recommended-profile contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def _write_altimeter_sidecar(tmp_path: Path) -> Path:
    altimeter_path = tmp_path / "altimeter.csv"
    altimeter_path.write_text(
        "timestamp_s,height_agl_m,height_source,snr,target_count,valid\n"
        "0.0,1.20,nar15,18.0,1,1\n"
        "0.7,1.40,nar15,20.0,1,1\n",
        encoding="utf-8",
    )
    return altimeter_path


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


def test_validate_config_rejects_nonfinite_numeric_params(tmp_path: Path):
    input_csv = _write_small_csv(tmp_path / "input.csv")
    cfg = {
        "jobs": [
            {
                "id": "bad-param",
                "input": str(input_csv),
                "methods": [
                    {"key": "sec_gain", "params": {"gain_max": float("nan")}},
                ],
            }
        ]
    }

    result = cli_batch.validate_config(cfg, repo_root=str(tmp_path))

    assert result.ok is False
    assert any("must be finite" in error for error in result.errors)


def test_load_gpr_csv_replaces_all_nonfinite_matrix_with_zero(tmp_path: Path):
    input_csv = tmp_path / "all_nan.csv"
    np.savetxt(
        input_csv,
        np.array([[np.nan, np.inf], [-np.inf, np.nan]], dtype=np.float32),
        delimiter=",",
    )

    data, header_info, trace_metadata = cli_batch.load_gpr_csv(str(input_csv))

    assert header_info is None
    assert trace_metadata is None
    assert np.array_equal(data, np.zeros((2, 2), dtype=float))


def test_detect_csv_header_ignores_nonfinite_and_nonpositive_shape(tmp_path: Path):
    bad_inf = tmp_path / "bad_inf.csv"
    bad_inf.write_text(
        "Number of Samples = 1e999\n"
        "Time windows (ns) = 120.0\n"
        "Number of Traces = 8\n"
        "Trace interval (m) = 1.0\n",
        encoding="utf-8",
    )
    bad_shape = tmp_path / "bad_shape.csv"
    bad_shape.write_text(
        "Number of Samples = 0\n"
        "Time windows (ns) = 120.0\n"
        "Number of Traces = 8\n"
        "Trace interval (m) = 1.0\n",
        encoding="utf-8",
    )

    assert cli_batch.detect_csv_header(str(bad_inf)) is None
    assert cli_batch.detect_csv_header(str(bad_shape)) is None


def test_cli_jsonable_removes_nonfinite_values_for_strict_summary_json():
    payload = {
        "metric": np.float64(np.inf),
        "array": np.array([1.0, np.nan, np.inf], dtype=np.float32),
        "flag": True,
    }

    safe = cli_batch._jsonable(payload)

    assert safe["metric"] is None
    assert safe["array"] == [1.0, None, None]
    assert safe["flag"] is True
    json.dumps(safe, allow_nan=False)


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


def test_resume_placeholder_returns_nonzero(capsys):
    result = cli_batch.cmd_resume(SimpleNamespace(summary="summary.json"))

    captured = capsys.readouterr()
    assert result == 2
    assert "not implemented" in captured.out
    assert "summary.json" in captured.out


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


def test_run_job_expands_motion_compensation_v2_profile(tmp_path: Path):
    input_csv = _write_airborne_csv(tmp_path / "airborne.csv")
    job = {
        "id": "motion-v2-runtime",
        "input": str(input_csv),
        "recommended_profile": "motion_compensation_v2",
    }

    result = cli_batch.run_job(job, repo_root=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert result["status"] == "ok"
    assert [step["key"] for step in result["steps"]] == ["motion_compensation_v2"]
    assert result["final_shape"] == [4, 8]


def test_run_job_expands_high_quality_uav_gpr_profile(tmp_path: Path):
    input_csv = _write_airborne_csv(tmp_path / "airborne.csv")
    job = {
        "id": "high-quality-uav",
        "input": str(input_csv),
        "recommended_profile": "high_quality_uav_gpr",
    }

    result = cli_batch.run_job(job, repo_root=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert result["status"] == "ok"
    assert [step["key"] for step in result["steps"]] == RECOMMENDED_RUN_PROFILES[
        "high_quality_uav_gpr"
    ]["order"]
    assert result["final_shape"][0] == 4
    assert "agcGain" not in [step["key"] for step in result["steps"]]
    workflow = result["profile_workflow"]
    stage2 = next(
        stage for stage in workflow["stages"] if stage["stage_key"] == "stage2"
    )
    assert workflow["profile_key"] == "high_quality_uav_gpr"
    assert stage2["method_keys"][:2] == [
        "frequency_filter_1d",
        "motion_compensation_v2",
    ]
    assert any(
        warning["method_key"] == "motion_compensation_v2"
        for warning in workflow["sensor_dependency_warnings"]
    )


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


def test_run_job_forwards_rtk_imu_altimeter_sidecars_into_motion_runtime(
    monkeypatch, tmp_path: Path
):
    input_csv = _write_airborne_csv(tmp_path / "airborne.csv")
    rtk_path, imu_path = _write_motion_sidecars(tmp_path)
    altimeter_path = _write_altimeter_sidecar(tmp_path)
    trace_timestamps_s = np.linspace(0.0, 0.7, 8, dtype=np.float64)
    seen: dict[str, np.ndarray] = {}

    def assert_sidecar_metadata(data, trace_metadata=None, **kwargs):
        assert trace_metadata is not None
        seen["height_agl_m"] = np.asarray(trace_metadata["height_agl_m"], dtype=np.float32)
        seen["height_confidence"] = np.asarray(
            trace_metadata["height_confidence"], dtype=np.float32
        )
        seen["height_source"] = np.asarray(trace_metadata["height_source"])
        seen["roll_deg"] = np.asarray(trace_metadata["roll_deg"], dtype=np.float32)
        seen["local_x_m"] = np.asarray(trace_metadata["local_x_m"], dtype=np.float32)
        seen["trace_timestamp_s"] = np.asarray(
            trace_metadata["trace_timestamp_s"], dtype=np.float64
        )
        return data, {"method": "test_cli_altimeter_runtime"}

    monkeypatch.setitem(
        cli_batch.PROCESSING_METHODS,
        "test_cli_altimeter_runtime",
        {
            "name": "test_cli_altimeter_runtime",
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
        "altimeter_path": str(altimeter_path),
        "methods": [{"key": "test_cli_altimeter_runtime"}],
    }

    validation = cli_batch.validate_config({"jobs": [job]}, repo_root=str(tmp_path))
    assert validation.ok is True

    result = cli_batch.run_job(job, repo_root=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert result["status"] == "ok"
    assert np.array_equal(seen["trace_timestamp_s"], trace_timestamps_s)
    assert seen["height_agl_m"].shape == (8,)
    assert seen["height_agl_m"][0] == np.float32(1.2)
    assert seen["height_agl_m"][-1] == np.float32(1.4)
    assert np.all(seen["height_confidence"] > 0.0)
    assert set(seen["height_source"].tolist()) == {"nar15"}
    assert seen["roll_deg"].shape == (8,)
    assert seen["local_x_m"].shape == (8,)
