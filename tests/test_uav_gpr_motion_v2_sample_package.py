#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic UAV-GPR motion V2 sample package tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import cli_batch

from scripts.generate_uav_gpr_motion_v2_sample import _jsonable, generate_package


def test_generate_uav_gpr_motion_v2_sample_package_files(tmp_path: Path):
    package_dir = tmp_path / "uav_gpr_motion_v2"
    config_path = tmp_path / "uav_gpr_motion_v2_synthetic.json"

    result = generate_package(package_dir, config_out=config_path)

    assert result.package_dir == package_dir
    assert result.config_path == config_path
    for name in [
        "main.csv",
        "rtk.csv",
        "imu.csv",
        "altimeter.csv",
        "batch_motion_v2.json",
        "README.md",
    ]:
        assert (package_dir / name).exists()

    main_lines = (package_dir / "main.csv").read_text(encoding="utf-8").splitlines()
    assert main_lines[0].startswith("Number of Samples =")
    assert main_lines[1].startswith("Time windows (ns) =")
    assert len(main_lines) > 100

    config = json.loads(config_path.read_text(encoding="utf-8"))
    job = config["jobs"][0]
    assert job["recommended_profile"] == "motion_compensation_v2"
    assert Path(job["input"]).is_absolute()
    assert Path(job["rtk_path"]).is_absolute()
    assert Path(job["imu_path"]).is_absolute()
    assert Path(job["altimeter_path"]).is_absolute()

    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    json.dumps(config, allow_nan=False)
    json.dumps(manifest, allow_nan=False)


def test_motion_v2_sample_jsonable_removes_nonfinite_values():
    payload = {
        "metric": np.float64(np.inf),
        "array": np.array([1.0, np.nan, np.inf], dtype=np.float32),
        "flag": True,
    }

    safe = _jsonable(payload)

    assert safe["metric"] is None
    assert safe["array"] == [1.0, None, None]
    assert safe["flag"] is True
    json.dumps(safe, allow_nan=False)


def test_generated_uav_gpr_motion_v2_package_validates_and_runs(tmp_path: Path):
    package_dir = tmp_path / "uav_gpr_motion_v2"
    config_path = tmp_path / "uav_gpr_motion_v2_synthetic.json"
    output_dir = tmp_path / "out"
    result = generate_package(package_dir, config_out=config_path, output_dir=output_dir)

    cfg = cli_batch.load_config(str(result.config_path))
    validation = cli_batch.validate_config(cfg, repo_root=str(tmp_path))
    assert validation.ok is True

    job = cfg["jobs"][0]
    run_result = cli_batch.run_job(
        job,
        repo_root=str(tmp_path),
        output_dir=str(output_dir),
    )

    assert run_result["status"] == "ok"
    assert [step["key"] for step in run_result["steps"]] == ["motion_compensation_v2"]
    assert run_result["final_shape"][0] > 0
    assert run_result["final_shape"][1] > 0
    assert (output_dir / job["id"] / "00_motion_compensation_v2.csv").exists()
