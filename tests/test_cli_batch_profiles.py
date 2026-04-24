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
    assert methods[-1]["params"] == {"window_length": 48, "rank": 4}


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
