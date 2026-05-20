#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gprMax simulation validity package auditing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.gprmax_benchmark.audit_gprmax_package import audit_gprmax_package


def _write_demo_package(root: Path) -> None:
    (root / "scenario.json").write_text(
        json.dumps(
            {
                "schema": "mygpr_gprmax_scenario_v1",
                "scenario_id": "demo",
                "source": {"kind": "synthetic_reference"},
                "simulation": {
                    "sample_count": 20,
                    "trace_count": 5,
                    "time_step_s": 1e-9,
                    "total_time_ns": 20.0,
                },
                "domain_m": [0.10, 0.08, 0.002],
                "dx_dy_dz_m": [0.002, 0.002, 0.002],
                "antenna": {
                    "source_position_m": [0.012, 0.04, 0.0],
                    "receiver_position_m": [0.082, 0.04, 0.0],
                    "source_step_m": [0.002, 0.0, 0.0],
                    "receiver_step_m": [0.004, 0.0, 0.0],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "model.in").write_text(
        "\n".join(
            [
                "#domain: 0.100 0.080 0.002",
                "#dx_dy_dz: 0.002 0.002 0.002",
                "#time_window: 2.0e-08",
                "#hertzian_dipole: z 0.012 0.040 0 my_ricker",
                "#rx: 0.082 0.040 0",
                "#src_steps: 0.002 0 0",
                "#rx_steps: 0.004 0 0",
            ]
        ),
        encoding="utf-8",
    )
    np.savetxt(root / "mygpr_bscan.csv", np.ones((20, 5)), delimiter=",")
    (root / "ground_truth.json").write_text(
        json.dumps(
            {
                "schema": "mygpr_gprmax_ground_truth_v1",
                "scenario_id": "demo",
                "targets": [
                    {
                        "target_id": "t0",
                        "roi": {
                            "time_start_idx": 4,
                            "time_end_idx": 8,
                            "dist_start_idx": 1,
                            "dist_end_idx": 3,
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_audit_detects_synthetic_reference_and_pml_risk(tmp_path: Path):
    _write_demo_package(tmp_path)

    audit = audit_gprmax_package(tmp_path)

    assert audit["source"]["kind"] == "synthetic_reference"
    assert audit["source"]["native_gprmax_verified"] is False
    assert audit["files"]["raw_out_exists"] is False
    assert audit["shape"]["shape_matches_expected"] is True
    assert audit["ground_truth"]["roi_inside_bscan"] is True
    assert audit["paper_usable"] is False
    assert any("synthetic_reference" in item for item in audit["warnings"])
    assert any("PML margin" in item for item in audit["geometry"]["pml_margin"]["risk_flags"])


def test_repository_cylinder_single_v1_is_smoke_only():
    audit = audit_gprmax_package("sample_data/gprmax_benchmarks/cylinder_single_v1")

    assert audit["source"]["kind"] == "synthetic_reference"
    assert audit["files"]["raw_out_exists"] is False
    assert audit["source"]["dt_source"] == "synthetic fallback/scenario metadata"
    assert audit["ground_truth"]["roi_inside_bscan"] is True
    assert audit["paper_usable"] is False
