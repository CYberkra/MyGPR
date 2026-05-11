#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for motion_compensation_v2 benchmark HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.gprmax_benchmark import motion_compensation_v2_benchmark_report as report


def test_synthetic_motion_v2_report_writes_html_json_and_images(tmp_path: Path):
    payload = report.run_motion_v2_benchmark_report(
        output_root=tmp_path,
        runs=12,
        sample_count=64,
        total_time_ns=18.0,
    )

    html_path = Path(payload["artifacts"]["html"])
    summary_path = Path(payload["artifacts"]["summary_json"])
    assert html_path.exists()
    assert summary_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "motion_compensation_v2 Benchmark" in html_text
    assert "补偿前后 B-scan" in html_text
    assert "质量告警" in html_text
    assert "max_shift_ns" in html_text

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["report_type"] == "motion_compensation_v2_benchmark"
    assert summary["source"]["kind"] == "synthetic_airborne_motion_case"
    assert summary["input_summary"]["shape"] == [64, 12]
    assert summary["motion_meta"]["height_correction_applied"] is True
    assert summary["motion_meta"]["max_shift_limit_source"].startswith("max_shift_ns")
    assert "raw_bscan" in summary["artifacts"]["images"]
    assert (html_path.parent / summary["artifacts"]["images"]["raw_bscan"]).exists()


def test_source_scenario_dir_report_loads_bscan_and_sidecars(tmp_path: Path):
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    bscan = np.zeros((32, 5), dtype=np.float32)
    bscan[8, :] = 1.0
    np.savetxt(scenario_dir / "mygpr_bscan.csv", bscan, delimiter=",", fmt="%.6f")
    (scenario_dir / "scenario.json").write_text(
        json.dumps(
            {
                "scenario_id": "airborne_height_variation_cylinder_v1",
                "label": "height variation",
                "simulation": {"total_time_ns": 18.0, "trace_step_m": 0.1},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_text(
        scenario_dir / "trace_timestamps.csv",
        "trace_idx,timestamp_s\n0,0.0\n1,0.1\n2,0.2\n3,1.2\n4,1.3\n",
    )
    _write_text(
        scenario_dir / "rtk.csv",
        "timestamp_s,local_x_m,local_y_m,latitude_deg,longitude_deg\n"
        "0.0,0.00,0.0,30.0,104.0\n"
        "0.1,0.10,0.0,30.0,104.0\n"
        "0.2,0.20,0.0,30.0,104.0\n"
        "1.2,2.00,0.0,30.0,104.0\n"
        "1.3,2.10,0.0,30.0,104.0\n",
    )
    _write_text(
        scenario_dir / "imu.csv",
        "timestamp_s,roll_deg,pitch_deg,yaw_deg\n"
        "0.0,0,0,0\n0.1,0,0,0\n0.2,0,0,0\n1.2,0,0,0\n1.3,0,0,0\n",
    )
    _write_text(
        scenario_dir / "altimeter.csv",
        "timestamp_s,flight_height_m,snr,target_count,valid\n"
        "0.0,0.11,20,1,1\n"
        "0.1,0.12,20,1,1\n"
        "0.2,0.13,20,1,1\n"
        "1.2,0.14,20,1,1\n"
        "1.3,0.15,20,1,1\n",
    )

    payload = report.run_motion_v2_benchmark_report(
        output_root=tmp_path / "out",
        source_scenario_dir=scenario_dir,
        save_images_flag=False,
    )

    assert payload["source"]["kind"] == "gprmax_airborne_scenario_dir"
    assert payload["source"]["scenario_id"] == "airborne_height_variation_cylinder_v1"
    assert payload["input_summary"]["shape"] == [32, 5]
    assert payload["motion_meta"]["height_source_used"] == "height_agl_m"
    assert "trace_timestamp_gap" in payload["quality_flags"]
    assert "trace_distance_gap" in payload["quality_flags"]
    assert Path(payload["artifacts"]["html"]).exists()


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
