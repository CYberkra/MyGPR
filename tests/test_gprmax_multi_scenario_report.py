#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the multi-scenario gprMax HTML report runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.gprmax_benchmark import gprmax_multi_scenario_report as report
from scripts.gprmax_benchmark.gprmax_multi_scenario_report import (
    build_gprmax_command,
    build_scenario_definitions,
    render_html_report,
)


def test_multi_scenario_definitions_cover_simple_validation_cases():
    scenarios = build_scenario_definitions()

    assert {
        "cylinder_single_v1",
        "cylinder_double_v1",
        "layered_interface_v1",
    } <= set(scenarios)

    for scenario_id, definition in scenarios.items():
        assert definition.scenario_id == scenario_id
        assert definition.label
        assert definition.description
        assert definition.structure_notes
        assert "#title:" in definition.model_in_text
        assert "#time_window:" in definition.model_in_text
        assert definition.targets


def test_build_gprmax_command_exposes_parallel_flags(tmp_path: Path):
    command = build_gprmax_command(
        tmp_path / "python.exe",
        tmp_path / "model.in",
        runs=36,
        geometry_fixed=True,
        mpi=4,
        gpu=["0"],
        extra_args=["--write-processed"],
    )

    assert command[:3] == [str(tmp_path / "python.exe"), "-m", "gprMax"]
    assert "-n" in command and "36" in command
    assert "--geometry-fixed" in command
    assert "-mpi" in command and "4" in command
    assert "-gpu" in command and "0" in command
    assert "--write-processed" in command


def test_build_gprmax_command_omits_parallel_flags_by_default(tmp_path: Path):
    command = build_gprmax_command(
        tmp_path / "python.exe",
        tmp_path / "model.in",
        runs=12,
    )

    assert "-mpi" not in command
    assert "-gpu" not in command
    assert "--geometry-fixed" not in command


def test_report_pipeline_uses_agc_gain_instead_of_sec_gain():
    pipeline = report._resolve_pipeline("uav_gpr_experience_baseline_v1")

    assert "agcGain" in pipeline
    assert "sec_gain" not in pipeline
    assert pipeline == [
        "set_zero_time",
        "dewow",
        "subtracting_average_2D",
        "agcGain",
        "svd_subspace",
    ]


def test_zero_time_align_policy_uses_auto_params_for_both_branches(monkeypatch):
    def fake_pipeline(profile_key: str):
        return ["set_zero_time"]

    def fake_manual_params(pipeline, profile_key):
        return {"set_zero_time": {"new_zero_time": 5.0}}

    def fake_auto_tune(*args, **kwargs):
        return {
            "method_key": "set_zero_time",
            "method_name": "零时矫正",
            "recommended_params": {"new_zero_time": 0.0},
            "best_params": {"new_zero_time": 0.0},
            "best_score": 1.0,
            "best_reason": "测试自动零时",
        }

    monkeypatch.setattr(report, "_resolve_pipeline", fake_pipeline)
    monkeypatch.setattr(report, "_resolve_manual_params", fake_manual_params)
    monkeypatch.setattr(report, "auto_tune_method", fake_auto_tune)

    data = np.arange(40, dtype=np.float32).reshape(10, 4)
    comparison = report.run_stepwise_comparison(
        data,
        header_info={"total_time_ns": 10.0},
        trace_metadata={},
        ground_truth={
            "scenario_id": "demo",
            "analysis_roi": {
                "time_start_idx": 2,
                "time_end_idx": 8,
                "dist_start_idx": 0,
                "dist_end_idx": 4,
            },
            "targets": [{"type": "hyperbola"}],
        },
        baseline_profile_key="uav_gpr_experience_baseline_v1",
        search_mode="fast",
        zero_time_policy="align_auto",
    )

    step = comparison["steps"][0]
    assert step["manual_original_params"] == {"new_zero_time": 5.0}
    assert step["manual_params"] == {"new_zero_time": 0.0}
    assert step["auto_params"] == {"new_zero_time": 0.0}
    assert step["policy_notes"]
    assert "不判定人工/自动优劣" in step["analysis"]["visual"]
    assert comparison["zero_time_policy"] == "align_auto"


def test_render_html_report_contains_required_research_sections(tmp_path: Path):
    payload = {
        "gprmax_root": "E:/gprMax/gprMax-v.3.1.7",
        "python_executable": "E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe",
        "baseline_profile_key": "uav_gpr_experience_baseline_v1",
        "search_mode": "fast",
        "zero_time_policy": "align_auto",
        "run_settings": {
            "runs": 36,
            "geometry_fixed": True,
            "mpi": None,
            "gpu": [],
        },
        "acceleration_support": {"mpi4py": False, "cupy": False},
        "scenarios": [
            {
                "scenario_id": "demo_scene",
                "label": "测试场景",
                "description": "用于验证 HTML 合同。",
                "structure_notes": ["真实结构说明"],
                "structure_preview": "assets/structure.png",
                "gprmax": {"command": ["python", "-m", "gprMax", "model.in"]},
                "comparison": {
                    "verdict": "auto_better",
                    "metrics": {
                        "manual": {"comparison_score": 1.0},
                        "auto": {"comparison_score": 1.2},
                        "delta_auto_minus_manual": {
                            "comparison_score": 0.2,
                            "low_freq_energy_reduction": 0.1,
                            "target_band_energy_ratio": 0.05,
                            "edge_preservation": 0.03,
                        },
                    },
                },
                "images": [
                    {
                        "method_key": "dewow",
                        "method_name": "低频漂移抑制",
                        "manual_params": {"window": 61},
                        "auto_params": {"window": 31},
                        "auto_tune_summary": {"best_reason": "评分最高"},
                        "analysis": {
                            "visual": "自动选参在视觉上保留目标。",
                            "metrics": "score 差值为 0.2。",
                        },
                        "manual_warnings": [],
                        "auto_warnings": [],
                        "images": {
                            "manual_input": "assets/manual_before.png",
                            "auto_input": "assets/auto_before.png",
                            "manual_output": "assets/manual_after.png",
                            "auto_output": "assets/auto_after.png",
                        },
                    }
                ],
            }
        ],
    }

    html_path = render_html_report(tmp_path, payload)
    html_text = html_path.read_text(encoding="utf-8")

    assert "真实地质结构" in html_text
    assert "人工选参" in html_text
    assert "自动选参" in html_text
    assert "逐步骤 BScan 对比" in html_text
    assert "gprMax 运行设置" in html_text
    assert "视觉评价" in html_text
    assert "指标评价" in html_text
