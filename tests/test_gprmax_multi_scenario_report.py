#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the multi-scenario gprMax HTML report runner."""

from __future__ import annotations

from pathlib import Path

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


def test_render_html_report_contains_required_research_sections(tmp_path: Path):
    payload = {
        "gprmax_root": "E:/gprMax/gprMax-v.3.1.7",
        "python_executable": "E:/gprMax/gprMax-v.3.1.7/.venv/Scripts/python.exe",
        "baseline_profile_key": "uav_gpr_experience_baseline_v1",
        "search_mode": "fast",
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
