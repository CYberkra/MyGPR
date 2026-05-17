#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the multi-scenario gprMax HTML report runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.gprmax_benchmark import gprmax_multi_scenario_report as report
from scripts.gprmax_benchmark.gprmax_multi_scenario_report import (
    AIRBORNE_GEOMETRY,
    build_gprmax_command,
    build_scenario_definitions,
    find_out_files,
    render_html_report,
    write_gprmax_inputs,
)


def test_default_scenario_definitions_cover_airborne_uav_gpr_cases():
    scenarios = build_scenario_definitions()

    assert {
        "airborne_single_cylinder_v1",
        "airborne_hyperbola_demo_v1",
        "airborne_rough_soil_hyperbola_v1",
        "airborne_double_cylinder_v1",
        "airborne_layered_interface_v1",
        "airborne_air_crack_v1",
        "airborne_no_target_background_v1",
        "airborne_height_variation_cylinder_v1",
    } <= set(scenarios)
    assert "legacy_surface_coupled_single_cylinder_v1" not in scenarios

    for scenario_id, definition in scenarios.items():
        assert definition.scenario_id == scenario_id
        assert definition.label
        assert definition.description
        assert definition.structure_notes
        assert "#title:" in definition.model_in_text
        assert "#time_window:" in definition.model_in_text
        assert "#geometry_view:" in definition.model_in_text
        assert definition.geometry_model == "airborne_2d_tmz_v1"
        assert definition.is_uav_gpr_evidence is True
        assert definition.source_start_m[1] > definition.ground_top_y_m
        assert definition.receiver_start_m[1] > definition.ground_top_y_m
        if scenario_id == "airborne_no_target_background_v1":
            assert definition.targets == []
        else:
            assert definition.targets


def test_legacy_scenarios_are_explicitly_named_and_not_default():
    legacy = build_scenario_definitions("legacy")
    all_scenarios = build_scenario_definitions("all")

    assert "legacy_surface_coupled_single_cylinder_v1" in legacy
    assert "airborne_single_cylinder_v1" in all_scenarios
    assert "legacy_surface_coupled_single_cylinder_v1" in all_scenarios
    assert legacy["legacy_surface_coupled_single_cylinder_v1"].is_uav_gpr_evidence is False
    assert "非 UAV-GPR" in legacy["legacy_surface_coupled_single_cylinder_v1"].description


def test_airborne_geometry_satisfies_gprmax_safety_margins():
    scenarios = build_scenario_definitions()

    assert AIRBORNE_GEOMETRY.domain_m[2] == AIRBORNE_GEOMETRY.dx_m
    assert AIRBORNE_GEOMETRY.top_clearance_cells >= 20

    for definition in scenarios.values():
        assert not definition.geometry_warnings
        assert definition.domain_m[2] == definition.dx_m
        positions = AIRBORNE_GEOMETRY.trace_positions(definition.default_runs)
        min_x = min(
            min(float(pos["source"][0]), float(pos["receiver"][0]))  # type: ignore[index]
            for pos in positions
        )
        max_x = max(
            max(float(pos["source"][0]), float(pos["receiver"][0]))  # type: ignore[index]
            for pos in positions
        )
        assert min_x >= 0.06
        assert definition.domain_m[0] - max_x >= 0.06


def test_hyperbola_demo_centers_target_within_default_airborne_aperture():
    scenario = build_scenario_definitions()["airborne_hyperbola_demo_v1"]
    target = scenario.targets[0]
    positions = AIRBORNE_GEOMETRY.trace_positions(scenario.default_runs)
    midpoints = np.asarray(
        [
            (float(pos["source"][0]) + float(pos["receiver"][0])) / 2.0  # type: ignore[index]
            for pos in positions
        ],
        dtype=np.float64,
    )
    target_x = float(target["center_m"][0])

    assert abs(target_x - float(midpoints.mean())) <= AIRBORNE_GEOMETRY.trace_step_m
    assert target_x - float(midpoints.min()) >= 0.18
    assert float(midpoints.max()) - target_x >= 0.18
    assert float(target["center_m"][1]) <= scenario.ground_top_y_m - 0.08
    assert scenario.antenna_height_m == pytest.approx(0.08)
    assert scenario.source_start_m[1] == pytest.approx(
        scenario.ground_top_y_m + scenario.antenna_height_m
    )
    assert scenario.targets[0]["radius_m"] == pytest.approx(0.01)
    assert "#material: 3 0.0002 1 0 very_dry_sand" in scenario.model_in_text
    assert "#cylinder: 0.370 0.185 0 0.370 0.185 0.002 0.010 pec" in (
        scenario.model_in_text
    )


def test_rough_soil_hyperbola_uses_deterministic_y_axis_roughness():
    scenario = build_scenario_definitions()["airborne_rough_soil_hyperbola_v1"]
    model_lines = scenario.model_in_text.splitlines()
    ground_boxes = [
        line
        for line in model_lines
        if line.startswith("#box:") and line.endswith("dry_silty_sand")
    ]

    assert scenario.targets[0]["target_id"] == "airborne_rough_soil_metal_cylinder"
    assert scenario.targets[0]["radius_m"] == pytest.approx(0.012)
    assert len(ground_boxes) == 18
    assert "#fractal_box:" not in scenario.model_in_text
    assert "#add_surface_roughness:" not in scenario.model_in_text
    assert "#material: 8 0.006 1 0 weak_wet_patch" in scenario.model_in_text
    assert "#material: 1.5 0 1 0 weak_air_patch" in scenario.model_in_text
    assert "#cylinder: 0.370 0.175 0 0.370 0.175 0.002 0.012 pec" in (
        scenario.model_in_text
    )
    assert scenario.layers[0]["kind"] == "segmented_rough_surface"
    assert scenario.layers[0]["segment_count"] == 18


def test_airborne_gprmax_inputs_distinguish_fixed_and_height_varying_geometry(tmp_path: Path):
    scenarios = build_scenario_definitions()

    fixed = scenarios["airborne_single_cylinder_v1"]
    fixed_inputs = write_gprmax_inputs(fixed, tmp_path / "fixed", runs=4)
    fixed_text = fixed_inputs[0].read_text(encoding="utf-8")
    assert len(fixed_inputs) == 1
    assert "#src_steps:" in fixed_text
    assert "#rx_steps:" in fixed_text
    assert "#geometry_view:" in fixed_text

    variable = scenarios["airborne_height_variation_cylinder_v1"]
    variable_dir = tmp_path / "variable"
    variable_dir.mkdir()
    variable_inputs = write_gprmax_inputs(variable, variable_dir, runs=4)
    assert len(variable_inputs) == 1
    text = variable_inputs[0].read_text(encoding="utf-8")
    assert "#src_steps:" not in text
    assert "#rx_steps:" not in text
    assert "#geometry_view:" in text
    assert "#python:" in text
    assert "current_model_run" in text
    assert "height_m = 0.12 + 0.035 * sin" in text
    assert "hertzian_dipole('z', source_x, antenna_y, 0.0, 'my_ricker')" in text
    assert "rx(receiver_x, antenna_y, 0.0)" in text


def test_airborne_synthetic_sidecars_are_generated_for_motion_compensation(tmp_path: Path):
    scenario = build_scenario_definitions()["airborne_height_variation_cylinder_v1"]
    sidecars = report.write_synthetic_sidecars(scenario, tmp_path, runs=6)

    assert {"trace_timestamps", "rtk", "imu", "altimeter"} <= set(sidecars)
    for path in sidecars.values():
        assert path.exists()
        assert len(path.read_text(encoding="utf-8").splitlines()) == 7

    metadata = report.build_synthetic_trace_metadata(scenario, trace_count=6)
    assert "flight_height_m" in metadata
    assert len(set(np.round(metadata["flight_height_m"], 5))) > 1
    assert metadata["trace_timestamp_s"].shape == (6,)


def test_new_scenario_ground_truth_contracts_are_explicit():
    scenarios = build_scenario_definitions()
    crack = scenarios["airborne_air_crack_v1"]
    crack_truth = report.build_ground_truth(
        crack,
        {
            "sample_count": 128,
            "trace_count": 36,
            "total_time_ns": 12.0,
        },
    )
    assert crack_truth["targets"][0]["type"] == "air_crack"
    assert crack_truth["targets"][0]["expected_features"] == [
        "narrow_vertical_reflector",
        "weak_diffraction_edges",
    ]
    crack_roi = crack_truth["targets"][0]["roi"]
    assert 0 <= crack_roi["time_start_idx"] < crack_roi["time_end_idx"] <= 128
    assert 0 <= crack_roi["dist_start_idx"] < crack_roi["dist_end_idx"] <= 36
    assert {
        "direct_air_wave",
        "air_ground_reflection",
        "subsurface_target",
        "background",
        "late_noise",
    } <= set(crack_truth["wavefield_rois"])
    direct = crack_truth["wavefield_rois"]["direct_air_wave"]["roi"]
    surface = crack_truth["wavefield_rois"]["air_ground_reflection"]["roi"]
    target = crack_truth["wavefield_rois"]["subsurface_target"]["roi"]
    assert direct["time_start_idx"] < surface["time_start_idx"] < target["time_start_idx"]

    no_target = scenarios["airborne_no_target_background_v1"]
    no_target_truth = report.build_ground_truth(
        no_target,
        {
            "sample_count": 128,
            "trace_count": 36,
            "total_time_ns": 12.0,
        },
    )
    assert no_target_truth["targets"] == []
    assert "subsurface_target" not in no_target_truth["wavefield_rois"]
    assert {"direct_air_wave", "air_ground_reflection", "background", "late_noise"} <= set(
        no_target_truth["wavefield_rois"]
    )
    assert no_target_truth["metrics_hint"]["false_positive_penalty"] > 0.0


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


def test_build_gprmax_command_resolves_model_input_to_absolute_path(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = build_gprmax_command(
        tmp_path / "python.exe",
        Path("relative_model.in"),
        runs=1,
    )

    assert Path(command[3]).is_absolute()
    assert command[3].endswith("relative_model.in")


def test_find_windows_vcvars64_accepts_explicit_path(tmp_path: Path):
    vcvars = tmp_path / "vcvars64.bat"
    vcvars.write_text("@echo off\n", encoding="utf-8")

    assert report.find_windows_vcvars64(str(vcvars)) == vcvars


def test_which_accepts_windows_path_key(tmp_path: Path):
    tool = tmp_path / "fake_tool.exe"
    tool.write_text("", encoding="utf-8")

    assert report._which("fake_tool.exe", env={"Path": str(tmp_path)}) == str(tool)


def test_resolve_gprmax_runtime_env_loads_vcvars_for_windows_gpu(monkeypatch, tmp_path: Path):
    vcvars = tmp_path / "vcvars64.bat"
    vcvars.write_text("@echo off\n", encoding="utf-8")
    loaded_env = {"Path": "with-cl"}

    def fake_which(executable: str, env=None):
        if executable == "cl.exe" and env is loaded_env:
            return r"C:\VS\cl.exe"
        if executable == "cl.exe":
            return None
        return None

    monkeypatch.setattr(report.os, "name", "nt")
    monkeypatch.setattr(report, "_which", fake_which)
    monkeypatch.setattr(report, "find_windows_vcvars64", lambda explicit_path=None: vcvars)
    monkeypatch.setattr(report, "_load_vcvars_env", lambda path: loaded_env)

    env, info = report.resolve_gprmax_runtime_env(gpu=["0"])

    assert env is loaded_env
    assert info["cuda_vcvars_loaded"] is True
    assert info["cuda_vcvars64_path"] == str(vcvars)
    assert info["cl_available"] is True
    assert info["cl_path"] == r"C:\VS\cl.exe"


def test_convert_gprmax_run_resamples_to_real_field_ascan_length(tmp_path: Path, monkeypatch):
    raw = np.linspace(0.0, 1.0, 1000, dtype=np.float32)[:, None]
    raw = np.hstack([raw, raw * 2.0, raw * 3.0])

    def fake_read_gprmax_out(path: str):
        return {
            "data": raw,
            "time_step_s": 1.0e-11,
            "total_time_ns": 10.0,
        }

    monkeypatch.setattr(report, "read_gprmax_out", fake_read_gprmax_out)
    scenario = build_scenario_definitions()["airborne_no_target_background_v1"]
    out_file = tmp_path / f"{scenario.scenario_id}001.out"
    out_file.write_text("", encoding="utf-8")
    run_result = report.GprMaxRunResult(
        scenario_dir=tmp_path,
        model_input=tmp_path / f"{scenario.scenario_id}.in",
        model_inputs=[tmp_path / f"{scenario.scenario_id}.in"],
        command=["python", "-m", "gprMax"],
        commands=[["python", "-m", "gprMax"]],
        out_files=[out_file],
        stdout_path=tmp_path / "stdout.txt",
        stderr_path=tmp_path / "stderr.txt",
        returncode=0,
    )

    package = report.convert_gprmax_run(
        scenario,
        run_result,
        runs=3,
        ascan_samples=501,
    )

    assert package["bscan"].shape == (501, 3)
    assert package["scenario"]["simulation"]["sample_count"] == 501
    assert package["scenario"]["simulation"]["raw_sample_count"] == 1000
    assert package["scenario"]["simulation"]["target_ascan_sample_count"] == 501
    assert package["scenario"]["simulation"]["resampled_from_raw"] is True
    assert package["header_info"]["a_scan_length"] == 501
    assert package["header_info"]["raw_a_scan_length"] == 1000
    assert package["ground_truth"]["wavefield_rois"]["air_ground_reflection"]["roi"][
        "time_end_idx"
    ] <= 501


def test_ascan_sample_argument_defaults_to_real_field_length():
    args = report._parse_args([])

    assert args.ascan_samples == 501
    assert report._parse_args(["--ascan-samples", "701"]).ascan_samples == 701
    assert report._parse_args(["--ascan-samples", "0"]).ascan_samples == 0
    assert report._parse_args(["--cuda-vcvars", "C:/VS/vcvars64.bat"]).cuda_vcvars == (
        "C:/VS/vcvars64.bat"
    )


def test_find_out_files_orders_per_trace_height_variation_outputs(tmp_path: Path):
    for name in [
        "airborne_height_variation_cylinder_v1_trace010.out",
        "airborne_height_variation_cylinder_v1_trace002.out",
        "airborne_height_variation_cylinder_v1_trace001.out",
    ]:
        (tmp_path / name).write_text("", encoding="utf-8")

    ordered = find_out_files(tmp_path, "airborne_height_variation_cylinder_v1")

    assert [path.name for path in ordered] == [
        "airborne_height_variation_cylinder_v1_trace001.out",
        "airborne_height_variation_cylinder_v1_trace002.out",
        "airborne_height_variation_cylinder_v1_trace010.out",
    ]


def test_json_safe_removes_nonfinite_values_for_strict_report_json():
    payload = {
        "float_nan": float("nan"),
        "float_inf": float("inf"),
        "array": np.array([1.0, np.nan, np.inf], dtype=np.float64),
        "np_scalar": np.float32(np.nan),
        "flag": True,
    }

    safe = report._json_safe(payload)

    assert safe["float_nan"] is None
    assert safe["float_inf"] is None
    assert safe["array"] == [1.0, None, None]
    assert safe["np_scalar"] is None
    assert safe["flag"] is True
    json.dumps(safe, allow_nan=False)


def test_standard_report_pipeline_covers_non_motion_uavgpr_flow():
    pipeline = report._resolve_pipeline("uav_gpr_experience_baseline_v1")

    assert "sec_gain" in pipeline
    assert "agcGain" not in pipeline
    assert "motion_compensation_v2" not in pipeline
    assert pipeline == [
        "set_zero_time",
        "dewow",
        "frequency_filter_1d",
        "subtracting_average_2D",
        "sec_gain",
        "wavelet_svd",
    ]
    manual_params = report._resolve_manual_params(
        pipeline,
        "uav_gpr_experience_baseline_v1",
    )
    assert manual_params["frequency_filter_1d"]["high_freq_mhz"] == 3000.0
    assert manual_params["wavelet_svd"]["rank_end"] == 20


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
    assert "truth_score" in step["metrics"]["manual"]
    assert "truth_target_energy_preservation" in step["metrics"]["auto"]
    assert "truth_score" in comparison["metrics"]["delta_auto_minus_manual"]
    assert step["policy_notes"]
    assert "不判定人工/自动优劣" in step["analysis"]["visual"]
    assert "真值评分差值" in step["analysis"]["metrics"]
    assert comparison["zero_time_policy"] == "align_auto"
    assert comparison["backend"] == "core.auto_tune_pipeline"
    assert comparison["overall_recommendation"] in {"adopt_auto", "review", "keep_manual"}
    assert "recommendation" in step
    assert "risk_flags" in step
    assert "rolled_back_to_manual" in step


def test_stepwise_comparison_exposes_pipeline_backend_decisions():
    data = np.zeros((64, 20), dtype=np.float32)
    data[24:34, 8:13] = 1.0
    ground_truth = {
        "scenario_id": "backend_decision_demo",
        "analysis_roi": {
            "time_start_idx": 16,
            "time_end_idx": 46,
            "dist_start_idx": 4,
            "dist_end_idx": 16,
        },
        "targets": [
            {
                "target_id": "target_01",
                "type": "hyperbola",
                "must_preserve": True,
                "roi": {
                    "time_start_idx": 22,
                    "time_end_idx": 36,
                    "dist_start_idx": 7,
                    "dist_end_idx": 14,
                },
            }
        ],
    }

    comparison = report.run_stepwise_comparison(
        data,
        header_info={"total_time_ns": 12.0},
        trace_metadata={},
        ground_truth=ground_truth,
        baseline_profile_key="uav_gpr_experience_baseline_v1",
        search_mode="fast",
        zero_time_policy="skip",
    )

    assert comparison["backend"] == "core.auto_tune_pipeline"
    assert "set_zero_time" not in comparison["pipeline"]
    assert comparison["overall_recommendation"] in {"adopt_auto", "review", "keep_manual"}
    assert isinstance(comparison["risk_flags"], list)
    assert "pipeline_score" in comparison["metrics"]["delta_auto_minus_manual"]
    assert "comparison_score" in comparison["metrics"]["delta_auto_minus_manual"]

    first_step = comparison["steps"][0]
    assert "recommendation" in first_step
    assert "risk_flags" in first_step
    assert "rolled_back_to_manual" in first_step
    assert "pipeline_score" in first_step["metrics"]["delta_auto_minus_manual"]


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
                "geometry_model": "airborne_2d_tmz_v1",
                "is_uav_gpr_evidence": True,
                "ground_truth": {
                    "wavefield_rois": {
                        "direct_air_wave": {
                            "roi": {
                                "time_start_idx": 2,
                                "time_end_idx": 6,
                                "dist_start_idx": 0,
                                "dist_end_idx": 12,
                            },
                            "risk": "直达波风险",
                        },
                        "air_ground_reflection": {
                            "roi": {
                                "time_start_idx": 8,
                                "time_end_idx": 12,
                                "dist_start_idx": 0,
                                "dist_end_idx": 12,
                            },
                            "risk": "地表反射风险",
                        },
                        "subsurface_target": {
                            "roi": {
                                "time_start_idx": 16,
                                "time_end_idx": 28,
                                "dist_start_idx": 3,
                                "dist_end_idx": 8,
                            },
                            "risk": "目标风险",
                        },
                    }
                },
                "gprmax": {"command": ["python", "-m", "gprMax", "model.in"]},
                "comparison": {
                    "verdict": "auto_better",
                    "overall_recommendation": "adopt_auto",
                    "risk_flags": [],
                    "metrics": {
                        "manual": {"comparison_score": 1.0},
                        "auto": {"comparison_score": 1.2},
                        "delta_auto_minus_manual": {
                            "comparison_score": 0.2,
                            "low_freq_energy_reduction": 0.1,
                            "target_band_energy_ratio": 0.05,
                            "edge_preservation": 0.03,
                            "truth_score": 0.25,
                            "truth_target_energy_preservation": 0.12,
                            "truth_background_energy_reduction": 0.18,
                            "truth_false_positive_ratio": -0.04,
                        },
                    },
                },
                "images": [
                    {
                        "method_key": "dewow",
                        "method_name": "低频漂移抑制",
                        "manual_params": {"window": 61},
                        "auto_params": {"window": 31},
                        "auto_tune_summary": {
                            "best_reason": "评分最高",
                            "risk_flags": ["constraint_adjusted"],
                            "selection_recommendation": "review",
                            "parameter_domain": {
                                "notes": [
                                    "部分候选参数被按数据尺度收缩，实际搜索域小于原始候选列表。",
                                ]
                            },
                        },
                        "analysis": {
                            "visual": "自动选参在视觉上保留目标。",
                            "metrics": "score 差值为 0.2。",
                        },
                        "manual_warnings": [],
                        "auto_warnings": [],
                        "recommendation": "review",
                        "risk_flags": ["near_tie"],
                        "rolled_back_to_manual": False,
                        "reason": "risk=near_tie; auto pipeline score delta=0.0100",
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
    assert "后端建议" in html_text
    assert "风险标记" in html_text
    assert "建议动作" in html_text
    assert "参数域提示" in html_text
    assert "是否回退人工结果" in html_text
    assert "真值评分差值" in html_text
    assert "目标能量保留差值" in html_text
    assert "目标外背景抑制差值" in html_text
    assert "假异常比例差值" in html_text
    assert "波场特征检查" in html_text
    assert "直达波" in html_text
    assert "地表反射" in html_text
    assert "天线离地高度" in html_text
    assert "高度变化" in html_text
