#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run multi-scenario gprMax validation and build an HTML auto-tune report."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.auto_tune import auto_tune_method  # noqa: E402
from core.gprmax_truth_metrics import compute_ground_truth_metrics  # noqa: E402
from core.gpr_io import read_gprmax_out  # noqa: E402
from core.methods_registry import PROCESSING_METHODS  # noqa: E402
from core.preset_profiles import GUI_PRESETS_V1, RECOMMENDED_RUN_PROFILES  # noqa: E402
from core.processing_engine import (  # noqa: E402
    clone_header_info,
    clone_trace_metadata,
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.quality_metrics import (  # noqa: E402
    compute_benchmark_metrics,
    ratio_fidelity,
)


DEFAULT_GPRMAX_ROOT = Path(r"E:\gprMax\gprMax-v.3.1.7")
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "gprmax_multi_scenario_reports"
DEFAULT_PROFILE_KEY = "uav_gpr_experience_baseline_v1"
DEFAULT_RUNS = 36
REPORT_PIPELINE_ORDER = [
    "set_zero_time",
    "dewow",
    "subtracting_average_2D",
    "agcGain",
    "svd_subspace",
]
REPORT_MANUAL_PARAM_OVERRIDES = {
    "agcGain": {"window": 31},
}
ZERO_TIME_ALIGN_NOTE = (
    "本报告将人工分支零时参数对齐自动结果，避免经验 5.0ns "
    "在小域正演数据中切掉有效结构。"
)

DOMAIN_M = (0.240, 0.210, 0.002)
DX_M = 0.002
TOTAL_TIME_NS = 12.0
TRACE_STEP_M = 0.002
GROUND_TOP_Y_M = 0.165
SOURCE_START_M = (0.035, GROUND_TOP_Y_M, 0.0)
RECEIVER_START_M = (0.075, GROUND_TOP_Y_M, 0.0)
SOURCE_RECEIVER_OFFSET_M = RECEIVER_START_M[0] - SOURCE_START_M[0]
LIGHT_SPEED_M_PER_NS = 0.299792458


@dataclass(frozen=True)
class ScenarioDefinition:
    """Static definition for one gprMax validation scene."""

    scenario_id: str
    label: str
    description: str
    structure_notes: list[str]
    model_in_text: str
    materials: list[dict[str, Any]]
    targets: list[dict[str, Any]]
    layers: list[dict[str, Any]]
    domain_m: tuple[float, float, float] = DOMAIN_M
    dx_m: float = DX_M
    total_time_ns: float = TOTAL_TIME_NS
    trace_step_m: float = TRACE_STEP_M
    source_start_m: tuple[float, float, float] = SOURCE_START_M
    receiver_start_m: tuple[float, float, float] = RECEIVER_START_M
    ground_top_y_m: float = GROUND_TOP_Y_M
    default_runs: int = DEFAULT_RUNS


@dataclass(frozen=True)
class GprMaxRunResult:
    """Paths and metadata from one gprMax run."""

    scenario_dir: Path
    model_input: Path
    command: list[str]
    out_files: list[Path]
    stdout_path: Path
    stderr_path: Path
    returncode: int


def build_scenario_definitions() -> dict[str, ScenarioDefinition]:
    """Return the simple scenes used by the research report."""
    return {
        "cylinder_single_v1": _single_cylinder_scenario(),
        "cylinder_double_v1": _double_cylinder_scenario(),
        "layered_interface_v1": _layered_interface_scenario(),
        "crack_air_filled_v1": _crack_air_filled_scenario(),
        "no_target_background_v1": _no_target_background_scenario(),
    }


def build_gprmax_command(
    python_exe: Path,
    model_input: Path,
    *,
    runs: int,
    geometry_fixed: bool = False,
    mpi: int | None = None,
    gpu: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build a gprMax command line with optional parallel flags."""
    command = [
        str(python_exe),
        "-m",
        "gprMax",
        str(model_input),
        "-n",
        str(int(runs)),
    ]
    if geometry_fixed:
        command.append("--geometry-fixed")
    if mpi is not None and int(mpi) > 0:
        command.extend(["-mpi", str(int(mpi))])
    if gpu:
        command.append("-gpu")
        command.extend(str(item) for item in gpu)
    if extra_args:
        command.extend(str(item) for item in extra_args)
    return command


def resolve_gprmax_python(
    gprmax_root: Path,
    python_override: str | None = None,
) -> Path:
    """Resolve the Python interpreter used to run gprMax."""
    if python_override:
        candidate = Path(python_override)
        if not candidate.exists():
            raise FileNotFoundError(f"python override not found: {candidate}")
        return candidate
    venv_python = Path(gprmax_root) / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def probe_acceleration_support(python_exe: Path) -> dict[str, Any]:
    """Check whether the gprMax Python environment has MPI/GPU packages."""
    probe = (
        "import importlib.util,json;"
        "print(json.dumps({"
        "'mpi4py': importlib.util.find_spec('mpi4py') is not None,"
        "'cupy': importlib.util.find_spec('cupy') is not None"
        "}))"
    )
    try:
        completed = subprocess.run(
            [str(python_exe), "-c", probe],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - defensive environment probe
        return {"mpi4py": False, "cupy": False, "probe_error": str(exc)}
    if completed.returncode != 0:
        return {
            "mpi4py": False,
            "cupy": False,
            "probe_error": completed.stderr[-1000:],
        }
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {
            "mpi4py": False,
            "cupy": False,
            "probe_error": completed.stdout[-1000:],
        }


def find_out_files(run_dir: Path, scenario_id: str) -> list[Path]:
    """Find gprMax output files using numeric suffix sorting."""
    found: list[tuple[int, Path]] = []
    for path in Path(run_dir).glob(f"{scenario_id}*.out"):
        suffix = path.stem[len(scenario_id) :]
        number = int(suffix) if suffix.isdigit() else 0
        found.append((number, path))
    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def run_multi_scenario_report(
    *,
    gprmax_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    scenario_ids: list[str] | None = None,
    runs: int = DEFAULT_RUNS,
    geometry_fixed: bool = True,
    mpi: int | None = None,
    gpu: list[str] | None = None,
    python_override: str | None = None,
    search_mode: str = "fast",
    baseline_profile_key: str = DEFAULT_PROFILE_KEY,
    zero_time_policy: str = "align_auto",
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run all requested scenarios and write the HTML report."""
    scenarios = build_scenario_definitions()
    selected_ids = list(scenario_ids or scenarios.keys())
    unknown = [scenario_id for scenario_id in selected_ids if scenario_id not in scenarios]
    if unknown:
        raise ValueError(f"Unknown scenario id(s): {', '.join(unknown)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_root) / timestamp
    assets_dir = report_dir / "assets"
    scenario_root = report_dir / "scenarios"
    assets_dir.mkdir(parents=True, exist_ok=True)
    scenario_root.mkdir(parents=True, exist_ok=True)

    gprmax_root = Path(gprmax_root)
    if not gprmax_root.exists():
        raise FileNotFoundError(f"gprMax root not found: {gprmax_root}")
    python_exe = resolve_gprmax_python(gprmax_root, python_override=python_override)
    acceleration = probe_acceleration_support(python_exe)

    scenario_records: list[dict[str, Any]] = []
    for scenario_id in selected_ids:
        definition = scenarios[scenario_id]
        scenario_dir = scenario_root / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        gprmax_run = run_gprmax_scenario(
            definition,
            scenario_dir=scenario_dir,
            gprmax_root=gprmax_root,
            python_exe=python_exe,
            runs=runs,
            geometry_fixed=geometry_fixed,
            mpi=mpi,
            gpu=gpu,
            extra_args=extra_args,
        )
        package = convert_gprmax_run(
            definition,
            gprmax_run,
            runs=runs,
        )
        comparison = run_stepwise_comparison(
            package["bscan"],
            header_info=package["header_info"],
            trace_metadata=package["trace_metadata"],
            ground_truth=package["ground_truth"],
            baseline_profile_key=baseline_profile_key,
            search_mode=search_mode,
            zero_time_policy=zero_time_policy,
        )
        image_records = save_step_images(
            scenario_id=scenario_id,
            steps=comparison["steps"],
            assets_dir=assets_dir,
            report_dir=report_dir,
        )
        structure_rel = _relpath(package["structure_preview_path"], report_dir)
        scenario_records.append(
            {
                "scenario_id": scenario_id,
                "label": definition.label,
                "description": definition.description,
                "structure_notes": list(definition.structure_notes),
                "scenario_json": _relpath(package["scenario_path"], report_dir),
                "ground_truth_json": _relpath(package["ground_truth_path"], report_dir),
                "bscan_csv": _relpath(package["bscan_csv_path"], report_dir),
                "structure_preview": structure_rel,
                "shape": list(package["bscan"].shape),
                "simulation": package["scenario"]["simulation"],
                "ground_truth": package["ground_truth"],
                "gprmax": {
                    "command": gprmax_run.command,
                    "stdout": _relpath(gprmax_run.stdout_path, report_dir),
                    "stderr": _relpath(gprmax_run.stderr_path, report_dir),
                    "returncode": int(gprmax_run.returncode),
                    "out_files": [_relpath(path, report_dir) for path in gprmax_run.out_files],
                },
                "comparison": _comparison_without_arrays(comparison),
                "images": image_records,
            }
        )

    payload = {
        "schema": "mygpr_gprmax_multi_scenario_report_v1",
        "generated_at": timestamp,
        "baseline_profile_key": baseline_profile_key,
        "search_mode": search_mode,
        "zero_time_policy": zero_time_policy,
        "gprmax_root": str(gprmax_root),
        "python_executable": str(python_exe),
        "run_settings": {
            "runs": int(runs),
            "geometry_fixed": bool(geometry_fixed),
            "mpi": int(mpi) if mpi is not None else None,
            "gpu": list(gpu or []),
            "extra_args": list(extra_args or []),
        },
        "acceleration_support": acceleration,
        "scenarios": scenario_records,
    }
    summary_json = report_dir / "summary.json"
    summary_json.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path = render_html_report(report_dir, payload)
    payload["summary_json"] = str(summary_json)
    payload["html_report"] = str(html_path)
    summary_json.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def run_gprmax_scenario(
    definition: ScenarioDefinition,
    *,
    scenario_dir: Path,
    gprmax_root: Path,
    python_exe: Path,
    runs: int,
    geometry_fixed: bool,
    mpi: int | None,
    gpu: list[str] | None,
    extra_args: list[str] | None,
) -> GprMaxRunResult:
    """Write the model file and run gprMax for one scene."""
    model_input = scenario_dir / f"{definition.scenario_id}.in"
    model_input.write_text(definition.model_in_text, encoding="utf-8")
    command = build_gprmax_command(
        python_exe,
        model_input,
        runs=runs,
        geometry_fixed=geometry_fixed,
        mpi=mpi,
        gpu=gpu,
        extra_args=extra_args,
    )
    completed = subprocess.run(
        command,
        cwd=str(gprmax_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_path = scenario_dir / "gprmax_stdout.txt"
    stderr_path = scenario_dir / "gprmax_stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    out_files = find_out_files(scenario_dir, definition.scenario_id)
    if completed.returncode != 0:
        raise RuntimeError(
            f"gprMax failed for {definition.scenario_id}; see {stderr_path}"
        )
    if not out_files:
        raise RuntimeError(f"gprMax produced no .out files for {definition.scenario_id}")
    return GprMaxRunResult(
        scenario_dir=scenario_dir,
        model_input=model_input,
        command=command,
        out_files=out_files,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=int(completed.returncode),
    )


def convert_gprmax_run(
    definition: ScenarioDefinition,
    run_result: GprMaxRunResult,
    *,
    runs: int,
) -> dict[str, Any]:
    """Convert gprMax `.out` files into MyGPR package artifacts."""
    load_result = read_gprmax_out(str(run_result.out_files[0]))
    bscan = np.asarray(load_result["data"], dtype=np.float32)
    time_step_s = load_result.get("time_step_s")
    total_time_ns = load_result.get("total_time_ns")
    if total_time_ns is None and time_step_s is not None:
        total_time_ns = float(time_step_s) * int(bscan.shape[0]) * 1e9
    if total_time_ns is None:
        total_time_ns = float(definition.total_time_ns)

    simulation = {
        "sample_count": int(bscan.shape[0]),
        "trace_count": int(bscan.shape[1]),
        "requested_runs": int(runs),
        "time_step_s": float(time_step_s) if time_step_s is not None else None,
        "total_time_ns": float(total_time_ns),
        "trace_step_m": float(definition.trace_step_m),
        "source_receiver_offset_m": float(SOURCE_RECEIVER_OFFSET_M),
    }
    ground_truth = build_ground_truth(definition, simulation)
    scenario = {
        "schema": "mygpr_gprmax_multiscenario_scenario_v1",
        "scenario_id": definition.scenario_id,
        "label": definition.label,
        "description": definition.description,
        "source": {
            "kind": "gprmax_out",
            "model_input": str(run_result.model_input),
            "first_out": str(run_result.out_files[0]),
        },
        "simulation": simulation,
        "domain_m": list(definition.domain_m),
        "dx_dy_dz_m": [definition.dx_m, definition.dx_m, definition.dx_m],
        "materials": definition.materials,
        "layers": definition.layers,
        "targets": definition.targets,
        "antenna": {
            "waveform": "ricker",
            "center_frequency_hz": 1.5e9,
            "source": "hertzian_dipole_z",
            "source_position_m": list(definition.source_start_m),
            "receiver_position_m": list(definition.receiver_start_m),
            "source_step_m": [definition.trace_step_m, 0.0, 0.0],
            "receiver_step_m": [definition.trace_step_m, 0.0, 0.0],
        },
    }
    bscan_csv_path = run_result.scenario_dir / "mygpr_bscan.csv"
    scenario_path = run_result.scenario_dir / "scenario.json"
    ground_truth_path = run_result.scenario_dir / "ground_truth.json"
    structure_preview_path = run_result.scenario_dir / "structure.png"
    np.savetxt(bscan_csv_path, bscan, delimiter=",", fmt="%.8e")
    scenario_path.write_text(
        json.dumps(_json_safe(scenario), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ground_truth_path.write_text(
        json.dumps(_json_safe(ground_truth), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_structure_preview(definition, structure_preview_path)
    return {
        "bscan": bscan,
        "scenario": scenario,
        "ground_truth": ground_truth,
        "bscan_csv_path": bscan_csv_path,
        "scenario_path": scenario_path,
        "ground_truth_path": ground_truth_path,
        "structure_preview_path": structure_preview_path,
        "header_info": {
            "a_scan_length": int(bscan.shape[0]),
            "num_traces": int(bscan.shape[1]),
            "total_time_ns": float(total_time_ns),
            "trace_interval_m": float(definition.trace_step_m),
            "track_length_m": float(definition.trace_step_m) * max(int(bscan.shape[1]) - 1, 1),
        },
        "trace_metadata": {},
    }


def build_ground_truth(
    definition: ScenarioDefinition,
    simulation: dict[str, Any],
) -> dict[str, Any]:
    """Build simulation-derived ground truth and analysis ROI."""
    samples = int(simulation["sample_count"])
    traces = int(simulation["trace_count"])
    total_time_ns = float(simulation["total_time_ns"])
    targets: list[dict[str, Any]] = []
    rois: list[dict[str, int]] = []

    for target in definition.targets:
        target_type = str(target.get("type"))
        if target_type == "metal_cylinder":
            roi_target = _target_ground_truth(definition, target, samples, traces, total_time_ns)
        elif target_type == "layer_interface":
            roi_target = _layer_ground_truth(definition, target, samples, traces, total_time_ns)
        elif target_type == "air_crack":
            roi_target = _crack_ground_truth(definition, target, samples, traces, total_time_ns)
        else:
            continue
        targets.append(roi_target)
        rois.append(dict(roi_target["roi"]))

    analysis_roi = _union_rois(rois, samples, traces) if rois else _full_roi(samples, traces)
    return {
        "schema": "mygpr_gprmax_multiscenario_ground_truth_v1",
        "scenario_id": definition.scenario_id,
        "structure_notes": list(definition.structure_notes),
        "targets": targets,
        "analysis_roi": analysis_roi,
        "known_background": {
            "air_ground_interface_y_m": float(definition.ground_top_y_m),
            "layers": definition.layers,
        },
        "metrics_hint": {
            "target_roi_weight": 1.0,
            "background_roi_weight": 0.5,
            "false_positive_penalty": 0.7,
        },
    }


def run_stepwise_comparison(
    data: np.ndarray,
    *,
    header_info: dict[str, Any] | None,
    trace_metadata: dict[str, np.ndarray] | None,
    ground_truth: dict[str, Any],
    baseline_profile_key: str,
    search_mode: str,
    zero_time_policy: str = "align_auto",
) -> dict[str, Any]:
    """Run manual baseline and auto-tuned branches while preserving step arrays."""
    if zero_time_policy not in {"align_auto", "manual", "skip"}:
        raise ValueError(f"Unsupported zero_time_policy: {zero_time_policy}")
    arr = np.asarray(data, dtype=np.float32)
    pipeline = _resolve_pipeline(baseline_profile_key)
    manual_params_by_method = _resolve_manual_params(pipeline, baseline_profile_key)
    roi_spec = {
        "mode": "manual",
        "bounds": ground_truth.get("analysis_roi") or _full_roi(*arr.shape),
        "label": f"{ground_truth.get('scenario_id', 'scenario')} ground-truth ROI",
    }

    manual_current = np.array(arr, copy=True)
    auto_current = np.array(arr, copy=True)
    manual_header = clone_header_info(header_info)
    auto_header = clone_header_info(header_info)
    manual_trace_metadata = clone_trace_metadata(trace_metadata)
    auto_trace_metadata = clone_trace_metadata(trace_metadata)

    auto_params_by_method: dict[str, dict[str, Any]] = {}
    auto_tune_results: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    manual_roi = dict(roi_spec["bounds"])
    auto_roi = dict(roi_spec["bounds"])

    for method_key in pipeline:
        method_info = PROCESSING_METHODS[method_key]
        method_name = str(method_info.get("name") or method_key)
        manual_input = np.array(manual_current, copy=True)
        auto_input = np.array(auto_current, copy=True)
        manual_input_roi = dict(manual_roi)
        auto_input_roi = dict(auto_roi)

        manual_params = dict(manual_params_by_method.get(method_key, {}))
        manual_original_params = dict(manual_params)
        auto_params = dict(manual_params_by_method.get(method_key, {}))
        tune_summary: dict[str, Any] = {}
        auto_warnings: list[str] = []
        policy_notes: list[str] = []
        skip_step = method_key == "set_zero_time" and zero_time_policy == "skip"

        if skip_step:
            manual_meta = {
                "method_id": method_key,
                "skipped": True,
                "reason": "zero_time_policy=skip",
            }
            auto_meta = dict(manual_meta)
            policy_notes.append("本报告按 zero_time_policy=skip 跳过零时矫正。")
        else:
            if method_info.get("auto_tune_enabled"):
                try:
                    tune_result = auto_tune_method(
                        auto_current,
                        method_key,
                        header_info=auto_header,
                        trace_metadata=auto_trace_metadata,
                        base_params=auto_params,
                        roi_spec=roi_spec,
                        search_mode=search_mode,
                    )
                    recommended = dict(
                        tune_result.get("recommended_params")
                        or tune_result.get("best_params")
                        or {}
                    )
                    auto_params.update(recommended)
                    tune_summary = _compact_auto_tune_result(tune_result)
                except Exception as exc:
                    auto_warnings.append(f"auto_tune_failed: {exc}")
                    tune_summary = {"error": str(exc)}

            if method_key == "set_zero_time" and zero_time_policy == "align_auto":
                if "new_zero_time" in auto_params:
                    manual_params = dict(auto_params)
                    policy_notes.append(ZERO_TIME_ALIGN_NOTE)
                else:
                    manual_params = {"new_zero_time": 0.0}
                    auto_params = {"new_zero_time": 0.0}
                    policy_notes.append(
                        "自动零时选参失败，本报告将两分支零时参数设为 0.0ns，"
                        "避免经验 5.0ns 切掉有效结构。"
                    )
            manual_params_by_method[method_key] = dict(manual_params)

            manual_runtime_params = prepare_runtime_params(
                method_key,
                manual_params,
                manual_header,
                manual_trace_metadata,
                manual_current.shape,
            )
            manual_current, manual_meta = run_processing_method(
                manual_current,
                method_key,
                manual_runtime_params,
            )
            manual_header = merge_result_header_info(
                manual_header, manual_meta, manual_current.shape
            )
            manual_trace_metadata = merge_result_trace_metadata(
                manual_trace_metadata, manual_meta
            )

            auto_runtime_params = prepare_runtime_params(
                method_key,
                auto_params,
                auto_header,
                auto_trace_metadata,
                auto_current.shape,
            )
            auto_current, auto_meta = run_processing_method(
                auto_current,
                method_key,
                auto_runtime_params,
            )
            auto_header = merge_result_header_info(
                auto_header, auto_meta, auto_current.shape
            )
            auto_trace_metadata = merge_result_trace_metadata(
                auto_trace_metadata, auto_meta
            )

        manual_output_roi = _roi_after_method(
            manual_input_roi,
            method_key,
            manual_meta,
            manual_current.shape,
        )
        auto_output_roi = _roi_after_method(
            auto_input_roi,
            method_key,
            auto_meta,
            auto_current.shape,
        )
        step_metrics = _step_metric_summary(
            manual_input,
            manual_current,
            auto_input,
            auto_current,
            manual_input_roi,
            manual_output_roi,
            auto_input_roi,
            auto_output_roi,
            ground_truth,
        )
        analysis = _build_step_analysis(
            method_key=method_key,
            method_name=method_name,
            metrics=step_metrics,
            ground_truth=ground_truth,
            policy_notes=policy_notes,
        )

        auto_params_by_method[method_key] = dict(auto_params)
        auto_tune_results[method_key] = tune_summary
        auto_warnings.extend(_extract_warning_messages(auto_meta))

        steps.append(
            {
                "method_key": method_key,
                "method_name": method_name,
                "manual_input": manual_input,
                "auto_input": auto_input,
                "manual_output": np.array(manual_current, copy=True),
                "auto_output": np.array(auto_current, copy=True),
                "manual_params": dict(manual_params),
                "manual_original_params": manual_original_params,
                "auto_params": dict(auto_params),
                "auto_tune_summary": tune_summary,
                "manual_warnings": _extract_warning_messages(manual_meta),
                "auto_warnings": auto_warnings,
                "policy_notes": policy_notes,
                "manual_input_roi": manual_input_roi,
                "auto_input_roi": auto_input_roi,
                "manual_output_roi": manual_output_roi,
                "auto_output_roi": auto_output_roi,
                "metrics": step_metrics,
                "analysis": analysis,
            }
        )
        manual_roi = dict(manual_output_roi)
        auto_roi = dict(auto_output_roi)

    metric_summary = _final_metric_summary(
        arr,
        manual_current,
        auto_current,
        roi_spec["bounds"],
        manual_roi,
        auto_roi,
        ground_truth,
    )
    return {
        "pipeline": pipeline,
        "manual_params_by_method": manual_params_by_method,
        "auto_params_by_method": auto_params_by_method,
        "auto_tune_results": auto_tune_results,
        "manual_final": manual_current,
        "auto_final": auto_current,
        "steps": steps,
        "roi_spec": roi_spec,
        "final_manual_roi": manual_roi,
        "final_auto_roi": auto_roi,
        "zero_time_policy": zero_time_policy,
        "metrics": metric_summary,
        "verdict": _comparison_verdict(metric_summary),
    }


def save_step_images(
    *,
    scenario_id: str,
    steps: list[dict[str, Any]],
    assets_dir: Path,
    report_dir: Path,
) -> list[dict[str, Any]]:
    """Render BScan images for every processing step."""
    records: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        arrays = [
            step["manual_input"],
            step["auto_input"],
            step["manual_output"],
            step["auto_output"],
        ]
        vlim = _locked_vlim(arrays)
        prefix = f"{scenario_id}_step{idx:02d}_{step['method_key']}"
        image_paths = {
            "manual_input": assets_dir / f"{prefix}_manual_before.png",
            "auto_input": assets_dir / f"{prefix}_auto_before.png",
            "manual_output": assets_dir / f"{prefix}_manual_after.png",
            "auto_output": assets_dir / f"{prefix}_auto_after.png",
        }
        save_bscan_image(
            step["manual_input"],
            image_paths["manual_input"],
            "manual before",
            vlim,
            step.get("manual_input_roi"),
        )
        save_bscan_image(
            step["auto_input"],
            image_paths["auto_input"],
            "auto before",
            vlim,
            step.get("auto_input_roi"),
        )
        save_bscan_image(
            step["manual_output"],
            image_paths["manual_output"],
            "manual after",
            vlim,
            step.get("manual_output_roi"),
        )
        save_bscan_image(
            step["auto_output"],
            image_paths["auto_output"],
            "auto after",
            vlim,
            step.get("auto_output_roi"),
        )
        records.append(
            {
                "method_key": step["method_key"],
                "method_name": step["method_name"],
                "manual_params": _json_safe(step["manual_params"]),
                "manual_original_params": _json_safe(step.get("manual_original_params", {})),
                "auto_params": _json_safe(step["auto_params"]),
                "auto_tune_summary": _json_safe(step["auto_tune_summary"]),
                "manual_warnings": list(step["manual_warnings"]),
                "auto_warnings": list(step["auto_warnings"]),
                "policy_notes": list(step.get("policy_notes", [])),
                "metrics": _json_safe(step.get("metrics", {})),
                "analysis": _json_safe(step.get("analysis", {})),
                "images": {
                    key: _relpath(path, report_dir) for key, path in image_paths.items()
                },
            }
        )
    return records


def render_html_report(report_dir: Path, payload: dict[str, Any]) -> Path:
    """Write a static HTML report and return its path."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "index.html"
    scenario_sections = "\n".join(_render_scenario_section(item) for item in payload["scenarios"])
    run_settings = payload.get("run_settings", {})
    support = payload.get("acceleration_support", {})
    css = _html_css()
    overall_summary = _render_overall_summary(payload)
    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MyGPR gprMax 自动选参验证报告</title>
  <style>{css}</style>
</head>
<body>
  <main>
    <section class="hero">
      <p class="eyebrow">MyGPR / gprMax validation</p>
      <h1>自动选参与人工经验参数对比报告</h1>
      <p>本报告使用 gprMax 正演数据构造已知真实结构，以逐步骤图像、参数和指标对比验证 MyGPR 自动选参相对经验 baseline 的表现。</p>
    </section>

    <section>
      <h2>gprMax 运行设置</h2>
      <div class="kv-grid">
        <div><span>gprMax 根目录</span><strong>{_esc(payload.get("gprmax_root"))}</strong></div>
        <div><span>Python</span><strong>{_esc(payload.get("python_executable"))}</strong></div>
        <div><span>每场景 A-scan 道数</span><strong>{_esc(run_settings.get("runs"))}</strong></div>
        <div><span>geometry-fixed</span><strong>{_esc(run_settings.get("geometry_fixed"))}</strong></div>
        <div><span>MPI 参数</span><strong>{_esc(run_settings.get("mpi") or "未启用")}</strong></div>
        <div><span>GPU 参数</span><strong>{_esc(", ".join(run_settings.get("gpu") or []) or "未启用")}</strong></div>
        <div><span>mpi4py 可用</span><strong>{_esc(support.get("mpi4py"))}</strong></div>
        <div><span>CuPy 可用</span><strong>{_esc(support.get("cupy"))}</strong></div>
      </div>
      <p class="note">本机环境探测只用于决定是否建议启用 MPI/GPU。当前脚本会暴露 <code>-mpi</code> 和 <code>-gpu</code>，但只有在相应依赖可用且用户显式传参时才加入命令。</p>
    </section>

    <section>
      <h2>参数策略</h2>
      <p>人工选参使用 <code>{_esc(payload.get("baseline_profile_key"))}</code> 的经验 baseline，并按本报告策略将增益步骤替换为 AGC。自动选参以同一组参数为 base，在每个支持 auto-tune 的算法步骤前根据当前 BScan 和 ground-truth ROI 重新评分并选择参数。搜索模式为 <code>{_esc(payload.get("search_mode"))}</code>，零时策略为 <code>{_esc(payload.get("zero_time_policy"))}</code>。</p>
    </section>

    {overall_summary}

    {scenario_sections}
  </main>
</body>
</html>
"""
    html_path.write_text(body, encoding="utf-8")
    return html_path


def save_bscan_image(
    data: np.ndarray,
    out_path: Path,
    title: str,
    vlim: float,
    roi: dict[str, int] | None = None,
) -> None:
    """Save a BScan image with optional ROI overlay."""
    arr = np.asarray(data, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    ax.imshow(arr, cmap="gray", aspect="auto", vmin=-vlim, vmax=vlim)
    if roi:
        rect = plt.Rectangle(
            (int(roi["dist_start_idx"]), int(roi["time_start_idx"])),
            int(roi["dist_end_idx"]) - int(roi["dist_start_idx"]),
            int(roi["time_end_idx"]) - int(roi["time_start_idx"]),
            fill=False,
            edgecolor="#d9480f",
            linewidth=1.2,
        )
        ax.add_patch(rect)
    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_structure_preview(definition: ScenarioDefinition, out_path: Path) -> None:
    """Render a simple true-structure preview from scenario metadata."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    domain_x, domain_y, _ = definition.domain_m
    ax.add_patch(
        plt.Rectangle((0, 0), domain_x, domain_y, facecolor="#f8f9fa", edgecolor="#333333")
    )
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            domain_x,
            definition.ground_top_y_m,
            facecolor="#d8c8a8",
            edgecolor="#9a7b4f",
            alpha=0.9,
        )
    )
    for layer in definition.layers:
        if layer.get("kind") == "box":
            y0 = float(layer.get("y0_m", 0.0))
            y1 = float(layer.get("y1_m", y0))
            color = str(layer.get("color", "#b7d8a8"))
            ax.add_patch(
                plt.Rectangle(
                    (0, y0),
                    domain_x,
                    max(0.0, y1 - y0),
                    facecolor=color,
                    edgecolor="#66885f",
                    alpha=0.78,
                )
            )
    for target in definition.targets:
        if target.get("type") == "metal_cylinder":
            center = target["center_m"]
            radius = float(target["radius_m"])
            ax.add_patch(
                plt.Circle(
                    (float(center[0]), float(center[1])),
                    radius,
                    facecolor="#495057",
                    edgecolor="#111111",
                    linewidth=1.0,
                )
            )
            ax.text(float(center[0]), float(center[1]) - radius * 1.7, target["target_id"], ha="center", fontsize=7)
        elif target.get("type") == "layer_interface":
            y = float(target["interface_y_m"])
            ax.axhline(y, color="#1864ab", linewidth=1.4)
            ax.text(domain_x * 0.02, y + 0.006, target["target_id"], fontsize=8, color="#1864ab")
        elif target.get("type") == "air_crack":
            x0 = float(target["x0_m"])
            x1 = float(target["x1_m"])
            y0 = float(target["y0_m"])
            y1 = float(target["y1_m"])
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    max(0.001, x1 - x0),
                    max(0.001, y1 - y0),
                    facecolor="#ffffff",
                    edgecolor="#7c2d12",
                    linewidth=1.2,
                    hatch="//",
                )
            )
            ax.text((x0 + x1) / 2.0, y1 + 0.006, target["target_id"], ha="center", fontsize=7)
    ax.scatter([definition.source_start_m[0]], [definition.source_start_m[1]], marker="^", color="#c92a2a", label="Tx start")
    ax.scatter([definition.receiver_start_m[0]], [definition.receiver_start_m[1]], marker="v", color="#1864ab", label="Rx start")
    ax.set_xlim(0, domain_x)
    ax.set_ylim(0, domain_y)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(definition.label)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _single_cylinder_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "silty_sand", "relative_permittivity": 6.0, "conductivity_s_per_m": 0.002},
        {"name": "metal_cylinder", "material": "pec"},
    ]
    targets = [
        {
            "target_id": "metal_cylinder_01",
            "type": "metal_cylinder",
            "center_m": [0.095, 0.080, 0.0],
            "radius_m": 0.010,
            "relative_permittivity": 6.0,
        }
    ]
    return ScenarioDefinition(
        scenario_id="cylinder_single_v1",
        label="单金属圆柱",
        description="均匀粉砂土中的单个 PEC 圆柱，主要验证双曲线目标能否保留且背景低频/水平杂波能否降低。",
        structure_notes=[
            "背景介质为相对介电常数 6、弱电导粉砂土。",
            "目标为 x=0.095 m、y=0.080 m、半径 0.010 m 的金属圆柱。",
        ],
        model_in_text=_model_text(
            "cylinder_single_v1",
            materials=["#material: 6 0.002 1 0 silty_sand"],
            bodies=[
                "#box: 0 0 0 0.240 0.165 0.002 silty_sand",
                "#cylinder: 0.095 0.080 0 0.095 0.080 0.002 0.010 pec",
            ],
        ),
        materials=materials,
        targets=targets,
        layers=[],
    )


def _double_cylinder_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "silty_sand", "relative_permittivity": 6.0, "conductivity_s_per_m": 0.002},
        {"name": "metal_cylinder", "material": "pec"},
    ]
    targets = [
        {
            "target_id": "shallow_metal_cylinder",
            "type": "metal_cylinder",
            "center_m": [0.085, 0.098, 0.0],
            "radius_m": 0.008,
            "relative_permittivity": 6.0,
        },
        {
            "target_id": "deep_metal_cylinder",
            "type": "metal_cylinder",
            "center_m": [0.118, 0.066, 0.0],
            "radius_m": 0.007,
            "relative_permittivity": 6.0,
        },
    ]
    return ScenarioDefinition(
        scenario_id="cylinder_double_v1",
        label="双金属圆柱",
        description="同一测线下两个不同深度的 PEC 圆柱，主要验证自动选参是否能同时保护浅层强目标和深层弱目标。",
        structure_notes=[
            "背景介质与单圆柱场景一致，为相对介电常数 6 的粉砂土。",
            "浅层目标位于 x=0.085 m、y=0.098 m；深层目标位于 x=0.118 m、y=0.066 m。",
        ],
        model_in_text=_model_text(
            "cylinder_double_v1",
            materials=["#material: 6 0.002 1 0 silty_sand"],
            bodies=[
                "#box: 0 0 0 0.240 0.165 0.002 silty_sand",
                "#cylinder: 0.085 0.098 0 0.085 0.098 0.002 0.008 pec",
                "#cylinder: 0.118 0.066 0 0.118 0.066 0.002 0.007 pec",
            ],
        ),
        materials=materials,
        targets=targets,
        layers=[],
    )


def _layered_interface_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "dry_sand", "relative_permittivity": 4.5, "conductivity_s_per_m": 0.001},
        {"name": "wet_sand", "relative_permittivity": 10.0, "conductivity_s_per_m": 0.012},
    ]
    layers = [
        {
            "kind": "box",
            "name": "wet_sand_lower_layer",
            "material": "wet_sand",
            "y0_m": 0.000,
            "y1_m": 0.082,
            "color": "#9bbf8f",
        }
    ]
    targets = [
        {
            "target_id": "dry_wet_interface",
            "type": "layer_interface",
            "interface_y_m": 0.082,
            "relative_permittivity": 4.5,
        }
    ]
    return ScenarioDefinition(
        scenario_id="layered_interface_v1",
        label="水平湿度分层界面",
        description="上覆干砂、下伏湿砂的水平界面，主要验证去背景和增益步骤不会把真实层状反射完全抹掉。",
        structure_notes=[
            "上层为相对介电常数 4.5、低电导干砂。",
            "下层 y=0.000-0.082 m 为相对介电常数 10、较高电导湿砂。",
            "真实结构是水平干湿界面，不是点状双曲线目标。",
        ],
        model_in_text=_model_text(
            "layered_interface_v1",
            materials=[
                "#material: 4.5 0.001 1 0 dry_sand",
                "#material: 10 0.012 1 0 wet_sand",
            ],
            bodies=[
                "#box: 0 0 0 0.240 0.165 0.002 dry_sand",
                "#box: 0 0 0 0.240 0.082 0.002 wet_sand",
            ],
        ),
        materials=materials,
        targets=targets,
        layers=layers,
    )


def _crack_air_filled_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "silty_sand", "relative_permittivity": 6.0, "conductivity_s_per_m": 0.002},
        {"name": "air_crack", "relative_permittivity": 1.0, "conductivity_s_per_m": 0.0},
    ]
    targets = [
        {
            "target_id": "air_crack_01",
            "type": "air_crack",
            "x0_m": 0.118,
            "x1_m": 0.124,
            "y0_m": 0.064,
            "y1_m": 0.132,
            "relative_permittivity": 6.0,
        }
    ]
    return ScenarioDefinition(
        scenario_id="crack_air_filled_v1",
        label="空气裂缝弱结构",
        description="均匀粉砂土中的窄空气裂缝，主要验证自动选参不会把窄弱反射和边缘绕射过度平滑。",
        structure_notes=[
            "背景介质为相对介电常数 6 的粉砂土。",
            "裂缝为 x=0.118-0.124 m、y=0.064-0.132 m 的低介电常数窄空腔。",
            "真实结构是窄线状弱反射和裂缝边缘绕射，不应被去噪或背景抑制完全抹掉。",
        ],
        model_in_text=_model_text(
            "crack_air_filled_v1",
            materials=[
                "#material: 6 0.002 1 0 silty_sand",
                "#material: 1 0 1 0 air_crack",
            ],
            bodies=[
                "#box: 0 0 0 0.240 0.165 0.002 silty_sand",
                "#box: 0.118 0.064 0 0.124 0.132 0.002 air_crack",
            ],
        ),
        materials=materials,
        targets=targets,
        layers=[],
    )


def _no_target_background_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "silty_sand", "relative_permittivity": 6.0, "conductivity_s_per_m": 0.002},
    ]
    return ScenarioDefinition(
        scenario_id="no_target_background_v1",
        label="无目标均匀背景",
        description="无地下异常体的均匀粉砂土背景，主要验证自动选参不会凭空制造强局部异常。",
        structure_notes=[
            "场景中只有均匀粉砂土背景，没有金属体、裂缝或层状界面目标。",
            "理想处理结果应降低背景和低频漂移，同时避免在目标外区域制造强假异常。",
        ],
        model_in_text=_model_text(
            "no_target_background_v1",
            materials=["#material: 6 0.002 1 0 silty_sand"],
            bodies=[
                "#box: 0 0 0 0.240 0.165 0.002 silty_sand",
            ],
        ),
        materials=materials,
        targets=[],
        layers=[],
    )


def _model_text(
    title: str,
    *,
    materials: list[str],
    bodies: list[str],
) -> str:
    lines = [
        f"#title: MyGPR {title}",
        f"#domain: {DOMAIN_M[0]:.3f} {DOMAIN_M[1]:.3f} {DOMAIN_M[2]:.3f}",
        f"#dx_dy_dz: {DX_M:.3f} {DX_M:.3f} {DX_M:.3f}",
        f"#time_window: {TOTAL_TIME_NS * 1e-9:.9g}",
        "",
        *materials,
        "",
        "#waveform: ricker 1 1.5e9 my_ricker",
        f"#hertzian_dipole: z {SOURCE_START_M[0]:.3f} {SOURCE_START_M[1]:.3f} 0 my_ricker",
        f"#rx: {RECEIVER_START_M[0]:.3f} {RECEIVER_START_M[1]:.3f} 0",
        f"#src_steps: {TRACE_STEP_M:.3f} 0 0",
        f"#rx_steps: {TRACE_STEP_M:.3f} 0 0",
        "",
        *bodies,
        "",
    ]
    return "\n".join(lines)


def _target_ground_truth(
    definition: ScenarioDefinition,
    target: dict[str, Any],
    samples: int,
    traces: int,
    total_time_ns: float,
) -> dict[str, Any]:
    center = target["center_m"]
    eps = float(target.get("relative_permittivity") or 6.0)
    apex_trace = _target_trace_index(definition, float(center[0]), traces)
    apex_time_ns = _two_way_target_time_ns(definition, center, apex_trace, eps)
    apex_sample = _time_to_sample(apex_time_ns, total_time_ns, samples)
    roi = _target_roi(samples, traces, apex_sample, apex_trace)
    return {
        "target_id": target["target_id"],
        "type": "hyperbola",
        "source_geometry": "metal_cylinder",
        "center_m": list(center),
        "radius_m": float(target["radius_m"]),
        "apex_trace_idx": int(apex_trace),
        "apex_sample_idx": int(apex_sample),
        "apex_time_ns": float(apex_time_ns),
        "roi": roi,
        "must_preserve": True,
        "expected_features": [
            "hyperbola_apex",
            "left_hyperbola_arm",
            "right_hyperbola_arm",
        ],
    }


def _layer_ground_truth(
    definition: ScenarioDefinition,
    target: dict[str, Any],
    samples: int,
    traces: int,
    total_time_ns: float,
) -> dict[str, Any]:
    eps = float(target.get("relative_permittivity") or 4.5)
    interface_y = float(target["interface_y_m"])
    velocity = LIGHT_SPEED_M_PER_NS / np.sqrt(max(eps, 1.0))
    depth = max(0.0, definition.ground_top_y_m - interface_y)
    apex_time_ns = 2.0 * depth / velocity
    sample = _time_to_sample(apex_time_ns, total_time_ns, samples)
    pad = max(12, samples // 80)
    roi = {
        "time_start_idx": max(0, sample - pad),
        "time_end_idx": min(samples, sample + pad + 1),
        "dist_start_idx": 0,
        "dist_end_idx": traces,
    }
    return {
        "target_id": target["target_id"],
        "type": "layer_interface",
        "interface_y_m": interface_y,
        "apex_sample_idx": int(sample),
        "apex_time_ns": float(apex_time_ns),
        "roi": roi,
        "must_preserve": True,
        "expected_features": ["continuous_horizontal_reflector"],
    }


def _crack_ground_truth(
    definition: ScenarioDefinition,
    target: dict[str, Any],
    samples: int,
    traces: int,
    total_time_ns: float,
) -> dict[str, Any]:
    eps = float(target.get("relative_permittivity") or 6.0)
    x0 = float(target["x0_m"])
    x1 = float(target["x1_m"])
    y0 = float(target["y0_m"])
    y1 = float(target["y1_m"])
    x_center = (x0 + x1) / 2.0
    trace_center = _target_trace_index(definition, x_center, traces)
    trace_half_width = max(3, min(8, traces // 8))

    velocity = LIGHT_SPEED_M_PER_NS / np.sqrt(max(eps, 1.0))
    top_depth = max(0.0, definition.ground_top_y_m - max(y0, y1))
    bottom_depth = max(0.0, definition.ground_top_y_m - min(y0, y1))
    top_sample = _time_to_sample(2.0 * top_depth / velocity, total_time_ns, samples)
    bottom_sample = _time_to_sample(2.0 * bottom_depth / velocity, total_time_ns, samples)
    pad = max(6, samples // 60)
    t0 = max(0, min(top_sample, bottom_sample) - pad)
    t1 = min(samples, max(top_sample, bottom_sample) + pad + 1)
    roi = {
        "time_start_idx": int(t0),
        "time_end_idx": int(max(t0 + 1, t1)),
        "dist_start_idx": max(0, int(trace_center - trace_half_width)),
        "dist_end_idx": min(traces, int(trace_center + trace_half_width + 1)),
    }
    return {
        "target_id": target["target_id"],
        "type": "air_crack",
        "source_geometry": "air_filled_box",
        "x0_m": x0,
        "x1_m": x1,
        "y0_m": y0,
        "y1_m": y1,
        "center_trace_idx": int(trace_center),
        "top_sample_idx": int(top_sample),
        "bottom_sample_idx": int(bottom_sample),
        "roi": roi,
        "must_preserve": True,
        "expected_features": [
            "narrow_vertical_reflector",
            "weak_diffraction_edges",
        ],
    }


def _target_trace_index(
    definition: ScenarioDefinition,
    target_x_m: float,
    traces: int,
) -> int:
    midpoint_start = (definition.source_start_m[0] + definition.receiver_start_m[0]) / 2.0
    trace_idx = int(round((target_x_m - midpoint_start) / definition.trace_step_m))
    return max(0, min(trace_idx, max(traces - 1, 0)))


def _two_way_target_time_ns(
    definition: ScenarioDefinition,
    center: list[float],
    trace_idx: int,
    eps: float,
) -> float:
    tx_x = definition.source_start_m[0] + trace_idx * definition.trace_step_m
    rx_x = definition.receiver_start_m[0] + trace_idx * definition.trace_step_m
    tx_y = definition.source_start_m[1]
    rx_y = definition.receiver_start_m[1]
    target_x = float(center[0])
    target_y = float(center[1])
    dist_tx = float(np.hypot(tx_x - target_x, tx_y - target_y))
    dist_rx = float(np.hypot(rx_x - target_x, rx_y - target_y))
    velocity = LIGHT_SPEED_M_PER_NS / np.sqrt(max(eps, 1.0))
    return (dist_tx + dist_rx) / velocity


def _time_to_sample(time_ns: float, total_time_ns: float, samples: int) -> int:
    if total_time_ns <= 0.0:
        return max(0, min(samples // 2, samples - 1))
    idx = int(round(float(time_ns) / float(total_time_ns) * max(samples - 1, 1)))
    return max(0, min(idx, max(samples - 1, 0)))


def _target_roi(
    samples: int,
    traces: int,
    apex_sample: int,
    apex_trace: int,
) -> dict[str, int]:
    half_width = max(4, min(18, traces // 3))
    before = max(10, min(60, samples // 12))
    after = max(18, min(110, samples // 7))
    return {
        "time_start_idx": max(0, int(apex_sample - before)),
        "time_end_idx": min(samples, int(apex_sample + after)),
        "dist_start_idx": max(0, int(apex_trace - half_width)),
        "dist_end_idx": min(traces, int(apex_trace + half_width + 1)),
    }


def _union_rois(rois: list[dict[str, int]], samples: int, traces: int) -> dict[str, int]:
    return {
        "time_start_idx": max(0, min(int(roi["time_start_idx"]) for roi in rois)),
        "time_end_idx": min(samples, max(int(roi["time_end_idx"]) for roi in rois)),
        "dist_start_idx": max(0, min(int(roi["dist_start_idx"]) for roi in rois)),
        "dist_end_idx": min(traces, max(int(roi["dist_end_idx"]) for roi in rois)),
    }


def _full_roi(samples: int, traces: int) -> dict[str, int]:
    return {
        "time_start_idx": 0,
        "time_end_idx": int(samples),
        "dist_start_idx": 0,
        "dist_end_idx": int(traces),
    }


def _resolve_pipeline(baseline_profile_key: str) -> list[str]:
    profile = RECOMMENDED_RUN_PROFILES.get(baseline_profile_key)
    if not profile:
        raise ValueError(f"Unknown baseline profile: {baseline_profile_key}")
    profile_order = [str(method_key) for method_key in profile.get("order", [])]
    if baseline_profile_key == DEFAULT_PROFILE_KEY:
        pipeline = list(REPORT_PIPELINE_ORDER)
    else:
        pipeline = ["agcGain" if item == "sec_gain" else item for item in profile_order]
    missing = [method_key for method_key in pipeline if method_key not in PROCESSING_METHODS]
    if missing:
        raise ValueError(f"Unknown processing method(s): {', '.join(missing)}")
    return pipeline


def _resolve_manual_params(
    pipeline: list[str],
    baseline_profile_key: str,
) -> dict[str, dict[str, Any]]:
    profile = RECOMMENDED_RUN_PROFILES.get(baseline_profile_key, {})
    preset_key = profile.get("preset_key")
    defaults: dict[str, dict[str, Any]] = {}
    if preset_key and preset_key in GUI_PRESETS_V1:
        for method_key, params in GUI_PRESETS_V1[preset_key].get("method_params", {}).items():
            defaults[str(method_key)] = dict(params)
    for method_key, params in profile.get("method_params", {}).items():
        defaults[str(method_key)] = dict(params)
    for method_key, params in REPORT_MANUAL_PARAM_OVERRIDES.items():
        defaults[str(method_key)] = dict(params)
    return {method_key: dict(defaults.get(method_key, {})) for method_key in pipeline}


def _final_metric_summary(
    raw: np.ndarray,
    manual_final: np.ndarray,
    auto_final: np.ndarray,
    raw_roi: dict[str, int],
    manual_roi: dict[str, int],
    auto_roi: dict[str, int],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    manual_metrics = _branch_metrics(
        raw,
        manual_final,
        raw_roi,
        manual_roi,
        ground_truth,
    )
    auto_metrics = _branch_metrics(
        raw,
        auto_final,
        raw_roi,
        auto_roi,
        ground_truth,
    )
    manual_score = _comparison_score(manual_metrics)
    auto_score = _comparison_score(auto_metrics)
    manual_metrics["comparison_score"] = float(manual_score)
    auto_metrics["comparison_score"] = float(auto_score)
    delta = {
        key: float(auto_metrics[key] - manual_metrics[key])
        for key in sorted(set(manual_metrics) & set(auto_metrics))
        if np.isfinite(manual_metrics[key]) and np.isfinite(auto_metrics[key])
    }
    return {
        "manual": {key: float(value) for key, value in manual_metrics.items()},
        "auto": {key: float(value) for key, value in auto_metrics.items()},
        "delta_auto_minus_manual": delta,
    }


def _step_metric_summary(
    manual_input: np.ndarray,
    manual_output: np.ndarray,
    auto_input: np.ndarray,
    auto_output: np.ndarray,
    manual_input_roi: dict[str, int],
    manual_output_roi: dict[str, int],
    auto_input_roi: dict[str, int],
    auto_output_roi: dict[str, int],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    manual_metrics = _branch_metrics(
        manual_input,
        manual_output,
        manual_input_roi,
        manual_output_roi,
        ground_truth,
    )
    auto_metrics = _branch_metrics(
        auto_input,
        auto_output,
        auto_input_roi,
        auto_output_roi,
        ground_truth,
    )
    manual_metrics["comparison_score"] = _comparison_score(manual_metrics)
    auto_metrics["comparison_score"] = _comparison_score(auto_metrics)
    delta = {
        key: float(auto_metrics[key] - manual_metrics[key])
        for key in sorted(set(manual_metrics) & set(auto_metrics))
        if np.isfinite(manual_metrics[key]) and np.isfinite(auto_metrics[key])
    }
    return {
        "manual": {key: float(value) for key, value in manual_metrics.items()},
        "auto": {key: float(value) for key, value in auto_metrics.items()},
        "delta_auto_minus_manual": delta,
    }


def _branch_metrics(
    before: np.ndarray,
    after: np.ndarray,
    before_roi: dict[str, int],
    after_roi: dict[str, int],
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, float]:
    before_data, after_data = _slice_pair_rois(before, after, before_roi, after_roi)
    metrics = compute_benchmark_metrics(before_data, after_data)
    if ground_truth:
        metrics.update(
            compute_ground_truth_metrics(
                before,
                after,
                ground_truth,
                reference_roi=before_roi,
                processed_roi=after_roi,
            )
        )
    return metrics


def _slice_pair_rois(
    before: np.ndarray,
    after: np.ndarray,
    before_roi: dict[str, int],
    after_roi: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    before_slice = _slice_roi(before, before_roi)
    after_slice = _slice_roi(after, after_roi)
    rows = max(1, min(before_slice.shape[0], after_slice.shape[0]))
    cols = max(1, min(before_slice.shape[1], after_slice.shape[1]))
    return before_slice[:rows, :cols], after_slice[:rows, :cols]


def _roi_after_method(
    roi: dict[str, int],
    method_key: str,
    meta: dict[str, Any],
    shape: tuple[int, int],
) -> dict[str, int]:
    if method_key != "set_zero_time":
        return _clamp_roi(roi, shape)
    shift = int(meta.get("shift_samples") or 0)
    shifted = {
        "time_start_idx": int(roi.get("time_start_idx", 0)) - shift,
        "time_end_idx": int(roi.get("time_end_idx", shape[0])) - shift,
        "dist_start_idx": int(roi.get("dist_start_idx", 0)),
        "dist_end_idx": int(roi.get("dist_end_idx", shape[1])),
    }
    return _clamp_roi(shifted, shape)


def _clamp_roi(roi: dict[str, int], shape: tuple[int, int]) -> dict[str, int]:
    samples, traces = int(shape[0]), int(shape[1])
    t0 = max(0, min(int(roi.get("time_start_idx", 0)), max(samples - 1, 0)))
    t1 = max(t0 + 1, min(int(roi.get("time_end_idx", samples)), samples))
    d0 = max(0, min(int(roi.get("dist_start_idx", 0)), max(traces - 1, 0)))
    d1 = max(d0 + 1, min(int(roi.get("dist_end_idx", traces)), traces))
    return {
        "time_start_idx": int(t0),
        "time_end_idx": int(t1),
        "dist_start_idx": int(d0),
        "dist_end_idx": int(d1),
    }


def _build_step_analysis(
    *,
    method_key: str,
    method_name: str,
    metrics: dict[str, Any],
    ground_truth: dict[str, Any],
    policy_notes: list[str],
) -> dict[str, str]:
    delta = metrics.get("delta_auto_minus_manual", {})
    manual = metrics.get("manual", {})
    auto = metrics.get("auto", {})
    structure_label = _structure_label(ground_truth)
    score_delta = float(delta.get("comparison_score", 0.0))
    visual_winner = _winner_label(score_delta)
    scale_note = ""
    if any(abs(float(value)) >= 1000.0 for value in delta.values() if _is_number(value)):
        scale_note = " 其中极端比值通常表示该 ROI 的参考能量接近 0，需要结合图像判断。"

    if method_key == "set_zero_time" and policy_notes:
        visual = (
            f"{' '.join(policy_notes)} 因此本步视觉上不判定人工/自动优劣，"
            f"重点检查 {structure_label} 是否仍落在有效时窗内，并为后续步骤提供公平输入。"
        )
    else:
        visual = (
            f"基于真实结构 ROI，{visual_winner}。视觉关注点是 {structure_label} "
            f"的连续性/边缘是否保留，以及背景或深部噪声是否被过度增强。"
        )

    metric = (
        f"{method_name} 的自动-人工 score 差值为 {_fmt_metric(score_delta)}；"
        f"真值评分差值 {_fmt_metric(delta.get('truth_score'))}，"
        f"目标能量保留差值 {_fmt_metric(delta.get('truth_target_energy_preservation'))}，"
        f"目标外背景抑制差值 {_fmt_metric(delta.get('truth_background_energy_reduction'))}，"
        f"假异常比例差值 {_fmt_metric(delta.get('truth_false_positive_ratio'))}；"
        f"低频能量降低差值 {_fmt_metric(delta.get('low_freq_energy_reduction'))}，"
        f"水平相干降低差值 {_fmt_metric(delta.get('horizontal_coherence_reduction'))}，"
        f"目标频带保真差值 {_fmt_metric(delta.get('target_band_energy_ratio'))}，"
        f"边缘保真差值 {_fmt_metric(delta.get('edge_preservation'))}。"
        f"人工 score={_fmt_metric(manual.get('comparison_score'))}，"
        f"自动 score={_fmt_metric(auto.get('comparison_score'))}。"
        f"{scale_note}"
    )
    return {"visual": visual, "metrics": metric}


def _structure_label(ground_truth: dict[str, Any]) -> str:
    target_types = {str(item.get("type")) for item in ground_truth.get("targets", [])}
    if not target_types:
        return "无目标背景区域"
    if "layer_interface" in target_types:
        return "水平层状反射界面"
    if "air_crack" in target_types:
        return "空气裂缝弱结构"
    if "hyperbola" in target_types:
        return "双曲线目标"
    return "真实正演结构"


def _winner_label(score_delta: float) -> str:
    if score_delta > 0.02:
        return "自动选参结果在综合指标上优于人工参数"
    if score_delta < -0.02:
        return "人工参数在综合指标上优于自动选参"
    return "两者综合指标接近"


def _comparison_score(metrics: dict[str, float]) -> float:
    band_fidelity = ratio_fidelity(metrics["target_band_energy_ratio"], tol=0.35)
    saliency_fidelity = ratio_fidelity(metrics["local_saliency_preservation"], tol=0.35)
    edge_fidelity = ratio_fidelity(metrics["edge_preservation"], tol=0.35)
    deep_gain = max(0.0, float(metrics["deep_zone_contrast_gain"]) - 1.0)
    target_loss_penalty = (
        max(0.0, 0.55 - float(metrics["target_band_energy_ratio"])) * 3.0
        + max(0.0, 0.55 - float(metrics["local_saliency_preservation"])) * 4.0
        + max(0.0, 0.55 - float(metrics["edge_preservation"])) * 3.0
    )
    artifact_penalty = (
        6.0 * float(metrics["clipping_ratio_after"])
        + 4.0 * float(metrics["hot_pixel_ratio_after"])
        + 0.08 * float(metrics["kurtosis_or_spikiness_after"])
    )
    truth_score = float(metrics.get("truth_score", 0.0))
    return float(
        1.2 * float(metrics["baseline_bias_reduction"])
        + 1.4 * float(metrics["low_freq_energy_reduction"])
        + 0.8 * float(metrics["horizontal_coherence_reduction"])
        + 1.8 * band_fidelity
        + 2.0 * saliency_fidelity
        + 1.4 * edge_fidelity
        + 0.4 * np.log1p(deep_gain)
        + 1.4 * truth_score
        - target_loss_penalty
        - artifact_penalty
    )


def _comparison_verdict(metric_summary: dict[str, Any]) -> str:
    score_delta = float(
        metric_summary.get("delta_auto_minus_manual", {}).get("comparison_score", 0.0)
    )
    if score_delta > 0.02:
        return "auto_better"
    if score_delta < -0.02:
        return "manual_better"
    return "tie"


def _slice_roi(data: np.ndarray, roi: dict[str, int]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    t0 = max(0, min(int(roi.get("time_start_idx", 0)), arr.shape[0] - 1))
    t1 = max(t0 + 1, min(int(roi.get("time_end_idx", arr.shape[0])), arr.shape[0]))
    d0 = max(0, min(int(roi.get("dist_start_idx", 0)), arr.shape[1] - 1))
    d1 = max(d0 + 1, min(int(roi.get("dist_end_idx", arr.shape[1])), arr.shape[1]))
    return arr[t0:t1, d0:d1]


def _locked_vlim(arrays: list[np.ndarray]) -> float:
    values = np.concatenate([np.ravel(np.asarray(arr, dtype=np.float32)) for arr in arrays])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    vlim = float(np.percentile(np.abs(finite), 99.3))
    if not np.isfinite(vlim) or vlim <= 0.0:
        return 1.0
    return vlim


def _comparison_without_arrays(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline": list(comparison["pipeline"]),
        "manual_params_by_method": _json_safe(comparison["manual_params_by_method"]),
        "auto_params_by_method": _json_safe(comparison["auto_params_by_method"]),
        "auto_tune_results": _json_safe(comparison["auto_tune_results"]),
        "roi_spec": _json_safe(comparison["roi_spec"]),
        "final_manual_roi": _json_safe(comparison.get("final_manual_roi", {})),
        "final_auto_roi": _json_safe(comparison.get("final_auto_roi", {})),
        "zero_time_policy": comparison.get("zero_time_policy"),
        "metrics": _json_safe(comparison["metrics"]),
        "verdict": comparison["verdict"],
        "steps": [
            {
                "method_key": step["method_key"],
                "method_name": step["method_name"],
                "manual_params": _json_safe(step["manual_params"]),
                "manual_original_params": _json_safe(
                    step.get("manual_original_params", {})
                ),
                "auto_params": _json_safe(step["auto_params"]),
                "manual_warnings": list(step["manual_warnings"]),
                "auto_warnings": list(step["auto_warnings"]),
                "policy_notes": list(step.get("policy_notes", [])),
                "metrics": _json_safe(step.get("metrics", {})),
                "analysis": _json_safe(step.get("analysis", {})),
                "auto_tune_summary": _json_safe(step["auto_tune_summary"]),
            }
            for step in comparison["steps"]
        ],
    }


def _compact_auto_tune_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_key": result.get("method_key"),
        "method_name": result.get("method_name"),
        "family": result.get("family"),
        "recommended_profile": result.get("recommended_profile"),
        "recommended_params": _json_safe(result.get("recommended_params", {})),
        "best_params": _json_safe(result.get("best_params", {})),
        "best_score": _json_safe(result.get("best_score")),
        "best_reason": result.get("best_reason"),
        "roi_info": _json_safe(result.get("roi_info", {})),
        "execution_stats": _json_safe(result.get("execution_stats", {})),
    }


def _extract_warning_messages(meta: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    for warning in meta.get("runtime_warnings", []) or []:
        if isinstance(warning, dict):
            messages.append(str(warning.get("message") or warning.get("code") or warning))
        else:
            messages.append(str(warning))
    for warning in meta.get("warnings", []) or []:
        messages.append(str(warning))
    if meta.get("skipped"):
        messages.append(str(meta.get("reason") or "method skipped"))
    return messages


def _render_scenario_section(record: dict[str, Any]) -> str:
    comparison = record.get("comparison", {})
    metrics = comparison.get("metrics", {})
    delta = metrics.get("delta_auto_minus_manual", {})
    verdict = comparison.get("verdict")
    verdict_label = {
        "auto_better": "自动选参更优",
        "manual_better": "人工 baseline 更优",
        "tie": "二者接近",
    }.get(str(verdict), str(verdict))
    notes = "".join(f"<li>{_esc(note)}</li>" for note in record.get("structure_notes", []))
    step_panels = "\n".join(_render_step_panel(step) for step in record.get("images", []))
    command = " ".join(str(part) for part in record.get("gprmax", {}).get("command", []))
    return f"""
    <section class="scenario">
      <div class="scenario-heading">
        <div>
          <p class="eyebrow">{_esc(record.get("scenario_id"))}</p>
          <h2>{_esc(record.get("label"))}</h2>
        </div>
        <span class="verdict">{_esc(verdict_label)}</span>
      </div>
      <p>{_esc(record.get("description"))}</p>

      <h3>真实地质结构</h3>
      <div class="structure-grid">
        <img src="{_esc(record.get("structure_preview"))}" alt="{_esc(record.get("label"))} true structure">
        <div>
          <ul>{notes}</ul>
          <p class="note">Ground truth ROI 来自正演结构的已知几何位置，用于评价目标保真与背景抑制，不依赖人工从真实数据中猜测双曲线位置。</p>
        </div>
      </div>

      <h3>gprMax 命令</h3>
      <pre>{_esc(command)}</pre>

      <h3>最终指标摘要</h3>
      <div class="metric-grid">
        <div><span>人工选参 score</span><strong>{_fmt_metric(metrics.get("manual", {}).get("comparison_score"))}</strong></div>
        <div><span>自动选参 score</span><strong>{_fmt_metric(metrics.get("auto", {}).get("comparison_score"))}</strong></div>
        <div><span>score 差值</span><strong>{_fmt_metric(delta.get("comparison_score"))}</strong></div>
        <div><span>真值评分差值</span><strong>{_fmt_metric(delta.get("truth_score"))}</strong></div>
        <div><span>目标能量保留差值</span><strong>{_fmt_metric(delta.get("truth_target_energy_preservation"))}</strong></div>
        <div><span>目标外背景抑制差值</span><strong>{_fmt_metric(delta.get("truth_background_energy_reduction"))}</strong></div>
        <div><span>假异常比例差值</span><strong>{_fmt_metric(delta.get("truth_false_positive_ratio"))}</strong></div>
        <div><span>低频能量降低差值</span><strong>{_fmt_metric(delta.get("low_freq_energy_reduction"))}</strong></div>
        <div><span>目标频带保真差值</span><strong>{_fmt_metric(delta.get("target_band_energy_ratio"))}</strong></div>
        <div><span>边缘保真差值</span><strong>{_fmt_metric(delta.get("edge_preservation"))}</strong></div>
      </div>

      <h3>逐步骤 BScan 对比</h3>
      {step_panels}
    </section>
"""


def _render_overall_summary(payload: dict[str, Any]) -> str:
    scenarios = list(payload.get("scenarios", []))
    verdicts = [str(item.get("comparison", {}).get("verdict")) for item in scenarios]
    auto_count = sum(1 for verdict in verdicts if verdict == "auto_better")
    manual_count = sum(1 for verdict in verdicts if verdict == "manual_better")
    tie_count = sum(1 for verdict in verdicts if verdict == "tie")
    rows = "\n".join(_render_overall_row(item) for item in scenarios)
    conclusion = (
        f"本次共 {len(scenarios)} 个正演场景，自动选参胜出 {auto_count} 个，"
        f"人工 baseline 胜出 {manual_count} 个，接近持平 {tie_count} 个。"
    )
    if manual_count:
        conclusion += " 未胜出的场景应作为后续改进 auto-tune 评分函数的回归样例，而不是从报告中删除。"
    return f"""
    <section>
      <h2>总体结论</h2>
      <p>{_esc(conclusion)}</p>
      <table class="params-table">
        <thead><tr><th>场景</th><th>判定</th><th>人工 score</th><th>自动 score</th><th>score 差值</th><th>真值评分差值</th><th>目标能量保留差值</th></tr></thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </section>
"""


def _render_overall_row(record: dict[str, Any]) -> str:
    comparison = record.get("comparison", {})
    metrics = comparison.get("metrics", {})
    delta = metrics.get("delta_auto_minus_manual", {})
    verdict = comparison.get("verdict")
    verdict_label = {
        "auto_better": "自动选参更优",
        "manual_better": "人工 baseline 更优",
        "tie": "二者接近",
    }.get(str(verdict), str(verdict))
    return f"""
            <tr>
              <td>{_esc(record.get("label"))} <code>{_esc(record.get("scenario_id"))}</code></td>
              <td>{_esc(verdict_label)}</td>
              <td>{_fmt_metric(metrics.get("manual", {}).get("comparison_score"))}</td>
              <td>{_fmt_metric(metrics.get("auto", {}).get("comparison_score"))}</td>
              <td>{_fmt_metric(delta.get("comparison_score"))}</td>
              <td>{_fmt_metric(delta.get("truth_score"))}</td>
              <td>{_fmt_metric(delta.get("truth_target_energy_preservation"))}</td>
            </tr>
"""


def _render_step_panel(step: dict[str, Any]) -> str:
    images = step.get("images", {})
    params_table = _render_params_table(step)
    warnings = _render_warnings(step)
    analysis = step.get("analysis") or {}
    return f"""
      <article class="step-card">
        <h4>{_esc(step.get("method_name"))} <code>{_esc(step.get("method_key"))}</code></h4>
        {params_table}
        <div class="analysis-grid">
          <div><strong>视觉评价</strong><p>{_esc(analysis.get("visual") or "暂无")}</p></div>
          <div><strong>指标评价</strong><p>{_esc(analysis.get("metrics") or "暂无")}</p></div>
        </div>
        {warnings}
        <div class="image-grid">
          <figure><img src="{_esc(images.get("manual_input"))}" alt="manual before"><figcaption>人工分支运行前</figcaption></figure>
          <figure><img src="{_esc(images.get("auto_input"))}" alt="auto before"><figcaption>自动分支运行前</figcaption></figure>
          <figure><img src="{_esc(images.get("manual_output"))}" alt="manual output"><figcaption>人工选参后</figcaption></figure>
          <figure><img src="{_esc(images.get("auto_output"))}" alt="auto output"><figcaption>自动选参后</figcaption></figure>
        </div>
      </article>
"""


def _render_params_table(step: dict[str, Any]) -> str:
    manual = json.dumps(step.get("manual_params", {}), ensure_ascii=False, sort_keys=True)
    manual_original = json.dumps(
        step.get("manual_original_params", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    auto = json.dumps(step.get("auto_params", {}), ensure_ascii=False, sort_keys=True)
    tune = step.get("auto_tune_summary") or {}
    reason = tune.get("best_reason") or tune.get("recommended_profile") or tune.get("error") or ""
    manual_note = "经验 baseline 或日常页面同步参数"
    if step.get("manual_original_params") != step.get("manual_params"):
        manual_note = f"本步实际参数已按报告策略修正；原人工参数为 {manual_original}"
    return f"""
        <table class="params-table">
          <thead><tr><th>分支</th><th>参数</th><th>说明</th></tr></thead>
          <tbody>
            <tr><td>人工选参</td><td><code>{_esc(manual)}</code></td><td>{_esc(manual_note)}</td></tr>
            <tr><td>自动选参</td><td><code>{_esc(auto)}</code></td><td>{_esc(reason)}</td></tr>
          </tbody>
        </table>
"""


def _render_warnings(step: dict[str, Any]) -> str:
    warnings = list(step.get("manual_warnings") or []) + list(step.get("auto_warnings") or [])
    if not warnings:
        return ""
    items = "".join(f"<li>{_esc(item)}</li>" for item in warnings)
    return f"<div class=\"warnings\"><strong>运行提示</strong><ul>{items}</ul></div>"


def _html_css() -> str:
    return """
body {
  margin: 0;
  font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
  color: #1f2933;
  background: #f6f7f9;
}
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px 20px 56px;
}
section {
  margin: 24px 0;
  padding: 22px;
  background: #ffffff;
  border: 1px solid #d9dee7;
  border-radius: 8px;
}
.hero {
  background: #20242a;
  color: #ffffff;
}
.hero p {
  max-width: 850px;
}
.eyebrow {
  margin: 0 0 8px;
  color: #5c7cfa;
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
}
.hero .eyebrow {
  color: #91a7ff;
}
h1, h2, h3, h4 {
  margin: 0 0 12px;
  letter-spacing: 0;
}
h1 {
  font-size: 32px;
}
h2 {
  font-size: 24px;
}
h3 {
  margin-top: 22px;
  font-size: 18px;
}
h4 {
  font-size: 16px;
}
p, li {
  line-height: 1.65;
}
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  padding: 12px;
  background: #f1f3f5;
  border-radius: 6px;
}
code {
  font-family: Consolas, "Liberation Mono", monospace;
}
.kv-grid, .metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}
.kv-grid div, .metric-grid div {
  padding: 12px;
  background: #f8f9fb;
  border: 1px solid #e4e7ec;
  border-radius: 6px;
}
.kv-grid span, .metric-grid span {
  display: block;
  margin-bottom: 5px;
  color: #667085;
  font-size: 13px;
}
.scenario-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}
.verdict {
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 4px 10px;
  border-radius: 6px;
  color: #0b5e36;
  background: #d3f9d8;
  font-weight: 700;
  white-space: nowrap;
}
.structure-grid {
  display: grid;
  grid-template-columns: minmax(280px, 420px) minmax(260px, 1fr);
  gap: 18px;
  align-items: start;
}
.structure-grid img {
  width: 100%;
  border: 1px solid #d9dee7;
  border-radius: 6px;
  background: #ffffff;
}
.step-card {
  margin-top: 16px;
  padding: 16px;
  border: 1px solid #d9dee7;
  border-radius: 8px;
  background: #fcfcfd;
}
.params-table {
  width: 100%;
  border-collapse: collapse;
  margin: 8px 0 14px;
  font-size: 13px;
}
.params-table th, .params-table td {
  padding: 8px;
  border: 1px solid #d9dee7;
  vertical-align: top;
}
.params-table th {
  background: #eef2f7;
  text-align: left;
}
.image-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(260px, 1fr));
  gap: 12px;
}
.analysis-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(260px, 1fr));
  gap: 12px;
  margin: 12px 0;
}
.analysis-grid div {
  padding: 10px 12px;
  background: #f8f9fb;
  border: 1px solid #e4e7ec;
  border-radius: 6px;
}
.analysis-grid p {
  margin: 6px 0 0;
  font-size: 14px;
}
figure {
  margin: 0;
}
figure img {
  display: block;
  width: 100%;
  border: 1px solid #d9dee7;
  border-radius: 6px;
  background: #ffffff;
}
figcaption {
  margin-top: 6px;
  color: #475467;
  font-size: 13px;
}
.note {
  color: #667085;
  font-size: 14px;
}
.warnings {
  margin: 10px 0;
  padding: 10px 12px;
  background: #fff4e6;
  border: 1px solid #ffd8a8;
  border-radius: 6px;
}
@media (max-width: 820px) {
  main {
    padding: 18px 12px 36px;
  }
  .structure-grid, .image-grid, .analysis-grid {
    grid-template-columns: 1fr;
  }
  .scenario-heading {
    display: block;
  }
}
"""


def _fmt_metric(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    if abs(number) >= 1000.0:
        return f"{number:.3e}"
    return f"{number:.4f}"


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number))


def _relpath(path: Path | str, base: Path) -> str:
    return os.path.relpath(str(path), str(base)).replace("\\", "/")


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-scenario gprMax validation and generate a MyGPR HTML report."
    )
    parser.add_argument("--gprmax-root", default=str(DEFAULT_GPRMAX_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario id to run; can be repeated. Default: all scenarios.",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--geometry-fixed",
        action="store_true",
        help="Pass --geometry-fixed to gprMax.",
    )
    parser.add_argument("--mpi", type=int, default=None, help="Optional gprMax -mpi value.")
    parser.add_argument(
        "--gpu",
        nargs="+",
        default=[],
        help="Optional gprMax -gpu device ids, for example --gpu 0.",
    )
    parser.add_argument("--python-exe", default=None)
    parser.add_argument(
        "--search-mode",
        choices=["fast", "standard", "thorough"],
        default="fast",
    )
    parser.add_argument("--baseline-profile", default=DEFAULT_PROFILE_KEY)
    parser.add_argument(
        "--zero-time-policy",
        choices=["align_auto", "manual", "skip"],
        default="align_auto",
        help=(
            "Zero-time handling for the report. align_auto copies auto-tuned "
            "zero-time into the manual branch for a fair downstream comparison."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument appended to the gprMax command; can be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_multi_scenario_report(
        gprmax_root=Path(args.gprmax_root),
        output_root=Path(args.output_root),
        scenario_ids=list(args.scenario or []) or None,
        runs=int(args.runs),
        geometry_fixed=bool(args.geometry_fixed),
        mpi=args.mpi,
        gpu=list(args.gpu or []),
        python_override=args.python_exe,
        search_mode=str(args.search_mode),
        baseline_profile_key=str(args.baseline_profile),
        zero_time_policy=str(args.zero_time_policy),
        extra_args=list(args.extra_arg or []),
    )
    print(f"HTML report: {payload['html_report']}")
    print(f"Summary JSON: {payload['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
