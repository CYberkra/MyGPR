#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run multi-scenario gprMax validation and build an HTML auto-tune report."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
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
from core.auto_tune_pipeline import (  # noqa: E402
    AutoTunePipelineRun,
    PipelineStepRecord,
    run_auto_tune_pipeline,
)
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
from core.trace_metadata_utils import resample_bscan_columns_linear  # noqa: E402


DEFAULT_GPRMAX_ROOT = Path(r"E:\gprMax\gprMax-v.3.1.7")
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "gprmax_multi_scenario_reports"
DEFAULT_PROFILE_KEY = "uav_gpr_experience_baseline_v1"
DEFAULT_RUNS = 96
DEFAULT_SCENARIO_FAMILY = "airborne"
DEFAULT_ASCAN_SAMPLE_COUNT = 501
RECOMMENDED_ASCAN_SAMPLE_COUNTS = (501, 701)
REPORT_PIPELINE_ORDER = [
    "set_zero_time",
    "dewow",
    "frequency_filter_1d",
    "subtracting_average_2D",
    "sec_gain",
    "wavelet_svd",
]
REPORT_MANUAL_PARAM_OVERRIDES: dict[str, dict[str, Any]] = {
    "frequency_filter_1d": {
        "filter_type": "bandpass",
        "low_freq_mhz": 10.0,
        "high_freq_mhz": 3000.0,
        "taper_ratio": 0.08,
    },
    "wavelet_svd": {
        "wavelet": "db4",
        "levels": 2,
        "threshold": 0.05,
        "threshold_strategy": "mad_universal",
        "rank_start": 1,
        "rank_end": 20,
    },
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

AIRBORNE_DOMAIN_M = (0.720, 0.560, 0.002)
AIRBORNE_DX_M = 0.002
AIRBORNE_TOTAL_TIME_NS = 18.0
AIRBORNE_TRACE_STEP_M = 0.004
AIRBORNE_GROUND_TOP_Y_M = 0.300
AIRBORNE_ANTENNA_HEIGHT_M = 0.120
AIRBORNE_SOURCE_START_X_M = 0.140
AIRBORNE_SOURCE_RECEIVER_OFFSET_M = 0.080
AIRBORNE_DEFAULT_RUNS = 96
AIRBORNE_PML_MARGIN_CELLS = 15
AIRBORNE_SAFE_MARGIN_M = 0.060
AIRBORNE_TOP_FREE_SPACE_M = 0.080


@dataclass(frozen=True)
class AirborneGeometry:
    """Shared air-launched 2D TMz geometry for UAV-GPR gprMax scenes."""

    domain_m: tuple[float, float, float] = AIRBORNE_DOMAIN_M
    dx_m: float = AIRBORNE_DX_M
    total_time_ns: float = AIRBORNE_TOTAL_TIME_NS
    ground_top_y_m: float = AIRBORNE_GROUND_TOP_Y_M
    antenna_height_m: float = AIRBORNE_ANTENNA_HEIGHT_M
    source_start_x_m: float = AIRBORNE_SOURCE_START_X_M
    source_receiver_offset_m: float = AIRBORNE_SOURCE_RECEIVER_OFFSET_M
    trace_step_m: float = AIRBORNE_TRACE_STEP_M
    default_runs: int = AIRBORNE_DEFAULT_RUNS
    pml_margin_cells: int = AIRBORNE_PML_MARGIN_CELLS
    safe_margin_m: float = AIRBORNE_SAFE_MARGIN_M
    top_free_space_m: float = AIRBORNE_TOP_FREE_SPACE_M

    @property
    def source_start_m(self) -> tuple[float, float, float]:
        return (
            float(self.source_start_x_m),
            float(self.ground_top_y_m + self.antenna_height_m),
            0.0,
        )

    @property
    def receiver_start_m(self) -> tuple[float, float, float]:
        return (
            float(self.source_start_x_m + self.source_receiver_offset_m),
            float(self.ground_top_y_m + self.antenna_height_m),
            0.0,
        )

    @property
    def air_layer_thickness_m(self) -> float:
        return float(self.domain_m[1] - self.ground_top_y_m)

    @property
    def top_clearance_m(self) -> float:
        return float(self.domain_m[1] - self.source_start_m[1])

    @property
    def top_clearance_cells(self) -> int:
        return int(round(self.top_clearance_m / self.dx_m))

    def height_profile(self, runs: int, *, variable: bool = False) -> list[float]:
        """Return per-trace antenna heights in metres above the ground surface."""
        count = max(1, int(runs))
        if not variable:
            return [float(self.antenna_height_m)] * count
        if count == 1:
            return [float(self.antenna_height_m)]
        idx = np.arange(count, dtype=np.float64)
        profile = self.antenna_height_m + 0.035 * np.sin(
            2.0 * np.pi * idx / float(count - 1)
        )
        return [float(value) for value in profile]

    def trace_positions(
        self,
        runs: int,
        *,
        variable_height: bool = False,
    ) -> list[dict[str, tuple[float, float, float] | float]]:
        """Return Tx/Rx coordinates and antenna height for each trace."""
        positions: list[dict[str, tuple[float, float, float] | float]] = []
        heights = self.height_profile(runs, variable=variable_height)
        for idx, height_m in enumerate(heights):
            source_x = self.source_start_x_m + idx * self.trace_step_m
            receiver_x = source_x + self.source_receiver_offset_m
            y = self.ground_top_y_m + height_m
            positions.append(
                {
                    "height_m": float(height_m),
                    "source": (float(source_x), float(y), 0.0),
                    "receiver": (float(receiver_x), float(y), 0.0),
                }
            )
        return positions

    def validate(
        self,
        *,
        runs: int,
        targets: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Validate gprMax geometry safety margins and return warnings."""
        warnings: list[str] = []
        if abs(float(self.domain_m[2]) - float(self.dx_m)) > 1.0e-12:
            warnings.append("domain_z_not_equal_dx")
        if self.top_clearance_cells < 20:
            warnings.append("source_top_clearance_less_than_20_cells")
        positions = self.trace_positions(runs)
        min_x = min(
            min(float(pos["source"][0]), float(pos["receiver"][0]))  # type: ignore[index]
            for pos in positions
        )
        max_x = max(
            max(float(pos["source"][0]), float(pos["receiver"][0]))  # type: ignore[index]
            for pos in positions
        )
        if min_x < self.safe_margin_m:
            warnings.append("source_receiver_left_margin_too_small")
        if self.domain_m[0] - max_x < self.safe_margin_m:
            warnings.append("source_receiver_right_margin_too_small")
        for target in targets or []:
            for x, y in _target_boundary_points(target):
                if x < self.safe_margin_m or self.domain_m[0] - x < self.safe_margin_m:
                    warnings.append("target_horizontal_margin_too_small")
                if y < self.safe_margin_m:
                    warnings.append("target_bottom_margin_too_small")
                if y > self.ground_top_y_m:
                    warnings.append("target_above_ground_surface")
        return sorted(set(warnings))


AIRBORNE_GEOMETRY = AirborneGeometry()


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
    scenario_family: str = "legacy_surface_coupled"
    geometry_model: str = "legacy_surface_coupled_2d_tmz_v1"
    antenna_height_m: float | None = None
    air_layer_thickness_m: float | None = None
    pml_margin_cells: int = AIRBORNE_PML_MARGIN_CELLS
    is_uav_gpr_evidence: bool = False
    uses_per_trace_inputs: bool = False
    uses_scripted_inputs: bool = False
    height_variation: bool = False
    geometry_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GprMaxRunResult:
    """Paths and metadata from one gprMax run."""

    scenario_dir: Path
    model_input: Path
    model_inputs: list[Path]
    command: list[str]
    commands: list[list[str]]
    out_files: list[Path]
    stdout_path: Path
    stderr_path: Path
    returncode: int
    synthetic_sidecars: dict[str, Path] = field(default_factory=dict)


def build_scenario_definitions(
    scenario_family: str = DEFAULT_SCENARIO_FAMILY,
) -> dict[str, ScenarioDefinition]:
    """Return gprMax scenes by family.

    The default family is the air-launched UAV-GPR geometry. The old surface
    coupled scenes remain available for regression smoke checks only.
    """
    family = str(scenario_family or DEFAULT_SCENARIO_FAMILY).lower()
    airborne = _airborne_scenario_definitions()
    legacy = _legacy_surface_coupled_scenario_definitions()
    if family == "airborne":
        return airborne
    if family == "legacy":
        return legacy
    if family == "all":
        return {**airborne, **legacy}
    raise ValueError(f"Unsupported scenario family: {scenario_family}")


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
        str(Path(model_input).resolve()),
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


def _which(executable: str, env: dict[str, str] | None = None) -> str | None:
    """Resolve an executable using an optional subprocess environment."""
    if env is None:
        return shutil.which(executable)
    path_values: list[str] = []
    for key in ("Path", "PATH", "path"):
        value = env.get(key)
        if value:
            path_values.append(value)
    for key, value in env.items():
        if key.lower() == "path" and value and value not in path_values:
            path_values.append(value)
    for path_value in path_values:
        found = shutil.which(executable, path=path_value)
        if found:
            return found
    return None


def find_windows_vcvars64(explicit_path: str | None = None) -> Path | None:
    """Find Visual Studio's vcvars64.bat for CUDA host compilation on Windows."""
    candidates: list[Path] = []
    for raw in [
        explicit_path,
        os.environ.get("MYGPR_CUDA_VCVARS64"),
        os.environ.get("GPRMAX_CUDA_VCVARS64"),
    ]:
        if raw:
            candidates.append(Path(raw))

    if os.name == "nt":
        vswhere = (
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Microsoft Visual Studio"
            / "Installer"
            / "vswhere.exe"
        )
        if vswhere.exists():
            try:
                completed = subprocess.run(
                    [
                        str(vswhere),
                        "-latest",
                        "-products",
                        "*",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-find",
                        r"VC\Auxiliary\Build\vcvars64.bat",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                )
                candidates.extend(
                    Path(line.strip())
                    for line in completed.stdout.splitlines()
                    if line.strip()
                )
            except Exception:
                pass

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_vcvars_env(vcvars_path: Path) -> dict[str, str]:
    """Return the environment produced by calling vcvars64.bat."""
    wrapper_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".cmd",
            delete=False,
            encoding="mbcs",
        ) as wrapper:
            wrapper_path = Path(wrapper.name)
            wrapper.write("@echo off\n")
            wrapper.write(f'call "{vcvars_path}" >nul\n')
            wrapper.write("if errorlevel 1 exit /b %errorlevel%\n")
            wrapper.write("set\n")
        completed = subprocess.run(
            ["cmd", "/d", "/c", str(wrapper_path)],
            capture_output=True,
            text=True,
            encoding="mbcs",
            errors="replace",
            timeout=60,
        )
    finally:
        if wrapper_path is not None:
            try:
                wrapper_path.unlink(missing_ok=True)
            except OSError:
                pass
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to initialize Visual Studio build environment: "
            f"{completed.stderr[-1000:]}"
        )
    env = dict(os.environ)
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            env[key] = value
    return env


def resolve_gprmax_runtime_env(
    *,
    gpu: list[str] | None = None,
    cuda_vcvars: str | None = None,
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """Resolve optional subprocess environment required by gprMax GPU runs."""
    info: dict[str, Any] = {
        "cuda_vcvars64_path": None,
        "cuda_vcvars_loaded": False,
        "cl_available": bool(_which("cl.exe")),
        "cl_path": _which("cl.exe"),
    }
    if not gpu or os.name != "nt":
        return None, info
    if _which("cl.exe"):
        return None, info

    vcvars = find_windows_vcvars64(cuda_vcvars)
    if vcvars is None:
        info["cuda_vcvars_error"] = "vcvars64.bat not found"
        return None, info

    env = _load_vcvars_env(vcvars)
    info.update(
        {
            "cuda_vcvars64_path": str(vcvars),
            "cuda_vcvars_loaded": True,
            "cl_available": bool(_which("cl.exe", env=env)),
            "cl_path": _which("cl.exe", env=env),
        }
    )
    return env, info


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


def probe_acceleration_support(
    python_exe: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check whether the gprMax Python environment has MPI/GPU packages."""
    nvcc_path = _which("nvcc", env=env)
    cl_path = _which("cl.exe", env=env)
    probe = (
        "import importlib.util,json;"
        "print(json.dumps({"
        "'mpi4py': importlib.util.find_spec('mpi4py') is not None,"
        "'pycuda': importlib.util.find_spec('pycuda') is not None,"
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
            env=env,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - defensive environment probe
        return {
            "mpi4py": False,
            "pycuda": False,
            "cupy": False,
            "nvcc": bool(nvcc_path),
            "nvcc_path": nvcc_path,
            "cl": bool(cl_path),
            "cl_path": cl_path,
            "probe_error": str(exc),
        }
    if completed.returncode != 0:
        return {
            "mpi4py": False,
            "pycuda": False,
            "cupy": False,
            "nvcc": bool(nvcc_path),
            "nvcc_path": nvcc_path,
            "cl": bool(cl_path),
            "cl_path": cl_path,
            "probe_error": completed.stderr[-1000:],
        }
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        result["nvcc"] = bool(nvcc_path)
        result["nvcc_path"] = nvcc_path
        result["cl"] = bool(cl_path)
        result["cl_path"] = cl_path
        return result
    except (IndexError, json.JSONDecodeError):
        return {
            "mpi4py": False,
            "pycuda": False,
            "cupy": False,
            "nvcc": bool(nvcc_path),
            "nvcc_path": nvcc_path,
            "cl": bool(cl_path),
            "cl_path": cl_path,
            "probe_error": completed.stdout[-1000:],
        }


def find_out_files(run_dir: Path, scenario_id: str) -> list[Path]:
    """Find gprMax output files using numeric suffix sorting."""
    found: list[tuple[int, Path]] = []
    for path in Path(run_dir).glob(f"{scenario_id}*.out"):
        suffix = path.stem[len(scenario_id) :]
        if suffix.isdigit():
            number = int(suffix)
        else:
            matches = re.findall(r"\d+", suffix)
            number = int(matches[-1]) if matches else 0
        found.append((number, path))
    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def run_multi_scenario_report(
    *,
    gprmax_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    scenario_ids: list[str] | None = None,
    scenario_family: str = DEFAULT_SCENARIO_FAMILY,
    runs: int = DEFAULT_RUNS,
    geometry_fixed: bool = True,
    mpi: int | None = None,
    gpu: list[str] | None = None,
    python_override: str | None = None,
    search_mode: str = "fast",
    baseline_profile_key: str = DEFAULT_PROFILE_KEY,
    zero_time_policy: str = "align_auto",
    ascan_samples: int | None = DEFAULT_ASCAN_SAMPLE_COUNT,
    cuda_vcvars: str | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run all requested scenarios and write the HTML report."""
    scenarios = build_scenario_definitions(scenario_family=scenario_family)
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
    runtime_env, runtime_env_info = resolve_gprmax_runtime_env(
        gpu=gpu,
        cuda_vcvars=cuda_vcvars,
    )
    acceleration = probe_acceleration_support(python_exe, env=runtime_env)
    acceleration.update(runtime_env_info)

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
            env=runtime_env,
            extra_args=extra_args,
        )
        package = convert_gprmax_run(
            definition,
            gprmax_run,
            runs=runs,
            ascan_samples=ascan_samples,
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
                "geometry_model": package["scenario"].get("geometry_model"),
                "scenario_family": package["scenario"].get("scenario_family"),
                "is_uav_gpr_evidence": package["scenario"].get("is_uav_gpr_evidence"),
                "ground_truth": package["ground_truth"],
                "gprmax": {
                    "command": gprmax_run.command,
                    "commands": gprmax_run.commands,
                    "command_count": len(gprmax_run.commands),
                    "stdout": _relpath(gprmax_run.stdout_path, report_dir),
                    "stderr": _relpath(gprmax_run.stderr_path, report_dir),
                    "returncode": int(gprmax_run.returncode),
                    "model_inputs": [_relpath(path, report_dir) for path in gprmax_run.model_inputs],
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
        "scenario_family": scenario_family,
        "search_mode": search_mode,
        "zero_time_policy": zero_time_policy,
        "gprmax_root": str(gprmax_root),
        "python_executable": str(python_exe),
        "run_settings": {
            "runs": int(runs),
            "ascan_samples": int(ascan_samples) if ascan_samples else None,
            "recommended_ascan_samples": list(RECOMMENDED_ASCAN_SAMPLE_COUNTS),
            "geometry_fixed": bool(geometry_fixed),
            "mpi": int(mpi) if mpi is not None else None,
            "gpu": list(gpu or []),
            "cuda_vcvars": str(cuda_vcvars) if cuda_vcvars else None,
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
    env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> GprMaxRunResult:
    """Write the model file and run gprMax for one scene."""
    model_inputs = write_gprmax_inputs(definition, scenario_dir, runs=runs)
    sidecars = write_synthetic_sidecars(definition, scenario_dir, runs=runs)
    commands: list[list[str]] = []
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    returncode = 0

    if definition.uses_per_trace_inputs:
        for model_input in model_inputs:
            command = build_gprmax_command(
                python_exe,
                model_input,
                runs=1,
                geometry_fixed=geometry_fixed,
                mpi=mpi,
                gpu=gpu,
                extra_args=extra_args,
            )
            commands.append(command)
            completed = subprocess.run(
                command,
                cwd=str(gprmax_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            stdout_chunks.append(f"$ {' '.join(command)}\n{completed.stdout}")
            stderr_chunks.append(f"$ {' '.join(command)}\n{completed.stderr}")
            returncode = int(completed.returncode)
            if completed.returncode != 0:
                break
    else:
        command = build_gprmax_command(
            python_exe,
            model_inputs[0],
            runs=runs,
            geometry_fixed=geometry_fixed,
            mpi=mpi,
            gpu=gpu,
            extra_args=extra_args,
        )
        commands.append(command)
        completed = subprocess.run(
            command,
            cwd=str(gprmax_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        stdout_chunks.append(completed.stdout)
        stderr_chunks.append(completed.stderr)
        returncode = int(completed.returncode)

    stdout_path = scenario_dir / "gprmax_stdout.txt"
    stderr_path = scenario_dir / "gprmax_stderr.txt"
    stdout_path.write_text("\n\n".join(stdout_chunks), encoding="utf-8")
    stderr_path.write_text("\n\n".join(stderr_chunks), encoding="utf-8")
    out_files = find_out_files(scenario_dir, definition.scenario_id)
    if returncode != 0:
        raise RuntimeError(
            f"gprMax failed for {definition.scenario_id}; see {stderr_path}"
        )
    if not out_files:
        raise RuntimeError(f"gprMax produced no .out files for {definition.scenario_id}")
    return GprMaxRunResult(
        scenario_dir=scenario_dir,
        model_input=model_inputs[0],
        model_inputs=model_inputs,
        command=commands[0],
        commands=commands,
        out_files=out_files,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=int(returncode),
        synthetic_sidecars=sidecars,
    )


def write_gprmax_inputs(
    definition: ScenarioDefinition,
    scenario_dir: Path,
    *,
    runs: int,
) -> list[Path]:
    """Write one fixed-height input or per-trace inputs for variable-height scenes."""
    scenario_dir.mkdir(parents=True, exist_ok=True)
    geometry = _airborne_geometry_for_definition(definition)
    if definition.uses_scripted_inputs:
        model_input = scenario_dir / f"{definition.scenario_id}.in"
        model_input.write_text(
            _airborne_scripted_height_model_text(definition),
            encoding="utf-8",
        )
        return [model_input]
    if not definition.uses_per_trace_inputs:
        model_input = scenario_dir / f"{definition.scenario_id}.in"
        model_input.write_text(definition.model_in_text, encoding="utf-8")
        return [model_input]

    positions = geometry.trace_positions(
        runs,
        variable_height=definition.height_variation,
    )
    model_inputs: list[Path] = []
    for trace_idx, position in enumerate(positions):
        model_input = scenario_dir / f"{definition.scenario_id}_trace{trace_idx:03d}.in"
        text = _airborne_model_text(
            f"{definition.scenario_id}_trace{trace_idx:03d}",
            materials=_model_material_lines(definition),
            bodies=_model_body_lines(definition),
            use_steps=False,
            geometry=geometry,
            source_m=position["source"],  # type: ignore[arg-type]
            receiver_m=position["receiver"],  # type: ignore[arg-type]
        )
        model_input.write_text(text, encoding="utf-8")
        model_inputs.append(model_input)
    return model_inputs


def write_synthetic_sidecars(
    definition: ScenarioDefinition,
    scenario_dir: Path,
    *,
    runs: int,
) -> dict[str, Path]:
    """Write synthetic RTK/IMU/altimeter sidecars for airborne scenarios."""
    if definition.scenario_family != "airborne":
        return {}
    metadata = build_synthetic_trace_metadata(definition, trace_count=runs)
    timestamps = np.asarray(metadata["trace_timestamp_s"], dtype=np.float64)
    sidecars = {
        "trace_timestamps": scenario_dir / "trace_timestamps.csv",
        "rtk": scenario_dir / "rtk.csv",
        "imu": scenario_dir / "imu.csv",
        "altimeter": scenario_dir / "altimeter.csv",
    }
    _write_csv_rows(
        sidecars["trace_timestamps"],
        ["trace_idx", "timestamp_s"],
        [
            [idx, f"{timestamps[idx]:.6f}"]
            for idx in range(timestamps.size)
        ],
    )
    _write_csv_rows(
        sidecars["rtk"],
        [
            "timestamp_s",
            "local_x_m",
            "local_y_m",
            "local_z_m",
            "latitude_deg",
            "longitude_deg",
            "altitude_m",
        ],
        [
            [
                f"{timestamps[idx]:.6f}",
                f"{float(metadata['local_x_m'][idx]):.6f}",
                f"{float(metadata['local_y_m'][idx]):.6f}",
                f"{float(metadata['local_z_m'][idx]):.6f}",
                f"{float(metadata['latitude_deg'][idx]):.9f}",
                f"{float(metadata['longitude_deg'][idx]):.9f}",
                f"{float(metadata['altitude_m'][idx]):.6f}",
            ]
            for idx in range(timestamps.size)
        ],
    )
    _write_csv_rows(
        sidecars["imu"],
        ["timestamp_s", "roll_deg", "pitch_deg", "yaw_deg"],
        [
            [
                f"{timestamps[idx]:.6f}",
                f"{float(metadata['roll_deg'][idx]):.6f}",
                f"{float(metadata['pitch_deg'][idx]):.6f}",
                f"{float(metadata['yaw_deg'][idx]):.6f}",
            ]
            for idx in range(timestamps.size)
        ],
    )
    _write_csv_rows(
        sidecars["altimeter"],
        ["timestamp_s", "flight_height_m"],
        [
            [
                f"{timestamps[idx]:.6f}",
                f"{float(metadata['flight_height_m'][idx]):.6f}",
            ]
            for idx in range(timestamps.size)
        ],
    )
    return sidecars


def _airborne_geometry_for_definition(definition: ScenarioDefinition) -> AirborneGeometry:
    """Resolve the per-scenario airborne geometry used for inputs and metadata."""
    if definition.scenario_family != "airborne":
        return AIRBORNE_GEOMETRY
    if definition.antenna_height_m is None:
        return AIRBORNE_GEOMETRY
    if abs(float(definition.antenna_height_m) - float(AIRBORNE_GEOMETRY.antenna_height_m)) <= 1.0e-12:
        return AIRBORNE_GEOMETRY
    return replace(AIRBORNE_GEOMETRY, antenna_height_m=float(definition.antenna_height_m))


def build_synthetic_trace_metadata(
    definition: ScenarioDefinition,
    *,
    trace_count: int,
) -> dict[str, np.ndarray]:
    """Return deterministic airborne sidecar-like metadata for each trace."""
    if definition.scenario_family != "airborne":
        return {}
    geometry = _airborne_geometry_for_definition(definition)
    positions = geometry.trace_positions(
        trace_count,
        variable_height=definition.height_variation,
    )
    timestamps = np.arange(trace_count, dtype=np.float64) * 0.10
    source_x = np.asarray([float(pos["source"][0]) for pos in positions], dtype=np.float32)  # type: ignore[index]
    receiver_x = np.asarray([float(pos["receiver"][0]) for pos in positions], dtype=np.float32)  # type: ignore[index]
    height = np.asarray([float(pos["height_m"]) for pos in positions], dtype=np.float32)
    midpoint_x = (source_x + receiver_x) / 2.0
    latitude0 = 30.000000
    longitude0 = 104.000000
    return {
        "trace_timestamp_s": timestamps.astype(np.float64),
        "local_x_m": midpoint_x.astype(np.float32),
        "local_y_m": np.zeros(trace_count, dtype=np.float32),
        "local_z_m": (geometry.ground_top_y_m + height).astype(np.float32),
        "flight_height_m": height.astype(np.float32),
        "altitude_m": (100.0 + height).astype(np.float32),
        "latitude_deg": (latitude0 + midpoint_x * 1.0e-5).astype(np.float64),
        "longitude_deg": (longitude0 + midpoint_x * 1.0e-5).astype(np.float64),
        "roll_deg": (
            1.2 * np.sin(np.linspace(0.0, 2.0 * np.pi, trace_count))
        ).astype(np.float32),
        "pitch_deg": (
            0.8 * np.cos(np.linspace(0.0, 2.0 * np.pi, trace_count))
        ).astype(np.float32),
        "yaw_deg": np.zeros(trace_count, dtype=np.float32),
    }


def _write_csv_rows(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    lines = [",".join(headers)]
    lines.extend(",".join(str(item) for item in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_ascan_sample_count(ascan_samples: int | None) -> int | None:
    """Return a valid target A-scan sample count, or None to keep raw gprMax output."""
    if ascan_samples is None:
        return None
    value = int(ascan_samples)
    if value <= 0:
        return None
    if value < 16:
        raise ValueError("ascan_samples must be >= 16, or <= 0 to keep raw gprMax samples")
    return value


def _resample_bscan_samples(data: np.ndarray, target_samples: int | None) -> np.ndarray:
    """Resample B-scan rows to match real UAV-GPR A-scan sample counts."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("gprMax B-scan data must be a 2D array")
    if target_samples is None or int(target_samples) == int(arr.shape[0]):
        return arr.astype(np.float32, copy=True)
    if arr.shape[0] <= 1:
        return np.repeat(arr.astype(np.float32, copy=True), int(target_samples), axis=0)

    source_axis = np.linspace(0.0, 1.0, int(arr.shape[0]), dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, int(target_samples), dtype=np.float64)
    return resample_bscan_columns_linear(
        arr.T,
        source_axis,
        target_axis,
    ).T.astype(np.float32, copy=False)


def convert_gprmax_run(
    definition: ScenarioDefinition,
    run_result: GprMaxRunResult,
    *,
    runs: int,
    ascan_samples: int | None = DEFAULT_ASCAN_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Convert gprMax `.out` files into MyGPR package artifacts."""
    load_result = read_gprmax_out(str(run_result.out_files[0]))
    raw_bscan = np.asarray(load_result["data"], dtype=np.float32)
    time_step_s = load_result.get("time_step_s")
    total_time_ns = load_result.get("total_time_ns")
    if total_time_ns is None and time_step_s is not None:
        total_time_ns = float(time_step_s) * int(raw_bscan.shape[0]) * 1e9
    if total_time_ns is None:
        total_time_ns = float(definition.total_time_ns)
    target_ascan_samples = _normalize_ascan_sample_count(ascan_samples)
    bscan = _resample_bscan_samples(raw_bscan, target_ascan_samples)
    output_time_step_s = (
        float(total_time_ns) * 1.0e-9 / max(int(bscan.shape[0]) - 1, 1)
        if int(bscan.shape[0]) > 1
        else float(time_step_s or 0.0)
    )

    simulation = {
        "sample_count": int(bscan.shape[0]),
        "raw_sample_count": int(raw_bscan.shape[0]),
        "target_ascan_sample_count": (
            int(target_ascan_samples) if target_ascan_samples is not None else None
        ),
        "resampled_from_raw": bool(int(bscan.shape[0]) != int(raw_bscan.shape[0])),
        "trace_count": int(bscan.shape[1]),
        "requested_runs": int(runs),
        "time_step_s": float(output_time_step_s) if output_time_step_s > 0.0 else None,
        "raw_time_step_s": float(time_step_s) if time_step_s is not None else None,
        "total_time_ns": float(total_time_ns),
        "trace_step_m": float(definition.trace_step_m),
        "source_receiver_offset_m": float(
            definition.receiver_start_m[0] - definition.source_start_m[0]
        ),
    }
    trace_metadata = build_synthetic_trace_metadata(
        definition,
        trace_count=int(bscan.shape[1]),
    )
    ground_truth = build_ground_truth(definition, simulation)
    scenario = {
        "schema": "mygpr_gprmax_multiscenario_scenario_v1",
        "scenario_id": definition.scenario_id,
        "label": definition.label,
        "description": definition.description,
        "scenario_family": definition.scenario_family,
        "geometry_model": definition.geometry_model,
        "is_uav_gpr_evidence": bool(definition.is_uav_gpr_evidence),
        "source": {
            "kind": "gprmax_out",
            "model_input": str(run_result.model_input),
            "model_inputs": [str(path) for path in run_result.model_inputs],
            "first_out": str(run_result.out_files[0]),
        },
        "simulation": simulation,
        "domain_m": list(definition.domain_m),
        "dx_dy_dz_m": [definition.dx_m, definition.dx_m, definition.dx_m],
        "ground_top_y_m": float(definition.ground_top_y_m),
        "air_layer_thickness_m": (
            float(definition.air_layer_thickness_m)
            if definition.air_layer_thickness_m is not None
            else None
        ),
        "antenna_height_m": (
            float(definition.antenna_height_m)
            if definition.antenna_height_m is not None
            else None
        ),
        "antenna_height_profile_m": _json_safe(
            trace_metadata.get("flight_height_m", np.asarray([], dtype=np.float32))
        ),
        "pml_margin_cells": int(definition.pml_margin_cells),
        "geometry_warnings": list(definition.geometry_warnings),
        "synthetic_sidecars": {
            key: str(path) for key, path in run_result.synthetic_sidecars.items()
        },
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
            "raw_a_scan_length": int(raw_bscan.shape[0]),
            "target_ascan_sample_count": (
                int(target_ascan_samples) if target_ascan_samples is not None else None
            ),
            "resampled_from_raw": bool(int(bscan.shape[0]) != int(raw_bscan.shape[0])),
            "num_traces": int(bscan.shape[1]),
            "total_time_ns": float(total_time_ns),
            "trace_interval_m": float(definition.trace_step_m),
            "track_length_m": float(definition.trace_step_m) * max(int(bscan.shape[1]) - 1, 1),
        },
        "trace_metadata": trace_metadata,
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

    wavefield_rois = _build_wavefield_rois(
        definition,
        simulation,
        target_rois=rois,
    )
    analysis_roi = _union_rois(rois, samples, traces) if rois else _default_background_roi(
        wavefield_rois,
        samples,
        traces,
    )
    return {
        "schema": "mygpr_gprmax_multiscenario_ground_truth_v1",
        "scenario_id": definition.scenario_id,
        "geometry_model": definition.geometry_model,
        "structure_notes": list(definition.structure_notes),
        "targets": targets,
        "wavefield_rois": wavefield_rois,
        "analysis_roi": analysis_roi,
        "known_background": {
            "air_ground_interface_y_m": float(definition.ground_top_y_m),
            "layers": definition.layers,
        },
        "risk_checks": [
            "direct_air_wave_misread_as_target",
            "air_ground_reflection_over_enhanced",
            "zero_time_breaks_direct_surface_target_order",
        ],
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
    """Run report comparison through the shared pipeline auto-tune backend."""
    if zero_time_policy not in {"align_auto", "manual", "skip"}:
        raise ValueError(f"Unsupported zero_time_policy: {zero_time_policy}")
    arr = np.asarray(data, dtype=np.float32)
    pipeline = _resolve_pipeline(baseline_profile_key)
    manual_params_by_method = _resolve_manual_params(pipeline, baseline_profile_key)
    manual_original_params_by_method = {
        method_key: dict(params)
        for method_key, params in manual_params_by_method.items()
    }
    roi_spec = {
        "mode": "manual",
        "bounds": ground_truth.get("analysis_roi") or _full_roi(*arr.shape),
        "label": f"{ground_truth.get('scenario_id', 'scenario')} ground-truth ROI",
    }

    locked_params_by_method: dict[str, dict[str, Any]] = {}
    locked_tune_summaries: dict[str, dict[str, Any]] = {}
    policy_notes_by_method: dict[str, list[str]] = {}

    if zero_time_policy == "skip" and "set_zero_time" in pipeline:
        pipeline = [method_key for method_key in pipeline if method_key != "set_zero_time"]
        manual_params_by_method.pop("set_zero_time", None)
        policy_notes_by_method["set_zero_time"] = [
            "本报告按 zero_time_policy=skip 跳过零时矫正。"
        ]
    elif zero_time_policy == "align_auto" and "set_zero_time" in pipeline:
        zero_params, tune_summary, policy_notes = _derive_aligned_zero_time_params(
            arr,
            header_info=header_info,
            trace_metadata=trace_metadata,
            roi_spec=roi_spec,
            search_mode=search_mode,
            base_params=manual_params_by_method.get("set_zero_time", {}),
        )
        manual_params_by_method["set_zero_time"] = dict(zero_params)
        locked_params_by_method["set_zero_time"] = dict(zero_params)
        locked_tune_summaries["set_zero_time"] = tune_summary
        policy_notes_by_method["set_zero_time"] = policy_notes

    pipeline_result = run_auto_tune_pipeline(
        arr,
        header_info=header_info,
        trace_metadata=trace_metadata,
        pipeline=pipeline,
        manual_params_by_method=manual_params_by_method,
        locked_params_by_method=locked_params_by_method,
        baseline_profile_key=baseline_profile_key,
        roi_spec=roi_spec,
        ground_truth=ground_truth,
        search_mode=search_mode,
        rollback_on_reject=True,
    )
    return _pipeline_run_to_report_dict(
        pipeline_result,
        zero_time_policy=zero_time_policy,
        manual_original_params_by_method=manual_original_params_by_method,
        policy_notes_by_method=policy_notes_by_method,
        locked_tune_summaries=locked_tune_summaries,
        ground_truth=ground_truth,
    )


def _derive_aligned_zero_time_params(
    data: np.ndarray,
    *,
    header_info: dict[str, Any] | None,
    trace_metadata: dict[str, np.ndarray] | None,
    roi_spec: dict[str, Any],
    search_mode: str,
    base_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Auto-tune zero-time once, then lock both report branches to it."""
    try:
        tune_result = auto_tune_method(
            data,
            "set_zero_time",
            header_info=header_info,
            trace_metadata=trace_metadata,
            base_params=base_params,
            roi_spec=roi_spec,
            search_mode=search_mode,
        )
        recommended = dict(
            tune_result.get("recommended_params")
            or tune_result.get("best_params")
            or {}
        )
        if "new_zero_time" not in recommended:
            recommended = {"new_zero_time": 0.0}
        return recommended, _compact_auto_tune_result(tune_result), [ZERO_TIME_ALIGN_NOTE]
    except Exception as exc:
        return (
            {"new_zero_time": 0.0},
            {"error": str(exc), "recommended_params": {"new_zero_time": 0.0}},
            [
                "自动零时选参失败，本报告将两分支零时参数设为 0.0ns，"
                "避免经验 5.0ns 切掉有效结构。"
            ],
        )


def _pipeline_run_to_report_dict(
    result: AutoTunePipelineRun,
    *,
    zero_time_policy: str,
    manual_original_params_by_method: dict[str, dict[str, Any]],
    policy_notes_by_method: dict[str, list[str]],
    locked_tune_summaries: dict[str, dict[str, Any]],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Adapt the shared backend result to the existing report schema."""
    steps = [
        _pipeline_step_to_report_dict(
            step,
            manual_original_params_by_method=manual_original_params_by_method,
            policy_notes=policy_notes_by_method.get(step.method_key, []),
            locked_tune_summary=locked_tune_summaries.get(step.method_key, {}),
            ground_truth=ground_truth,
        )
        for step in result.steps
    ]
    metrics = _metric_summary_with_compat(
        result.manual.metrics,
        result.automatic.metrics,
        result.metric_delta,
    )
    auto_tune_results = dict(result.automatic.auto_tune_results)
    for method_key, summary in locked_tune_summaries.items():
        auto_tune_results.setdefault(method_key, summary)
    final_manual_roi = (
        dict(result.steps[-1].manual_roi_after)
        if result.steps
        else dict(result.roi_info.get("bounds", {}))
    )
    final_auto_roi = (
        dict(result.steps[-1].auto_roi_after)
        if result.steps
        else dict(result.roi_info.get("bounds", {}))
    )
    return {
        "backend": "core.auto_tune_pipeline",
        "pipeline": list(result.pipeline),
        "manual_params_by_method": result.manual.params_by_method,
        "auto_params_by_method": result.automatic.params_by_method,
        "auto_tune_results": auto_tune_results,
        "manual_final": result.manual.result,
        "auto_final": result.automatic.result,
        "steps": steps,
        "roi_spec": result.roi_info,
        "final_manual_roi": final_manual_roi,
        "final_auto_roi": final_auto_roi,
        "zero_time_policy": zero_time_policy,
        "metrics": metrics,
        "verdict": _pipeline_verdict(result.overall_recommendation, metrics),
        "overall_recommendation": result.overall_recommendation,
        "risk_flags": list(result.risk_flags),
    }


def _pipeline_step_to_report_dict(
    step: PipelineStepRecord,
    *,
    manual_original_params_by_method: dict[str, dict[str, Any]],
    policy_notes: list[str],
    locked_tune_summary: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    metrics = _metric_summary_with_compat(
        step.manual_metrics,
        step.auto_metrics,
        step.metric_delta,
    )
    notes = list(policy_notes)
    if step.rolled_back_to_manual:
        notes.append("本步自动候选触发 keep_manual，后续流程回退到人工结果。")
    analysis = _build_step_analysis(
        method_key=step.method_key,
        method_name=step.method_name,
        metrics=metrics,
        ground_truth=ground_truth,
        policy_notes=notes,
        recommendation=step.recommendation,
        risk_flags=step.risk_flags,
    )
    tune_summary = dict(step.auto_tune_result or locked_tune_summary or {})
    return {
        "method_key": step.method_key,
        "method_name": step.method_name,
        "manual_input": step.manual_before,
        "auto_input": step.auto_before,
        "manual_output": step.manual_after,
        "auto_output": step.auto_after,
        "manual_params": dict(step.manual_params),
        "manual_original_params": dict(
            manual_original_params_by_method.get(step.method_key, step.manual_params)
        ),
        "auto_params": dict(step.auto_params),
        "auto_tune_summary": tune_summary,
        "manual_warnings": list(step.warnings.get("manual", [])),
        "auto_warnings": list(step.warnings.get("automatic", [])),
        "policy_notes": notes,
        "manual_input_roi": dict(step.manual_roi_before),
        "auto_input_roi": dict(step.auto_roi_before),
        "manual_output_roi": dict(step.manual_roi_after),
        "auto_output_roi": dict(step.auto_roi_after),
        "metrics": metrics,
        "analysis": analysis,
        "recommendation": step.recommendation,
        "risk_flags": list(step.risk_flags),
        "rolled_back_to_manual": bool(step.rolled_back_to_manual),
        "reason": step.reason,
    }


def _metric_summary_with_compat(
    manual_metrics: dict[str, float],
    auto_metrics: dict[str, float],
    metric_delta: dict[str, float],
) -> dict[str, Any]:
    manual = _metrics_with_comparison_alias(manual_metrics)
    auto = _metrics_with_comparison_alias(auto_metrics)
    delta = _metrics_with_comparison_alias(metric_delta)
    return {
        "manual": {key: float(value) for key, value in manual.items()},
        "auto": {key: float(value) for key, value in auto.items()},
        "delta_auto_minus_manual": {
            key: float(value) for key, value in delta.items()
        },
    }


def _metrics_with_comparison_alias(metrics: dict[str, float]) -> dict[str, float]:
    resolved = {key: float(value) for key, value in metrics.items()}
    if "pipeline_score" in resolved and "comparison_score" not in resolved:
        resolved["comparison_score"] = float(resolved["pipeline_score"])
    return resolved


def _pipeline_verdict(
    overall_recommendation: str,
    metric_summary: dict[str, Any],
) -> str:
    if overall_recommendation == "adopt_auto":
        return "auto_better"
    if overall_recommendation == "keep_manual":
        return "manual_better"
    return _comparison_verdict(metric_summary)


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
                "recommendation": step.get("recommendation"),
                "risk_flags": list(step.get("risk_flags", [])),
                "rolled_back_to_manual": bool(step.get("rolled_back_to_manual", False)),
                "reason": step.get("reason", ""),
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
      <p>本报告使用 gprMax 正演数据构造已知真实结构，以逐步骤图像、参数和指标对比验证 MyGPR 自动选参相对经验 baseline 的表现。Airborne 场景显式记录天线离地高度；高度变化场景逐道生成输入文件和 sidecar。</p>
    </section>

    <section>
      <h2>gprMax 运行设置</h2>
      <div class="kv-grid">
        <div><span>gprMax 根目录</span><strong>{_esc(payload.get("gprmax_root"))}</strong></div>
        <div><span>Python</span><strong>{_esc(payload.get("python_executable"))}</strong></div>
        <div><span>每场景 A-scan 道数</span><strong>{_esc(run_settings.get("runs"))}</strong></div>
        <div><span>处理用每道采样点</span><strong>{_esc(run_settings.get("ascan_samples") or "保留 gprMax 原始点数")}</strong></div>
        <div><span>场景族</span><strong>{_esc(payload.get("scenario_family") or "airborne")}</strong></div>
        <div><span>geometry-fixed</span><strong>{_esc(run_settings.get("geometry_fixed"))}</strong></div>
        <div><span>MPI 参数</span><strong>{_esc(run_settings.get("mpi") or "未启用")}</strong></div>
        <div><span>GPU 参数</span><strong>{_esc(", ".join(run_settings.get("gpu") or []) or "未启用")}</strong></div>
        <div><span>mpi4py 可用</span><strong>{_esc(support.get("mpi4py"))}</strong></div>
        <div><span>PyCUDA 可用</span><strong>{_esc(support.get("pycuda"))}</strong></div>
        <div><span>nvcc 可用</span><strong>{_esc(support.get("nvcc"))}</strong></div>
        <div><span>cl.exe 可用</span><strong>{_esc(support.get("cl"))}</strong></div>
        <div><span>vcvars64 自动加载</span><strong>{_esc(support.get("cuda_vcvars_loaded"))}</strong></div>
        <div><span>CuPy 可用</span><strong>{_esc(support.get("cupy"))}</strong></div>
      </div>
      <p class="note">本机环境探测只用于决定是否建议启用 MPI/GPU。当前脚本会暴露 <code>-mpi</code> 和 <code>-gpu</code>，但只有在相应依赖可用且用户显式传参时才加入命令。gprMax 原始 FDTD 输出会保留在 <code>.out</code> 中；报告默认将 BScan 重采样到真实 UAV-GPR 常见的 501/701 点 A-scan 长度后再进入 MyGPR 流程。</p>
    </section>

    <section>
      <h2>参数策略</h2>
      <p>人工选参使用 <code>{_esc(payload.get("baseline_profile_key"))}</code> 的经验 baseline。本标准流程报告使用 SEC/energy-decay 作为默认解释增益；若目标是比较 AGC、SEC、TGC 或无增益，应运行增益方法报告。自动选参以同一组参数为 base，在每个支持 auto-tune 的算法步骤前根据当前 BScan 和 ground-truth ROI 重新评分并选择参数。搜索模式为 <code>{_esc(payload.get("search_mode"))}</code>，零时策略为 <code>{_esc(payload.get("zero_time_policy"))}</code>。</p>
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
    if definition.geometry_model.startswith("airborne"):
        ax.add_patch(
            plt.Rectangle(
                (0, definition.ground_top_y_m),
                domain_x,
                max(0.0, domain_y - definition.ground_top_y_m),
                facecolor="#eef6ff",
                edgecolor="#91a7ff",
                alpha=0.55,
            )
        )
        ax.axhline(
            definition.ground_top_y_m,
            color="#7c5c28",
            linewidth=1.4,
            linestyle="-",
        )
        ax.text(
            domain_x * 0.015,
            definition.ground_top_y_m + 0.012,
            "air-ground surface",
            fontsize=7,
            color="#5c3d0c",
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
    if definition.geometry_model.startswith("airborne"):
        positions = AIRBORNE_GEOMETRY.trace_positions(
            definition.default_runs,
            variable_height=definition.height_variation,
        )
        tx_x = [float(pos["source"][0]) for pos in positions]  # type: ignore[index]
        tx_y = [float(pos["source"][1]) for pos in positions]  # type: ignore[index]
        rx_x = [float(pos["receiver"][0]) for pos in positions]  # type: ignore[index]
        rx_y = [float(pos["receiver"][1]) for pos in positions]  # type: ignore[index]
        ax.plot(tx_x, tx_y, color="#c92a2a", linewidth=1.2, label="Tx flight path")
        ax.plot(rx_x, rx_y, color="#1864ab", linewidth=1.2, label="Rx flight path")
        ax.scatter([tx_x[0]], [tx_y[0]], marker="^", color="#c92a2a")
        ax.scatter([rx_x[0]], [rx_y[0]], marker="v", color="#1864ab")
        if definition.height_variation:
            ax.text(
                domain_x * 0.56,
                max(tx_y) + 0.008,
                "height variation",
                fontsize=7,
                color="#364fc7",
            )
    else:
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


def _airborne_scenario_definitions() -> dict[str, ScenarioDefinition]:
    """Build the default air-launched UAV-GPR validation scene family."""
    scenarios = [
        _airborne_single_cylinder_scenario(),
        _airborne_hyperbola_demo_scenario(),
        _airborne_rough_soil_hyperbola_scenario(),
        _airborne_double_cylinder_scenario(),
        _airborne_layered_interface_scenario(),
        _airborne_air_crack_scenario(),
        _airborne_no_target_background_scenario(),
        _airborne_height_variation_cylinder_scenario(),
    ]
    return {scenario.scenario_id: scenario for scenario in scenarios}


def _legacy_surface_coupled_scenario_definitions() -> dict[str, ScenarioDefinition]:
    """Build old near-ground scenes retained only as regression smoke benchmarks."""
    mapping = {
        "legacy_surface_coupled_single_cylinder_v1": _single_cylinder_scenario(),
        "legacy_surface_coupled_double_cylinder_v1": _double_cylinder_scenario(),
        "legacy_surface_coupled_layered_interface_v1": _layered_interface_scenario(),
        "legacy_surface_coupled_air_crack_v1": _crack_air_filled_scenario(),
        "legacy_surface_coupled_no_target_background_v1": _no_target_background_scenario(),
    }
    result: dict[str, ScenarioDefinition] = {}
    for scenario_id, definition in mapping.items():
        result[scenario_id] = replace(
            definition,
            scenario_id=scenario_id,
            label=f"旧贴地场景 / {definition.label}",
            description=(
                f"{definition.description} 该场景为非 UAV-GPR、source/receiver 位于地表附近的 "
                "legacy toy benchmark，不作为 UAV-GPR 论文级证据。"
            ),
            structure_notes=[
                "Legacy surface-coupled geometry: Tx/Rx near the air-ground interface.",
                "Use only as smoke/regression benchmark, not as UAV-GPR evidence.",
                *definition.structure_notes,
            ],
            model_in_text=definition.model_in_text.replace(
                f"MyGPR {definition.scenario_id}",
                f"MyGPR {scenario_id}",
            ),
            default_runs=36,
            scenario_family="legacy_surface_coupled",
            geometry_model="legacy_surface_coupled_2d_tmz_v1",
            is_uav_gpr_evidence=False,
        )
    return result


def _airborne_single_cylinder_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "dry_silty_sand", "relative_permittivity": 6.0, "conductivity_s_per_m": 0.002},
        {"name": "metal_cylinder", "material": "pec"},
    ]
    targets = [
        {
            "target_id": "airborne_metal_cylinder_01",
            "type": "metal_cylinder",
            "center_m": [0.360, 0.175, 0.0],
            "radius_m": 0.020,
            "relative_permittivity": 6.0,
        }
    ]
    return _airborne_definition(
        scenario_id="airborne_single_cylinder_v1",
        label="UAV-GPR 单金属圆柱",
        description=(
            "离地 Tx/Rx、空气层、地表强反射和单个地下 PEC 圆柱；"
            "用于验证自动选参能否在直达波/地表反射之后保留目标双曲线。"
        ),
        structure_notes=[
            "Tx/Rx 位于空气层，天线离地 0.12 m，收发距 0.08 m。",
            "地表位于 y=0.30 m，地下为相对介电常数 6 的弱导电粉砂土。",
            "目标为 x=0.360 m、y=0.175 m、半径 0.020 m 的金属圆柱。",
        ],
        materials=materials,
        targets=targets,
        layers=[],
        model_materials=["#material: 6 0.002 1 0 dry_silty_sand"],
        bodies=[
            _airborne_ground_box("dry_silty_sand"),
            "#cylinder: 0.360 0.175 0 0.360 0.175 0.002 0.020 pec",
        ],
    )


def _airborne_hyperbola_demo_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "very_dry_sand", "relative_permittivity": 3.0, "conductivity_s_per_m": 0.0002},
        {"name": "metal_cylinder", "material": "pec"},
    ]
    targets = [
        {
            "target_id": "airborne_centered_hyperbola_cylinder",
            "type": "metal_cylinder",
            "center_m": [0.370, 0.185, 0.0],
            "radius_m": 0.010,
            "relative_permittivity": 3.0,
        }
    ]
    return _airborne_definition(
        scenario_id="airborne_hyperbola_demo_v1",
        label="UAV-GPR 完整双曲线演示",
        description=(
            "居中单个小型 PEC 圆柱演示场景，目标是让 96 道标准孔径内能看到较完整的双曲线两翼。"
        ),
        structure_notes=[
            "目标位于 x=0.370 m，接近 96 道 Tx/Rx 中点孔径中心。",
            "目标位于 y=0.185 m，仍位于地表下方且与地表反射保持可分辨时间间隔。",
            "目标采用 1 cm 半径 PEC 圆柱，尽量接近点目标，以增强双曲线边缘可见性。",
            "介质采用极干低损耗砂土，削弱地表反射支配性，便于展示完整双曲线形态。",
        ],
        materials=materials,
        targets=targets,
        layers=[],
        model_materials=["#material: 3 0.0002 1 0 very_dry_sand"],
        bodies=[
            _airborne_ground_box("very_dry_sand"),
            "#cylinder: 0.370 0.185 0 0.370 0.185 0.002 0.010 pec",
        ],
        antenna_height_m=0.080,
    )


def _airborne_rough_soil_hyperbola_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "dry_silty_sand", "relative_permittivity": 6.0, "conductivity_s_per_m": 0.002},
        {"name": "weak_wet_patch", "relative_permittivity": 8.0, "conductivity_s_per_m": 0.006},
        {"name": "weak_air_patch", "relative_permittivity": 1.5, "conductivity_s_per_m": 0.0},
        {"name": "metal_cylinder", "material": "pec"},
    ]
    targets = [
        {
            "target_id": "airborne_rough_soil_metal_cylinder",
            "type": "metal_cylinder",
            "center_m": [0.370, 0.175, 0.0],
            "radius_m": 0.012,
            "relative_permittivity": 6.0,
        }
    ]
    clutter_bodies = [
        "#box: 0.118 0.080 0 0.172 0.126 0.002 weak_wet_patch",
        "#box: 0.498 0.165 0 0.562 0.214 0.002 weak_air_patch",
        "#cylinder: 0.245 0.108 0 0.245 0.108 0.002 0.010 weak_wet_patch",
        "#cylinder: 0.585 0.092 0 0.585 0.092 0.002 0.012 weak_wet_patch",
    ]
    return _airborne_definition(
        scenario_id="airborne_rough_soil_hyperbola_v1",
        label="UAV-GPR 粗糙非均匀土中双曲线",
        description=(
            "参考 gprMax heterogeneous_soil 与官方圆柱 B-scan 思路，构建粗糙地表、弱非均匀背景和中心金属目标。"
        ),
        structure_notes=[
            "地表由 18 个分段 box 形成确定性起伏，避免直接套用官方 z 向 roughness 到本项目 y 向 2D 模型。",
            "地下加入弱介电/弱空腔夹杂体作为背景杂波，但真值目标只标注中心 PEC 圆柱。",
            "该场景用于检验标准处理链在粗糙地表和非均匀背景下是否仍能保留目标双曲线。",
        ],
        materials=materials,
        targets=targets,
        layers=[
            {
                "kind": "segmented_rough_surface",
                "name": "deterministic_rough_air_ground_interface",
                "base_y_m": AIRBORNE_GROUND_TOP_Y_M,
                "amplitude_m": 0.010,
                "segment_count": 18,
                "material": "dry_silty_sand",
            },
            {
                "kind": "background_clutter",
                "name": "weak_dielectric_air_and_wet_patches",
                "notes": "These bodies are nuisance clutter, not target truth.",
            },
        ],
        model_materials=[
            "#material: 6 0.002 1 0 dry_silty_sand",
            "#material: 8 0.006 1 0 weak_wet_patch",
            "#material: 1.5 0 1 0 weak_air_patch",
        ],
        bodies=[
            *_airborne_segmented_ground_boxes("dry_silty_sand"),
            *clutter_bodies,
            "#cylinder: 0.370 0.175 0 0.370 0.175 0.002 0.012 pec",
        ],
    )


def _airborne_double_cylinder_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "dry_silty_sand", "relative_permittivity": 6.0, "conductivity_s_per_m": 0.002},
        {"name": "metal_cylinder", "material": "pec"},
    ]
    targets = [
        {
            "target_id": "airborne_shallow_metal_cylinder",
            "type": "metal_cylinder",
            "center_m": [0.305, 0.205, 0.0],
            "radius_m": 0.016,
            "relative_permittivity": 6.0,
        },
        {
            "target_id": "airborne_deep_metal_cylinder",
            "type": "metal_cylinder",
            "center_m": [0.455, 0.135, 0.0],
            "radius_m": 0.014,
            "relative_permittivity": 6.0,
        },
    ]
    return _airborne_definition(
        scenario_id="airborne_double_cylinder_v1",
        label="UAV-GPR 双深度圆柱",
        description="两个不同深度的 PEC 圆柱，考察增益与去噪是否同时保护浅层强目标和深层弱目标。",
        structure_notes=[
            "浅层目标位于 x=0.305 m、y=0.205 m；深层目标位于 x=0.455 m、y=0.135 m。",
            "两目标均满足距 PML 安全边界大于 0.06 m。",
            "目标回波应出现在直达波和地表反射之后。",
        ],
        materials=materials,
        targets=targets,
        layers=[],
        model_materials=["#material: 6 0.002 1 0 dry_silty_sand"],
        bodies=[
            _airborne_ground_box("dry_silty_sand"),
            "#cylinder: 0.305 0.205 0 0.305 0.205 0.002 0.016 pec",
            "#cylinder: 0.455 0.135 0 0.455 0.135 0.002 0.014 pec",
        ],
    )


def _airborne_layered_interface_scenario() -> ScenarioDefinition:
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
            "y1_m": 0.145,
            "color": "#9bbf8f",
        }
    ]
    targets = [
        {
            "target_id": "airborne_dry_wet_interface",
            "type": "layer_interface",
            "interface_y_m": 0.145,
            "relative_permittivity": 4.5,
        }
    ]
    return _airborne_definition(
        scenario_id="airborne_layered_interface_v1",
        label="UAV-GPR 干湿分层界面",
        description="空气发射下的水平干湿界面，验证背景抑制和增益不会把真实连续层状反射抹掉。",
        structure_notes=[
            "地表以下上层为低电导干砂，下层为较高介电常数湿砂。",
            "真实结构为 y=0.145 m 的连续水平界面，而不是点状双曲线。",
            "地表强反射和层状界面反射在时序上必须可区分。",
        ],
        materials=materials,
        targets=targets,
        layers=layers,
        model_materials=[
            "#material: 4.5 0.001 1 0 dry_sand",
            "#material: 10 0.012 1 0 wet_sand",
        ],
        bodies=[
            _airborne_ground_box("dry_sand"),
            "#box: 0 0 0 0.720 0.145 0.002 wet_sand",
        ],
    )


def _airborne_air_crack_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "dry_silty_sand", "relative_permittivity": 6.0, "conductivity_s_per_m": 0.002},
        {"name": "air_crack", "relative_permittivity": 1.0, "conductivity_s_per_m": 0.0},
    ]
    targets = [
        {
            "target_id": "airborne_air_crack_01",
            "type": "air_crack",
            "x0_m": 0.392,
            "x1_m": 0.408,
            "y0_m": 0.115,
            "y1_m": 0.235,
            "relative_permittivity": 6.0,
        }
    ]
    return _airborne_definition(
        scenario_id="airborne_air_crack_v1",
        label="UAV-GPR 空气裂缝弱结构",
        description="窄空气裂缝弱反射场景，验证自动选参不会把窄线状反射和边缘绕射过度平滑。",
        structure_notes=[
            "裂缝为 x=0.392-0.408 m、y=0.115-0.235 m 的低介电常数空腔。",
            "该场景对去噪和背景抑制更敏感，过强参数会损伤边缘绕射。",
            "AGC 类增益可能让裂缝更醒目，但也更容易放大空气波后的背景。",
        ],
        materials=materials,
        targets=targets,
        layers=[],
        model_materials=[
            "#material: 6 0.002 1 0 dry_silty_sand",
            "#material: 1 0 1 0 air_crack",
        ],
        bodies=[
            _airborne_ground_box("dry_silty_sand"),
            "#box: 0.392 0.115 0 0.408 0.235 0.002 air_crack",
        ],
    )


def _airborne_no_target_background_scenario() -> ScenarioDefinition:
    materials = [
        {"name": "dry_silty_sand", "relative_permittivity": 6.0, "conductivity_s_per_m": 0.002},
    ]
    return _airborne_definition(
        scenario_id="airborne_no_target_background_v1",
        label="UAV-GPR 无目标背景",
        description="只有空气层、地表和均匀地下介质，用于验证自动选参与增益不会制造假异常。",
        structure_notes=[
            "不存在地下异常体，真值只包含直达波、地表反射、背景区和晚时窗噪声区。",
            "理想处理应控制地表反射后的背景和深部噪声，不能凭空产生局部强异常。",
        ],
        materials=materials,
        targets=[],
        layers=[],
        model_materials=["#material: 6 0.002 1 0 dry_silty_sand"],
        bodies=[_airborne_ground_box("dry_silty_sand")],
    )


def _airborne_height_variation_cylinder_scenario() -> ScenarioDefinition:
    base = _airborne_single_cylinder_scenario()
    return replace(
        base,
        scenario_id="airborne_height_variation_cylinder_v1",
        label="UAV-GPR 高度变化圆柱",
        description=(
            "逐道改变天线离地高度的单圆柱场景，用于运动补偿和高度 sidecar 链路验证；"
            "该场景不使用 #src_steps/#rx_steps。"
        ),
        structure_notes=[
            "天线高度曲线为 h_i=0.12+0.035*sin(2*pi*i/(runs-1)) m。",
            "使用 gprMax 官方 Python scripting 和 current_model_run 在单个 .in 内逐道定义 Tx/Rx 高度，配套生成 RTK/IMU/高度计 sidecar。",
            *base.structure_notes,
        ],
        model_in_text=_airborne_model_text(
            "airborne_height_variation_cylinder_v1",
            materials=["#material: 6 0.002 1 0 dry_silty_sand"],
            bodies=[
                _airborne_ground_box("dry_silty_sand"),
                "#cylinder: 0.360 0.175 0 0.360 0.175 0.002 0.020 pec",
            ],
            use_steps=False,
        ),
        uses_per_trace_inputs=False,
        uses_scripted_inputs=True,
        height_variation=True,
    )


def _airborne_definition(
    *,
    scenario_id: str,
    label: str,
    description: str,
    structure_notes: list[str],
    materials: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    layers: list[dict[str, Any]],
    model_materials: list[str],
    bodies: list[str],
    antenna_height_m: float | None = None,
) -> ScenarioDefinition:
    geometry = AIRBORNE_GEOMETRY
    if antenna_height_m is not None:
        geometry = replace(geometry, antenna_height_m=float(antenna_height_m))
    warnings = geometry.validate(runs=geometry.default_runs, targets=targets)
    return ScenarioDefinition(
        scenario_id=scenario_id,
        label=label,
        description=description,
        structure_notes=structure_notes,
        model_in_text=_airborne_model_text(
            scenario_id,
            materials=model_materials,
            bodies=bodies,
            use_steps=True,
            geometry=geometry,
        ),
        materials=materials,
        targets=targets,
        layers=layers,
        domain_m=geometry.domain_m,
        dx_m=geometry.dx_m,
        total_time_ns=geometry.total_time_ns,
        trace_step_m=geometry.trace_step_m,
        source_start_m=geometry.source_start_m,
        receiver_start_m=geometry.receiver_start_m,
        ground_top_y_m=geometry.ground_top_y_m,
        default_runs=geometry.default_runs,
        scenario_family="airborne",
        geometry_model="airborne_2d_tmz_v1",
        antenna_height_m=geometry.antenna_height_m,
        air_layer_thickness_m=geometry.air_layer_thickness_m,
        pml_margin_cells=geometry.pml_margin_cells,
        is_uav_gpr_evidence=True,
        geometry_warnings=warnings,
    )


def _airborne_ground_box(material: str) -> str:
    return f"#box: 0 0 0 0.720 {AIRBORNE_GROUND_TOP_Y_M:.3f} 0.002 {material}"


def _airborne_segmented_ground_boxes(
    material: str,
    *,
    segment_count: int = 18,
    base_y_m: float = AIRBORNE_GROUND_TOP_Y_M,
    amplitude_m: float = 0.010,
) -> list[str]:
    """Return deterministic stepped boxes that approximate a rough air-ground interface."""
    boxes: list[str] = []
    segment_width = AIRBORNE_DOMAIN_M[0] / float(segment_count)
    for idx in range(segment_count):
        x0 = idx * segment_width
        x1 = AIRBORNE_DOMAIN_M[0] if idx == segment_count - 1 else (idx + 1) * segment_width
        phase = (2.0 * np.pi * idx) / float(segment_count)
        top_y = base_y_m + amplitude_m * (0.65 * np.sin(phase) + 0.35 * np.sin(2.7 * phase + 0.4))
        top_y = float(np.clip(top_y, base_y_m - amplitude_m, base_y_m + amplitude_m))
        boxes.append(f"#box: {x0:.3f} 0 0 {x1:.3f} {top_y:.3f} 0.002 {material}")
    return boxes


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
        f"#geometry_view: 0 0 0 {DOMAIN_M[0]:.3f} {DOMAIN_M[1]:.3f} {DOMAIN_M[2]:.3f} {DX_M:.3f} {DX_M:.3f} {DX_M:.3f} {title}_geometry n",
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


def _airborne_model_text(
    title: str,
    *,
    materials: list[str],
    bodies: list[str],
    use_steps: bool,
    geometry: AirborneGeometry | None = None,
    source_m: tuple[float, float, float] | None = None,
    receiver_m: tuple[float, float, float] | None = None,
) -> str:
    geometry = geometry or AIRBORNE_GEOMETRY
    source = source_m or geometry.source_start_m
    receiver = receiver_m or geometry.receiver_start_m
    lines = [
        f"#title: MyGPR {title}",
        f"#domain: {geometry.domain_m[0]:.3f} {geometry.domain_m[1]:.3f} {geometry.domain_m[2]:.3f}",
        f"#dx_dy_dz: {geometry.dx_m:.3f} {geometry.dx_m:.3f} {geometry.dx_m:.3f}",
        f"#time_window: {geometry.total_time_ns * 1e-9:.9g}",
        f"#geometry_view: 0 0 0 {geometry.domain_m[0]:.3f} {geometry.domain_m[1]:.3f} {geometry.domain_m[2]:.3f} {geometry.dx_m:.3f} {geometry.dx_m:.3f} {geometry.dx_m:.3f} {title}_geometry n",
        "",
        *materials,
        "",
        "#waveform: ricker 1 1.5e9 my_ricker",
        f"#hertzian_dipole: z {source[0]:.3f} {source[1]:.3f} {source[2]:.3f} my_ricker",
        f"#rx: {receiver[0]:.3f} {receiver[1]:.3f} {receiver[2]:.3f}",
    ]
    if use_steps:
        lines.extend(
            [
                f"#src_steps: {geometry.trace_step_m:.3f} 0 0",
                f"#rx_steps: {geometry.trace_step_m:.3f} 0 0",
            ]
        )
    lines.extend(["", *bodies, ""])
    return "\n".join(lines)


def _airborne_scripted_height_model_text(definition: ScenarioDefinition) -> str:
    """Build one gprMax input whose Python block moves Tx/Rx height per run."""
    geometry = AIRBORNE_GEOMETRY
    lines = [
        f"#title: MyGPR {definition.scenario_id}",
        f"#domain: {geometry.domain_m[0]:.3f} {geometry.domain_m[1]:.3f} {geometry.domain_m[2]:.3f}",
        f"#dx_dy_dz: {geometry.dx_m:.3f} {geometry.dx_m:.3f} {geometry.dx_m:.3f}",
        f"#time_window: {geometry.total_time_ns * 1e-9:.9g}",
        f"#geometry_view: 0 0 0 {geometry.domain_m[0]:.3f} {geometry.domain_m[1]:.3f} {geometry.domain_m[2]:.3f} {geometry.dx_m:.3f} {geometry.dx_m:.3f} {geometry.dx_m:.3f} {definition.scenario_id}_geometry n",
        "",
        *_model_material_lines(definition),
        "",
        "#waveform: ricker 1 1.5e9 my_ricker",
        "#python:",
        "from math import pi, sin",
        "from gprMax.input_cmd_funcs import hertzian_dipole, rx",
        "run_idx = current_model_run - 1",
        "denom = max(number_model_runs - 1, 1)",
        f"height_m = {geometry.antenna_height_m:.12g} + 0.035 * sin(2.0 * pi * run_idx / denom)",
        f"source_x = {geometry.source_start_x_m:.12g} + run_idx * {geometry.trace_step_m:.12g}",
        f"receiver_x = source_x + {geometry.source_receiver_offset_m:.12g}",
        f"antenna_y = {geometry.ground_top_y_m:.12g} + height_m",
        "hertzian_dipole('z', source_x, antenna_y, 0.0, 'my_ricker')",
        "rx(receiver_x, antenna_y, 0.0)",
        "#end_python:",
        "",
        *_model_body_lines(definition),
        "",
    ]
    return "\n".join(lines)


def _model_material_lines(definition: ScenarioDefinition) -> list[str]:
    return [
        line
        for line in definition.model_in_text.splitlines()
        if line.strip().startswith("#material:")
    ]


def _model_body_lines(definition: ScenarioDefinition) -> list[str]:
    prefixes = ("#box:", "#cylinder:", "#sphere:", "#triangle:", "#plate:")
    return [
        line
        for line in definition.model_in_text.splitlines()
        if line.strip().startswith(prefixes)
    ]


def _target_boundary_points(target: dict[str, Any]) -> list[tuple[float, float]]:
    target_type = str(target.get("type") or "")
    if target_type == "metal_cylinder":
        center = target.get("center_m") or [0.0, 0.0, 0.0]
        radius = float(target.get("radius_m") or 0.0)
        x = float(center[0])
        y = float(center[1])
        return [
            (x - radius, y),
            (x + radius, y),
            (x, y - radius),
            (x, y + radius),
        ]
    if target_type == "air_crack":
        x0 = float(target.get("x0_m", 0.0))
        x1 = float(target.get("x1_m", x0))
        y0 = float(target.get("y0_m", 0.0))
        y1 = float(target.get("y1_m", y0))
        return [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
    if target_type == "layer_interface":
        y = float(target.get("interface_y_m", 0.0))
        return [
            (AIRBORNE_SAFE_MARGIN_M, y),
            (AIRBORNE_DOMAIN_M[0] - AIRBORNE_SAFE_MARGIN_M, y),
        ]
    return []


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
    depth = max(0.0, definition.ground_top_y_m - interface_y)
    if definition.geometry_model.startswith("airborne"):
        apex_time_ns = _air_ground_reflection_time_ns(definition) + (
            2.0 * depth / (LIGHT_SPEED_M_PER_NS / np.sqrt(max(eps, 1.0)))
        )
    else:
        velocity = LIGHT_SPEED_M_PER_NS / np.sqrt(max(eps, 1.0))
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

    top_depth = max(0.0, definition.ground_top_y_m - max(y0, y1))
    bottom_depth = max(0.0, definition.ground_top_y_m - min(y0, y1))
    velocity = LIGHT_SPEED_M_PER_NS / np.sqrt(max(eps, 1.0))
    air_time = _air_ground_reflection_time_ns(definition) if definition.geometry_model.startswith("airborne") else 0.0
    top_sample = _time_to_sample(air_time + 2.0 * top_depth / velocity, total_time_ns, samples)
    bottom_sample = _time_to_sample(air_time + 2.0 * bottom_depth / velocity, total_time_ns, samples)
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
    if definition.geometry_model.startswith("airborne"):
        return _airborne_two_way_time_ns(
            definition,
            tx_x=tx_x,
            tx_y=tx_y,
            rx_x=rx_x,
            rx_y=rx_y,
            target_x=target_x,
            target_y=target_y,
            eps=eps,
        )
    dist_tx = float(np.hypot(tx_x - target_x, tx_y - target_y))
    dist_rx = float(np.hypot(rx_x - target_x, rx_y - target_y))
    velocity = LIGHT_SPEED_M_PER_NS / np.sqrt(max(eps, 1.0))
    return (dist_tx + dist_rx) / velocity


def _airborne_two_way_time_ns(
    definition: ScenarioDefinition,
    *,
    tx_x: float,
    tx_y: float,
    rx_x: float,
    rx_y: float,
    target_x: float,
    target_y: float,
    eps: float,
) -> float:
    return _airborne_one_way_time_ns(
        definition,
        antenna_x=tx_x,
        antenna_y=tx_y,
        target_x=target_x,
        target_y=target_y,
        eps=eps,
    ) + _airborne_one_way_time_ns(
        definition,
        antenna_x=rx_x,
        antenna_y=rx_y,
        target_x=target_x,
        target_y=target_y,
        eps=eps,
    )


def _airborne_one_way_time_ns(
    definition: ScenarioDefinition,
    *,
    antenna_x: float,
    antenna_y: float,
    target_x: float,
    target_y: float,
    eps: float,
) -> float:
    depth = max(0.0, definition.ground_top_y_m - target_y)
    air_height = max(0.0, antenna_y - definition.ground_top_y_m)
    total_vertical = max(air_height + depth, 1.0e-9)
    distance = float(np.hypot(antenna_x - target_x, antenna_y - target_y))
    air_len = distance * air_height / total_vertical
    soil_len = distance * depth / total_vertical
    soil_velocity = LIGHT_SPEED_M_PER_NS / np.sqrt(max(eps, 1.0))
    return air_len / LIGHT_SPEED_M_PER_NS + soil_len / soil_velocity


def _direct_air_wave_time_ns(definition: ScenarioDefinition) -> float:
    offset = abs(float(definition.receiver_start_m[0] - definition.source_start_m[0]))
    return offset / LIGHT_SPEED_M_PER_NS


def _air_ground_reflection_time_ns(definition: ScenarioDefinition) -> float:
    height = max(0.0, float(definition.source_start_m[1] - definition.ground_top_y_m))
    half_offset = abs(float(definition.receiver_start_m[0] - definition.source_start_m[0])) / 2.0
    return 2.0 * float(np.hypot(half_offset, height)) / LIGHT_SPEED_M_PER_NS


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


def _build_wavefield_rois(
    definition: ScenarioDefinition,
    simulation: dict[str, Any],
    *,
    target_rois: list[dict[str, int]],
) -> dict[str, dict[str, Any]]:
    samples = int(simulation["sample_count"])
    traces = int(simulation["trace_count"])
    total_time_ns = float(simulation["total_time_ns"])
    direct_sample = _time_to_sample(
        _direct_air_wave_time_ns(definition),
        total_time_ns,
        samples,
    )
    surface_sample = _time_to_sample(
        _air_ground_reflection_time_ns(definition),
        total_time_ns,
        samples,
    )
    direct_roi = _horizontal_wave_roi(samples, traces, direct_sample, pad=max(3, samples // 180))
    surface_roi = _asymmetric_horizontal_wave_roi(
        samples,
        traces,
        surface_sample,
        before=max(8, samples // 120),
        after=max(24, samples // 12),
    )
    rois: dict[str, dict[str, Any]] = {
        "direct_air_wave": {
            "label": "Tx-Rx 直达空气波",
            "time_ns": float(_direct_air_wave_time_ns(definition)),
            "roi": direct_roi,
            "risk": "不应被误增强或误判为地下目标。",
        },
        "air_ground_reflection": {
            "label": "空气-地表强反射",
            "time_ns": float(_air_ground_reflection_time_ns(definition)),
            "roi": surface_roi,
            "risk": "增益和零时校正不能破坏其与地下目标的时序关系。",
        },
        "background": {
            "label": "地表反射后的目标外背景",
            "roi": _background_roi(samples, traces, surface_roi, target_rois),
            "risk": "AGC/强增益可能把该区域放大为假异常。",
        },
        "late_noise": {
            "label": "晚时窗深部噪声区",
            "roi": {
                "time_start_idx": max(0, int(samples * 0.78)),
                "time_end_idx": samples,
                "dist_start_idx": 0,
                "dist_end_idx": traces,
            },
            "risk": "深部补偿过强时该区域会过曝。",
        },
    }
    if target_rois:
        target_union = _union_rois(target_rois, samples, traces)
        target_start = max(
            target_union["time_start_idx"],
            surface_roi["time_end_idx"] + 1,
        )
        target_union = {
            **target_union,
            "time_start_idx": min(target_start, max(target_union["time_end_idx"] - 1, 0)),
        }
        rois["subsurface_target"] = {
            "label": "地下目标/真实结构",
            "roi": _clamp_roi(target_union, (samples, traces)),
            "risk": "自动选参应保留该 ROI 内的真实结构能量和边缘。",
        }
    return rois


def _horizontal_wave_roi(
    samples: int,
    traces: int,
    sample: int,
    *,
    pad: int,
) -> dict[str, int]:
    return {
        "time_start_idx": max(0, int(sample - pad)),
        "time_end_idx": min(samples, int(sample + pad + 1)),
        "dist_start_idx": 0,
        "dist_end_idx": traces,
    }


def _asymmetric_horizontal_wave_roi(
    samples: int,
    traces: int,
    sample: int,
    *,
    before: int,
    after: int,
) -> dict[str, int]:
    return {
        "time_start_idx": max(0, int(sample - before)),
        "time_end_idx": min(samples, int(sample + after + 1)),
        "dist_start_idx": 0,
        "dist_end_idx": traces,
    }


def _background_roi(
    samples: int,
    traces: int,
    surface_roi: dict[str, int],
    target_rois: list[dict[str, int]],
) -> dict[str, int]:
    start = min(samples - 1, int(surface_roi["time_end_idx"]) + max(6, samples // 80))
    end = max(start + 1, int(samples * 0.72))
    if target_rois:
        target_union = _union_rois(target_rois, samples, traces)
        end = max(start + 1, min(end, int(target_union["time_start_idx"])))
    return {
        "time_start_idx": max(0, start),
        "time_end_idx": min(samples, end),
        "dist_start_idx": 0,
        "dist_end_idx": traces,
    }


def _default_background_roi(
    wavefield_rois: dict[str, dict[str, Any]],
    samples: int,
    traces: int,
) -> dict[str, int]:
    background = wavefield_rois.get("background", {}).get("roi")
    if isinstance(background, dict):
        return _clamp_roi(background, (samples, traces))
    return _full_roi(samples, traces)


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
        pipeline = list(profile_order)
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
    recommendation: str | None = None,
    risk_flags: list[str] | None = None,
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
    if recommendation:
        risk_text = "、".join(risk_flags or []) or "无"
        visual += f" 后端建议为 {recommendation}，风险标记：{risk_text}。"

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
        "backend": comparison.get("backend"),
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
        "overall_recommendation": comparison.get("overall_recommendation"),
        "risk_flags": list(comparison.get("risk_flags", [])),
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
                "recommendation": step.get("recommendation"),
                "risk_flags": list(step.get("risk_flags", [])),
                "rolled_back_to_manual": bool(step.get("rolled_back_to_manual", False)),
                "reason": step.get("reason", ""),
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
        "parameter_domain": _json_safe(result.get("parameter_domain", {})),
        "risk_flags": _json_safe(result.get("risk_flags", [])),
        "risk_level": _json_safe(result.get("risk_level")),
        "risk_reason": result.get("risk_reason"),
        "selection_recommendation": result.get("selection_recommendation"),
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
    simulation = record.get("simulation") or {}
    metrics = comparison.get("metrics", {})
    delta = metrics.get("delta_auto_minus_manual", {})
    verdict = comparison.get("verdict")
    recommendation = comparison.get("overall_recommendation") or verdict
    risk_flags = ", ".join(comparison.get("risk_flags") or []) or "无"
    verdict_label = {
        "auto_better": "自动选参更优",
        "manual_better": "人工 baseline 更优",
        "tie": "二者接近",
    }.get(str(verdict), str(verdict))
    notes = "".join(f"<li>{_esc(note)}</li>" for note in record.get("structure_notes", []))
    step_panels = "\n".join(_render_step_panel(step) for step in record.get("images", []))
    command = " ".join(str(part) for part in record.get("gprmax", {}).get("command", []))
    wavefield_checks = _render_wavefield_checks(record)
    legacy_warning = ""
    if record.get("is_uav_gpr_evidence") is False:
        legacy_warning = (
            "<div class=\"warnings\"><strong>非 UAV-GPR 论文证据</strong>"
            "<p>该 legacy 场景的 Tx/Rx 位于地表附近，只能用于回归 smoke benchmark，"
            "不能支撑航空探地雷达自动选参结论。</p></div>"
        )
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
      {legacy_warning}

      <h3>真实地质结构</h3>
      <div class="structure-grid">
        <img src="{_esc(record.get("structure_preview"))}" alt="{_esc(record.get("label"))} true structure">
        <div>
          <ul>{notes}</ul>
          <p class="note">几何模型：<code>{_esc(record.get("geometry_model"))}</code>。Airborne 场景显式包含空气层、天线离地高度、Tx/Rx 航迹、直达波、地表反射和地下结构；Ground truth ROI 来自正演结构的已知几何位置。</p>
        </div>
      </div>

      {wavefield_checks}

      <h3>gprMax 命令</h3>
      <pre>{_esc(command)}</pre>

      <h3>最终指标摘要</h3>
      <div class="metric-grid">
        <div><span>后端建议</span><strong>{_esc(recommendation)}</strong></div>
        <div><span>全流程风险标记</span><strong>{_esc(risk_flags)}</strong></div>
        <div><span>处理 BScan 尺寸</span><strong>{_esc(f"{simulation.get('sample_count')} x {simulation.get('trace_count')}")}</strong></div>
        <div><span>gprMax 原始点数</span><strong>{_esc(simulation.get("raw_sample_count") or simulation.get("sample_count"))}</strong></div>
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


def _render_wavefield_checks(record: dict[str, Any]) -> str:
    ground_truth = record.get("ground_truth") or {}
    rois = ground_truth.get("wavefield_rois") or {}
    if not isinstance(rois, dict) or not rois:
        return ""
    label_map = {
        "direct_air_wave": "直达波",
        "air_ground_reflection": "地表反射",
        "subsurface_target": "地下目标",
        "background": "背景区",
        "late_noise": "深部噪声",
    }
    rows = []
    for key in [
        "direct_air_wave",
        "air_ground_reflection",
        "subsurface_target",
        "background",
        "late_noise",
    ]:
        item = rois.get(key)
        if not isinstance(item, dict):
            continue
        roi = item.get("roi") or {}
        rows.append(
            f"<tr><td>{_esc(label_map.get(key, key))}</td>"
            f"<td><code>{_esc(json.dumps(roi, ensure_ascii=False, sort_keys=True))}</code></td>"
            f"<td>{_esc(item.get('risk') or '')}</td></tr>"
        )
    if not rows:
        return ""
    return f"""
      <h3>波场特征检查</h3>
      <p class="note">检查顺序应为直达波 &lt; 地表反射 &lt; 地下目标。自动选参和增益不能把直达波/地表反射误增强为地下异常，也不能让零时校正破坏三者时序。</p>
      <table class="params-table">
        <thead><tr><th>波场特征</th><th>ROI</th><th>风险检查</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
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
    decision = _render_step_decision(step)
    analysis = step.get("analysis") or {}
    return f"""
      <article class="step-card">
        <h4>{_esc(step.get("method_name"))} <code>{_esc(step.get("method_key"))}</code></h4>
        {params_table}
        {decision}
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


def _render_step_decision(step: dict[str, Any]) -> str:
    recommendation = step.get("recommendation") or "n/a"
    risk_flags = ", ".join(step.get("risk_flags") or []) or "无"
    rolled_back = "是" if step.get("rolled_back_to_manual") else "否"
    reason = step.get("reason") or ""
    return f"""
        <div class="decision-grid">
          <div><span>后端建议</span><strong>{_esc(recommendation)}</strong></div>
          <div><span>风险标记</span><strong>{_esc(risk_flags)}</strong></div>
          <div><span>是否回退人工结果</span><strong>{_esc(rolled_back)}</strong></div>
          <div><span>判定依据</span><strong>{_esc(reason)}</strong></div>
        </div>
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
    risk_flags = ", ".join(tune.get("risk_flags") or []) or "无"
    recommendation = tune.get("selection_recommendation") or "review"
    manual_note = "经验 baseline 或日常页面同步参数"
    if step.get("manual_original_params") != step.get("manual_params"):
        manual_note = f"本步实际参数已按报告策略修正；原人工参数为 {manual_original}"
    domain = tune.get("parameter_domain") or {}
    domain_notes = list(domain.get("notes") or [])
    domain_note_html = ""
    if domain_notes:
        domain_note_html = (
            f"<p class=\"note\"><strong>参数域提示：</strong>{_esc('；'.join(domain_notes[:3]))}</p>"
        )
    return f"""
        <table class="params-table">
          <thead><tr><th>分支</th><th>参数</th><th>说明</th></tr></thead>
        <tbody>
            <tr><td>人工选参</td><td><code>{_esc(manual)}</code></td><td>{_esc(manual_note)}</td></tr>
            <tr><td>自动选参</td><td><code>{_esc(auto)}</code></td><td>{_esc(reason)}</td></tr>
        </tbody>
        </table>
        <p class="note"><strong>风险标记：</strong>{_esc(risk_flags)}，<strong>建议动作：</strong>{_esc(recommendation)}</p>
        {domain_note_html}
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
.kv-grid, .metric-grid, .decision-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}
.kv-grid div, .metric-grid div, .decision-grid div {
  padding: 12px;
  background: #f8f9fb;
  border: 1px solid #e4e7ec;
  border-radius: 6px;
}
.kv-grid span, .metric-grid span, .decision-grid span {
  display: block;
  margin-bottom: 5px;
  color: #667085;
  font-size: 13px;
}
.decision-grid {
  margin: 10px 0 12px;
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
    parser.add_argument(
        "--scenario-family",
        choices=["airborne", "legacy", "all"],
        default=DEFAULT_SCENARIO_FAMILY,
        help="Scenario family to expose. Default: airborne UAV-GPR scenes.",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--ascan-samples",
        type=int,
        default=DEFAULT_ASCAN_SAMPLE_COUNT,
        help=(
            "Target A-scan sample count used by MyGPR processing/reporting. "
            "Use 501 or 701 to match current UAV-GPR field data; use 0 to keep raw gprMax samples."
        ),
    )
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
    parser.add_argument(
        "--cuda-vcvars",
        default=None,
        help=(
            "Optional path to Visual Studio vcvars64.bat. On Windows GPU runs, "
            "the script auto-loads this environment when cl.exe is not already on PATH."
        ),
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
        scenario_family=str(args.scenario_family),
        runs=int(args.runs),
        geometry_fixed=bool(args.geometry_fixed),
        mpi=args.mpi,
        gpu=list(args.gpu or []),
        python_override=args.python_exe,
        search_mode=str(args.search_mode),
        baseline_profile_key=str(args.baseline_profile),
        zero_time_policy=str(args.zero_time_policy),
        ascan_samples=int(args.ascan_samples),
        cuda_vcvars=args.cuda_vcvars,
        extra_args=list(args.extra_arg or []),
    )
    print(f"HTML report: {payload['html_report']}")
    print(f"Summary JSON: {payload['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
