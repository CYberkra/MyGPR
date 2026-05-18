#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a real gprMax smoke for cylinder_single_v1 and convert outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gprmax_benchmark.generate_cylinder_single_v1 import (  # noqa: E402
    SCENARIO_ID,
    generate_package,
)


DEFAULT_GPRMAX_ROOT = Path(r"E:\gprMax\gprMax-v.3.1.7")
DEFAULT_SMOKE_ROOT = ROOT / "output" / "gprmax_smoke" / SCENARIO_ID


@dataclass(frozen=True)
class SmokeResult:
    """Summary paths for one smoke execution."""

    smoke_dir: Path
    model_input: Path
    command: list[str]
    raw_out_files: list[Path]
    converted_package_dir: Path | None
    summary_json: Path


def resolve_gprmax_python(
    gprmax_root: Path,
    python_override: str | None = None,
) -> Path:
    """Resolve Python interpreter for running gprMax."""
    if python_override:
        candidate = Path(python_override)
        if not candidate.exists():
            raise FileNotFoundError(f"python override not found: {candidate}")
        return candidate
    venv_python = gprmax_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def build_gprmax_command(
    python_exe: Path,
    model_input: Path,
    *,
    runs: int,
    geometry_only: bool,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the command line for gprMax."""
    command = [
        str(python_exe),
        "-m",
        "gprMax",
        str(model_input),
        "-n",
        str(int(runs)),
    ]
    if geometry_only:
        command.append("--geometry-only")
    if extra_args:
        command.extend(str(item) for item in extra_args)
    return command


def find_out_files(smoke_dir: Path, base_stem: str) -> list[Path]:
    """Find trace outputs sorted by numeric suffix."""
    out_files = []
    for path in smoke_dir.glob(f"{base_stem}*.out"):
        suffix = path.stem[len(base_stem) :]
        number = int(suffix) if suffix.isdigit() else 0
        out_files.append((number, path))
    out_files.sort(key=lambda item: item[0])
    return [path for _, path in out_files]


def run_smoke(
    *,
    gprmax_root: Path = DEFAULT_GPRMAX_ROOT,
    smoke_root: Path = DEFAULT_SMOKE_ROOT,
    runs: int = 9,
    geometry_only: bool = False,
    python_override: str | None = None,
    extra_args: list[str] | None = None,
) -> SmokeResult:
    """Run gprMax and convert the first output series into MyGPR benchmark package."""
    gprmax_root = Path(gprmax_root)
    if not gprmax_root.exists():
        raise FileNotFoundError(f"gprMax root not found: {gprmax_root}")
    if runs <= 0:
        raise ValueError("runs must be >= 1")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    smoke_dir = Path(smoke_root) / ts
    smoke_dir.mkdir(parents=True, exist_ok=True)

    base_package = generate_package(smoke_dir / "scenario_seed")
    model_input = smoke_dir / f"{SCENARIO_ID}.in"
    model_input.write_text(base_package.model_in_path.read_text(encoding="utf-8"), encoding="utf-8")

    python_exe = resolve_gprmax_python(gprmax_root, python_override=python_override)
    command = build_gprmax_command(
        python_exe,
        model_input,
        runs=runs,
        geometry_only=geometry_only,
        extra_args=extra_args,
    )
    env = os.environ.copy()
    process = subprocess.run(
        command,
        cwd=str(gprmax_root),
        env=env,
        capture_output=True,
        text=True,
    )

    out_files = find_out_files(smoke_dir, SCENARIO_ID)
    converted_package_dir = None
    if process.returncode == 0 and (not geometry_only) and out_files:
        converted = generate_package(
            smoke_dir / "converted_package",
            raw_out_path=out_files[0],
        )
        converted_package_dir = converted.package_dir

    summary = {
        "scenario_id": SCENARIO_ID,
        "timestamp": ts,
        "gprmax_root": str(gprmax_root),
        "python_executable": str(python_exe),
        "command": command,
        "returncode": int(process.returncode),
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
        "geometry_only": bool(geometry_only),
        "runs": int(runs),
        "raw_out_files": [str(path) for path in out_files],
        "converted_package_dir": str(converted_package_dir) if converted_package_dir else None,
    }
    summary_json = smoke_dir / "smoke_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if process.returncode != 0:
        raise RuntimeError(
            "gprMax smoke failed. See summary: "
            + str(summary_json)
        )

    return SmokeResult(
        smoke_dir=smoke_dir,
        model_input=model_input,
        command=command,
        raw_out_files=out_files,
        converted_package_dir=converted_package_dir,
        summary_json=summary_json,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gprMax smoke for MyGPR cylinder_single_v1 benchmark."
    )
    parser.add_argument(
        "--gprmax-root",
        default=str(DEFAULT_GPRMAX_ROOT),
        help="gprMax repository root containing module `gprMax`.",
    )
    parser.add_argument(
        "--smoke-root",
        default=str(DEFAULT_SMOKE_ROOT),
        help="Output root for smoke runs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=9,
        help="Number of model runs (`-n`).",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Run gprMax with --geometry-only.",
    )
    parser.add_argument(
        "--python-exe",
        default=None,
        help="Optional Python interpreter override for gprMax.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument appended to gprMax command; can be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_smoke(
        gprmax_root=Path(args.gprmax_root),
        smoke_root=Path(args.smoke_root),
        runs=int(args.runs),
        geometry_only=bool(args.geometry_only),
        python_override=args.python_exe,
        extra_args=list(args.extra_arg or []),
    )
    print(f"Smoke run directory: {result.smoke_dir}")
    print(f"Summary: {result.summary_json}")
    if result.converted_package_dir:
        print(f"Converted package: {result.converted_package_dir}")
    else:
        print("No converted package generated (geometry-only or no .out files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
