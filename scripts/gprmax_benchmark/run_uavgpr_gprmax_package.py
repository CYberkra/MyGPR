#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch the gprMax UAV-GPR package worker from MyGPR."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("_uavgpr_gprmax_worker.py")


def default_gprmax_python(gprmax_root: str | Path) -> Path:
    """Return the preferred Python executable for a local gprMax checkout."""
    root = Path(gprmax_root).expanduser().resolve()
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
        root / "venv" / "Scripts" / "python.exe",
        root / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def build_worker_command(args: argparse.Namespace) -> list[str]:
    """Build the subprocess command for the gprMax worker."""
    python_exe = Path(args.python).expanduser().resolve() if args.python else default_gprmax_python(args.gprmax_root)
    command = [
        str(python_exe),
        str(WORKER),
        "--gprmax-root",
        str(Path(args.gprmax_root).expanduser().resolve()),
        "--output-root",
        str(Path(args.output_root).expanduser().resolve()),
        "--output-name",
        args.output_name,
        "--preset",
        args.preset,
        "--run-timeout-s",
        str(args.run_timeout_s),
    ]
    _append_optional(command, "--python", str(python_exe))
    _append_optional(command, "--title", args.title)
    _append_optional(command, "--traces", args.traces)
    _append_flag(command, "--gpu", args.gpu)
    _append_flag(command, "--no-geometry-fixed", not args.geometry_fixed)
    _append_flag(command, "--timestamp-output", args.timestamp_output)
    _append_flag(command, "--write-geometry-view", args.write_geometry_view)
    _append_flag(command, "--no-center-target", not args.center_target)
    for cli_name, value in (
        ("--domain-x", args.domain_x),
        ("--domain-y", args.domain_y),
        ("--dx", args.dx),
        ("--dy", args.dy),
        ("--time-window-ns", args.time_window_ns),
        ("--host-name", args.host_name),
        ("--host-eps-r", args.host_eps_r),
        ("--host-sigma", args.host_sigma),
        ("--ground-surface-y", args.ground_surface_y),
        ("--lift-off", args.lift_off),
        ("--source-start-x", args.source_start_x),
        ("--receiver-offset", args.receiver_offset),
        ("--scan-step", args.scan_step),
        ("--center-freq-mhz", args.center_freq_mhz),
        ("--target-shape", args.target_shape),
        ("--target-name", args.target_name),
        ("--target-eps-r", args.target_eps_r),
        ("--target-sigma", args.target_sigma),
        ("--target-center-x", args.target_center_x),
        ("--target-center-y", args.target_center_y),
        ("--target-radius", args.target_radius),
        ("--target-width", args.target_width),
        ("--target-height", args.target_height),
        ("--target-orientation", args.target_orientation),
        ("--target-angle-deg", args.target_angle_deg),
        ("--source-type", args.source_type),
        ("--waveform-type", args.waveform_type),
        ("--source-resistance", args.source_resistance),
        ("--source-polarisation", args.source_polarisation),
        ("--background-layer-json", args.background_layer_json),
    ):
        _append_optional(command, cli_name, value)
    return command


def run_worker(args: argparse.Namespace) -> int:
    """Run the worker and stream output."""
    command = build_worker_command(args)
    print("Running worker:", flush=True)
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, text=True)
    return int(completed.returncode)


def _append_optional(command: list[str], name: str, value: object | None) -> None:
    if value is None:
        return
    text = str(value)
    if text == "":
        return
    command.extend([name, text])


def _append_flag(command: list[str], name: str, enabled: bool) -> None:
    if enabled:
        command.append(name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a UAV-GPR gprMax package using a local gprMax checkout.",
    )
    parser.add_argument("--gprmax-root", required=True)
    parser.add_argument("--python", default="")
    parser.add_argument("--output-root", default=str(ROOT / "output" / "gprmax_datasets"))
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--preset", default="uav_pipe_gain_workflow_bscan")
    parser.add_argument("--title", default="")
    parser.add_argument("--traces", type=int)
    parser.add_argument("--run-timeout-s", type=float, default=0.0)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--geometry-fixed", action="store_true", default=True)
    parser.add_argument("--no-geometry-fixed", dest="geometry_fixed", action="store_false")
    parser.add_argument("--timestamp-output", action="store_true")
    parser.add_argument("--write-geometry-view", action="store_true")
    parser.add_argument("--center-target", action="store_true", default=True)
    parser.add_argument("--no-center-target", dest="center_target", action="store_false")
    parser.add_argument("--domain-x", type=float)
    parser.add_argument("--domain-y", type=float)
    parser.add_argument("--dx", type=float)
    parser.add_argument("--dy", type=float)
    parser.add_argument("--time-window-ns", type=float)
    parser.add_argument("--host-name")
    parser.add_argument("--host-eps-r", type=float)
    parser.add_argument("--host-sigma", type=float)
    parser.add_argument("--ground-surface-y", type=float)
    parser.add_argument("--lift-off", type=float)
    parser.add_argument("--source-start-x", type=float)
    parser.add_argument("--receiver-offset", type=float)
    parser.add_argument("--scan-step", type=float)
    parser.add_argument("--center-freq-mhz", type=float)
    parser.add_argument("--target-shape")
    parser.add_argument("--target-name")
    parser.add_argument("--target-eps-r", type=float)
    parser.add_argument("--target-sigma", type=float)
    parser.add_argument("--target-center-x", type=float)
    parser.add_argument("--target-center-y", type=float)
    parser.add_argument("--target-radius", type=float)
    parser.add_argument("--target-width", type=float)
    parser.add_argument("--target-height", type=float)
    parser.add_argument("--target-orientation")
    parser.add_argument("--target-angle-deg", type=float)
    parser.add_argument("--source-type")
    parser.add_argument("--waveform-type")
    parser.add_argument("--source-resistance", type=float)
    parser.add_argument("--source-polarisation")
    parser.add_argument(
        "--background-layer-json",
        default="",
        help="JSON list of gprMax background layer objects.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
