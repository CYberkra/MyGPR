#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Worker executed with the gprMax Python environment to generate UAV-GPR packages."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any


def _import_gprmax_gui(gprmax_root: Path):
    root = gprmax_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gprmax_gui_pyside6 as gui  # type: ignore

    return gui


def build_config(args: argparse.Namespace) -> Any:
    """Build a gprMax GUI SimulationConfig from CLI arguments."""
    gui = _import_gprmax_gui(Path(args.gprmax_root))
    preset = gui.PRESETS[args.preset]
    config = gui.SimulationConfig(
        title=args.title or preset["title"],
        output_root=args.output_root,
        output_name=args.output_name,
        python_executable=args.python or sys.executable,
        timestamp_output=bool(args.timestamp_output),
        use_gpu=bool(args.gpu),
        geometry_fixed=bool(args.geometry_fixed),
        geometry_only=False,
        run_timeout_s=float(args.run_timeout_s),
        write_geometry_view=bool(args.write_geometry_view),
        domain_x=float(args.domain_x if args.domain_x is not None else preset["domain_x"]),
        domain_y=float(args.domain_y if args.domain_y is not None else preset["domain_y"]),
        dx=float(args.dx if args.dx is not None else preset["dx"]),
        dy=float(args.dy if args.dy is not None else preset["dy"]),
        time_window_ns=float(
            args.time_window_ns if args.time_window_ns is not None else preset["time_window_ns"]
        ),
        host_name=str(args.host_name or preset["host_name"]),
        host_eps_r=float(args.host_eps_r if args.host_eps_r is not None else preset["host_eps_r"]),
        host_sigma=float(
            args.host_sigma if args.host_sigma is not None else preset["host_sigma"]
        ),
        ground_surface_y=float(
            args.ground_surface_y
            if args.ground_surface_y is not None
            else preset["ground_surface_y"]
        ),
        lift_off=float(args.lift_off if args.lift_off is not None else preset["lift_off"]),
        source_start_x=float(
            args.source_start_x
            if args.source_start_x is not None
            else preset["source_start_x"]
        ),
        receiver_offset=float(
            args.receiver_offset
            if args.receiver_offset is not None
            else preset["receiver_offset"]
        ),
        scan_step=float(args.scan_step if args.scan_step is not None else preset["scan_step"]),
        n_traces=int(args.traces if args.traces is not None else preset["n_traces"]),
        center_freq_mhz=float(
            args.center_freq_mhz
            if args.center_freq_mhz is not None
            else preset["center_freq_mhz"]
        ),
        target_shape=str(args.target_shape or preset["target_shape"]),
        target_name=str(args.target_name or preset["target_name"]),
        target_eps_r=float(
            args.target_eps_r if args.target_eps_r is not None else preset["target_eps_r"]
        ),
        target_sigma=float(
            args.target_sigma
            if args.target_sigma is not None
            else preset["target_sigma"]
        ),
        target_center_x=float(
            args.target_center_x
            if args.target_center_x is not None
            else preset["target_center_x"]
        ),
        target_center_y=float(
            args.target_center_y
            if args.target_center_y is not None
            else preset["target_center_y"]
        ),
        target_radius=float(
            args.target_radius
            if args.target_radius is not None
            else preset["target_radius"]
        ),
        target_width=float(
            args.target_width if args.target_width is not None else preset["target_width"]
        ),
        target_height=float(
            args.target_height
            if args.target_height is not None
            else preset["target_height"]
        ),
        target_orientation=str(args.target_orientation or preset["target_orientation"]),
        target_angle_deg=float(
            args.target_angle_deg
            if args.target_angle_deg is not None
            else preset["target_angle_deg"]
        ),
        use_curved_crack=bool(preset.get("use_curved_crack", False)),
        crack_path_text=str(preset.get("crack_path_text", "")),
        background_layers=copy.deepcopy(preset.get("background_layers", [])),
        source_type=str(args.source_type or preset.get("source_type", "hertzian_dipole")),
        waveform_type=str(args.waveform_type or preset.get("waveform_type", "ricker")),
        source_resistance=float(
            args.source_resistance
            if args.source_resistance is not None
            else preset.get("source_resistance", 0.0)
        ),
        source_polarisation=str(
            args.source_polarisation or preset.get("source_polarisation", "z")
        ),
        preset_key=args.preset,
    )
    if args.background_layer_json:
        layers = json.loads(args.background_layer_json)
        if not isinstance(layers, list):
            raise ValueError("--background-layer-json must decode to a list")
        config.background_layers = layers
    if args.center_target:
        desired_start_x = (
            config.target_center_x
            - 0.5 * config.receiver_offset
            - 0.5 * (config.n_traces - 1) * config.effective_scan_step
        )
        max_start_x = (
            config.domain_x
            - config.receiver_offset
            - (config.n_traces - 1) * config.effective_scan_step
        )
        config.source_start_x = min(max(desired_start_x, 0.0), max(0.0, max_start_x))
    return config


def run_package(args: argparse.Namespace) -> dict[str, Any]:
    """Generate a package and return a JSON-safe run summary."""
    gui = _import_gprmax_gui(Path(args.gprmax_root))
    config = build_config(args)
    logs: list[str] = []
    auditor = gui.PhysicsAuditor()
    report = auditor.build_report(config)
    if report.has_errors():
        messages = [item.text for item in report.messages]
        raise RuntimeError("gprMax physics audit failed: " + "; ".join(messages))

    builder = gui.ScenarioBuilder()
    artifacts = builder.build_files(config, report)
    runner = gui.GprMaxRunner(log_callback=logs.append)
    start = time.time()
    artifacts = runner.run(config, artifacts)
    end = time.time()
    _write_readme(config, artifacts, start_time=start, end_time=end)

    summary = {
        "scenario_id": config.output_name,
        "preset": config.preset_key,
        "start_time": start,
        "end_time": end,
        "duration_s": end - start,
        "output_dir": artifacts.output_dir,
        "input_file": artifacts.input_path,
        "primary_out_file": artifacts.primary_out_path,
        "merged_out_file": artifacts.merged_out_path,
        "manifest_file": artifacts.manifest_path,
        "ground_truth_file": artifacts.ground_truth_path,
        "metadata_file": artifacts.metadata_path,
        "preview_file": artifacts.preview_path,
        "bscan_preview_file": artifacts.bscan_png_path,
        "background_removed_preview_file": artifacts.background_removed_png_path,
        "background_removed_mild_gain_preview_file": artifacts.background_removed_gain_png_path,
        "n_traces": config.n_traces,
        "scan_step_m": config.effective_scan_step,
        "target": {
            "shape": config.target_shape,
            "material": config.target_name,
            "center_x_m": config.target_center_x,
            "center_y_m": config.target_center_y,
            "radius_m": config.target_radius,
        },
        "medium": {
            "host_name": config.host_name,
            "host_eps_r": config.host_eps_r,
            "host_sigma": config.host_sigma,
        },
        "log_tail": logs[-80:],
    }
    summary_path = Path(artifacts.output_dir) / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["generation_summary_file"] = str(summary_path)
    return summary


def _write_readme(config: Any, artifacts: Any, *, start_time: float, end_time: float) -> None:
    path = Path(artifacts.output_dir) / "README.md"
    text = "\n".join(
        [
            f"# {config.output_name}",
            "",
            "Generated by MyGPR gprMax AutoTune validation bridge.",
            "",
            "## Scenario",
            f"- preset: `{config.preset_key}`",
            f"- traces: `{config.n_traces}`",
            f"- scan_step_m: `{config.effective_scan_step}`",
            f"- tx_start_x_m: `{config.source_start_x}`",
            f"- rx_offset_m: `{config.receiver_offset}`",
            f"- target: `{config.target_shape}` / `{config.target_name}`",
            f"- target_center_m: `({config.target_center_x}, {config.target_center_y})`",
            f"- target_radius_m: `{config.target_radius}`",
            f"- host: `{config.host_name}`, eps_r=`{config.host_eps_r}`, sigma=`{config.host_sigma}`",
            f"- time_window_ns: `{config.time_window_ns}`",
            "",
            "## Files",
            f"- input: `{Path(artifacts.input_path).name}`",
            f"- primary_out: `{Path(artifacts.primary_out_path).name if artifacts.primary_out_path else ''}`",
            f"- manifest: `{Path(artifacts.manifest_path).name}`",
            f"- ground_truth: `{Path(artifacts.ground_truth_path).name}`",
            f"- metadata: `{Path(artifacts.metadata_path).name}`",
            "",
            "## Runtime",
            f"- start_unix: `{start_time}`",
            f"- end_unix: `{end_time}`",
            f"- duration_s: `{end_time - start_time:.3f}`",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a gprMax UAV-GPR validation package.")
    parser.add_argument("--gprmax-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--preset", default="uav_pipe_gain_workflow_bscan")
    parser.add_argument("--python", default="")
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
        help="JSON list of background layer objects with name/eps_r/sigma/y_min/y_max.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_package(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
