#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the MyGPR-to-gprMax UAV-GPR package runner."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.gprmax_benchmark.run_uavgpr_gprmax_package import (
    build_worker_command,
    default_gprmax_python,
)


def _args(tmp_path: Path, **overrides):
    values = {
        "gprmax_root": str(tmp_path / "gprmax"),
        "python": "",
        "output_root": str(tmp_path / "out"),
        "output_name": "pipe_demo_longline_v1",
        "preset": "uav_pipe_gain_workflow_bscan",
        "title": "",
        "traces": 90,
        "run_timeout_s": 0.0,
        "gpu": False,
        "geometry_fixed": True,
        "timestamp_output": False,
        "write_geometry_view": False,
        "center_target": True,
        "domain_x": None,
        "domain_y": None,
        "dx": None,
        "dy": None,
        "time_window_ns": None,
        "host_name": None,
        "host_eps_r": None,
        "host_sigma": None,
        "ground_surface_y": None,
        "lift_off": None,
        "source_start_x": None,
        "receiver_offset": None,
        "scan_step": None,
        "center_freq_mhz": None,
        "target_shape": None,
        "target_name": None,
        "target_eps_r": None,
        "target_sigma": None,
        "target_center_x": None,
        "target_center_y": None,
        "target_radius": None,
        "target_width": None,
        "target_height": None,
        "target_orientation": None,
        "target_angle_deg": None,
        "source_type": None,
        "waveform_type": None,
        "source_resistance": None,
        "source_polarisation": None,
        "background_layer_json": "",
    }
    values.update(overrides)
    return Namespace(**values)


def test_default_gprmax_python_prefers_local_venv(tmp_path: Path):
    root = tmp_path / "gprmax"
    exe = root / ".venv" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")

    assert default_gprmax_python(root) == exe.resolve()


def test_build_worker_command_preserves_explicit_dataset_controls(tmp_path: Path):
    root = tmp_path / "gprmax"
    python = root / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    command = build_worker_command(
        _args(
            tmp_path,
            gprmax_root=str(root),
            output_root=str(tmp_path / "datasets"),
            output_name="pipe_demo_v1",
            traces=72,
            gpu=True,
            host_eps_r=9.0,
            target_center_x=0.62,
            background_layer_json='[{"name":"moist_layer","eps_r":14,"sigma":0.01,"y_min":0.32,"y_max":0.36}]',
        )
    )

    assert command[0] == str(python.resolve())
    assert "--gprmax-root" in command
    assert str(root.resolve()) in command
    assert "--output-name" in command
    assert "pipe_demo_v1" in command
    assert "--traces" in command
    assert "72" in command
    assert "--gpu" in command
    assert "--host-eps-r" in command
    assert "9.0" in command
    assert "--target-center-x" in command
    assert "0.62" in command
    assert "--background-layer-json" in command
    assert "moist_layer" in command[command.index("--background-layer-json") + 1]
