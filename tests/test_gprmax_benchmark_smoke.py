#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-run helper tests for GPRMAX benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.gprmax_benchmark.run_gprmax_cylinder_single_v1_smoke import (
    build_gprmax_command,
    find_out_files,
    resolve_gprmax_python,
)


def test_resolve_gprmax_python_prefers_venv_python(tmp_path: Path):
    root = tmp_path / "gprmax"
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    resolved = resolve_gprmax_python(root)
    assert resolved == venv_python


def test_resolve_gprmax_python_uses_override_when_provided(tmp_path: Path):
    root = tmp_path / "gprmax"
    root.mkdir()
    override = tmp_path / "python_custom.exe"
    override.write_text("", encoding="utf-8")

    resolved = resolve_gprmax_python(root, python_override=str(override))
    assert resolved == override


def test_build_gprmax_command_includes_runs_and_geometry_flag(tmp_path: Path):
    command = build_gprmax_command(
        tmp_path / "python.exe",
        tmp_path / "model.in",
        runs=12,
        geometry_only=True,
        extra_args=["--write-processed"],
    )
    assert command[:3] == [str(tmp_path / "python.exe"), "-m", "gprMax"]
    assert "-n" in command and "12" in command
    assert "--geometry-only" in command
    assert "--write-processed" in command


def test_find_out_files_uses_numeric_suffix_sort(tmp_path: Path):
    for name in [
        "cylinder_single_v110.out",
        "cylinder_single_v12.out",
        "cylinder_single_v130.out",
    ]:
        (tmp_path / name).write_text("", encoding="utf-8")
    ordered = find_out_files(tmp_path, "cylinder_single_v1")
    assert [path.name for path in ordered] == [
        "cylinder_single_v12.out",
        "cylinder_single_v110.out",
        "cylinder_single_v130.out",
    ]
