#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for explicit gprMax receiver component selection in converter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gprmax_campaign_convert_scene001.py"


def _write_series(base_dir: Path, stem: str, run_count: int, components: list[str]) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    base = base_dir / f"{stem}.out"
    for i in range(1, run_count + 1):
        p = base_dir / f"{stem}{i}.out"
        with h5py.File(p, "w") as f:
            rxs = f.create_group("rxs")
            rx1 = rxs.create_group("rx1")
            for comp in components:
                rx1.create_dataset(comp, data=np.array([float(i), float(i) + 0.5], dtype=np.float64))
    return base


def test_convert_series_with_explicit_component(tmp_path):
    raw_base = _write_series(tmp_path / "raw", "raw_with_target", 3, ["Ey", "Ez"])
    bg_base = _write_series(tmp_path / "bg", "background_only", 3, ["Ey", "Ez"])
    out_json = tmp_path / "summary.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--raw-out",
        str(raw_base),
        "--background-out",
        str(bg_base),
        "--raw-converted-dir",
        str(tmp_path / "raw_conv"),
        "--background-converted-dir",
        str(tmp_path / "bg_conv"),
        "--raw-run-count",
        "3",
        "--background-run-count",
        "3",
        "--component",
        "Ey",
        "--json",
        str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["selected_component"] == "Ey"
    assert payload["raw"]["selected_component"] == "Ey"
    assert "Ez" in payload["raw"]["available_components"]


def test_convert_series_missing_component_fails(tmp_path):
    raw_base = _write_series(tmp_path / "raw", "raw_with_target", 2, ["Ez"])
    bg_base = _write_series(tmp_path / "bg", "background_only", 2, ["Ez"])
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--raw-out",
        str(raw_base),
        "--background-out",
        str(bg_base),
        "--raw-converted-dir",
        str(tmp_path / "raw_conv"),
        "--background-converted-dir",
        str(tmp_path / "bg_conv"),
        "--raw-run-count",
        "2",
        "--background-run-count",
        "2",
        "--component",
        "Ey",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "Missing requested component" in (proc.stderr + proc.stdout)

