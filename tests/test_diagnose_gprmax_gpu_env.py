#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for GPU diagnostic script output schema."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_gprmax_gpu_env.py"
WRAPPER = ROOT / "scripts" / "run_gprmax_gpu_env.bat"


def test_diagnose_outputs_readiness_fields(tmp_path):
    out = tmp_path / "diag.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--skip-gprmax-smoke",
        "--json",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    readiness = payload["readiness"]
    required = {
        "host_python_pycuda_available",
        "gprmax_python",
        "gprmax_python_exists",
        "gprmax_help_ok",
        "cl_available",
        "nvcc_available",
        "nvidia_smi_available",
        "minimal_smoke_ok",
        "gprmax_runtime_gpu_ready",
        "readiness_reason",
    }
    assert required.issubset(set(readiness.keys()))


def test_gpu_wrapper_exists_and_has_required_modes():
    assert WRAPPER.exists()
    text = WRAPPER.read_text(encoding="utf-8", errors="ignore")
    assert "--check" in text
    assert "--smoke" in text
    assert "MYGPR_VCVARS64" in text
    assert "MYGPR_GPRMAX_PYTHON" in text
    assert "MYGPR_GPU_DEVICE" in text
