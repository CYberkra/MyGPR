#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for single motion demo case export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import export_single_motion_demo_cases as demo


def test_jsonable_removes_nonfinite_values_for_strict_motion_demo_json(tmp_path: Path):
    payload = {
        "metric": np.float64(np.inf),
        "array": np.array([1.0, np.nan, np.inf], dtype=np.float32),
        "path": tmp_path / "demo",
    }

    safe = demo._jsonable(payload)

    assert safe["metric"] is None
    assert safe["array"] == [1.0, None, None]
    assert safe["path"] == str(tmp_path / "demo")
    json.dumps(safe, allow_nan=False)


def test_write_json_uses_strict_motion_demo_json(tmp_path: Path):
    out_path = tmp_path / "summary.json"

    demo._write_json(out_path, {"value": float("nan")})

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["value"] is None
    json.dumps(payload, allow_nan=False)
