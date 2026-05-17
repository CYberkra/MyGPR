#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for workflow template manager persistence boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from core.workflow_template_manager import WorkflowTemplateManager


def test_template_manager_writes_strict_json_for_nonfinite_params(tmp_path: Path):
    manager = WorkflowTemplateManager(config_dir=str(tmp_path))
    template = manager.create_template("严格模板")
    template.add_method(
        "sec_gain",
        {
            "gain_min": float("nan"),
            "gain_max": float("inf"),
            "enabled": True,
        },
    )
    manager._save_templates()

    payload = json.loads((tmp_path / "templates.json").read_text(encoding="utf-8"))
    params = payload["严格模板"]["methods"][0]["params"]
    assert params["gain_min"] is None
    assert params["gain_max"] is None
    assert params["enabled"] is True
    json.dumps(payload, allow_nan=False)


def test_template_export_writes_strict_json_for_nonfinite_params(tmp_path: Path):
    manager = WorkflowTemplateManager(config_dir=str(tmp_path / "templates"))
    template = manager.create_template("导出模板")
    template.add_method("agcGain", {"window": float("nan")})

    export_path = tmp_path / "export.json"
    manager.export_template(template.name, str(export_path))

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["methods"][0]["params"]["window"] is None
    json.dumps(payload, allow_nan=False)
