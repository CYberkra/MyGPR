#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gprMax dataset contract readiness auditing."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.gprmax_benchmark.audit_gprmax_dataset_contract import (
    audit_dataset_dir,
    audit_gprmax_path,
    write_mygpr_manifest,
)


def _write_ground_truth(path: Path, *, output_file: str = "pipe_merged.out") -> None:
    path.write_text(
        "\n".join(
            [
                "schema: gprmax_ground_truth_v1",
                "dataset_id: sim_pipe_001",
                "model_file: sim_pipe_001.in",
                f"output_file: {output_file}",
                "target_roi:",
                "  trace_range: [42, 48]",
                "  sample_range: [720, 860]",
                "target:",
                "  type: pipe",
                "  material: pec",
            ]
        ),
        encoding="utf-8",
    )


def test_audit_reports_missing_output_and_manifest(tmp_path: Path):
    _write_ground_truth(tmp_path / "ground_truth.yaml", output_file="missing_merged.out")

    audit = audit_dataset_dir(tmp_path)

    assert audit["scenario_id"] == "sim_pipe_001"
    assert audit["has_ground_truth"] is True
    assert audit["has_output"] is False
    assert audit["has_manifest"] is False
    assert audit["ready_for_mygpr_smoke"] is False
    assert audit["can_write_manifest"] is False
    assert any("missing referenced output_file" in item for item in audit["errors"])
    assert any("missing MyGPR manifest JSON" in item for item in audit["warnings"])


def test_audit_can_write_manifest_when_output_exists(tmp_path: Path):
    _write_ground_truth(tmp_path / "ground_truth.yaml")
    (tmp_path / "pipe_merged.out").write_bytes(b"placeholder")

    audit = audit_dataset_dir(tmp_path)
    manifest_path = write_mygpr_manifest(audit)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert audit["can_write_manifest"] is True
    assert manifest_path.name == "sim_pipe_001_manifest.json"
    assert manifest["schema"] == "gprmax_dataset_manifest_v1"
    assert manifest["scenario_id"] == "sim_pipe_001"
    assert manifest["primary_out_file"] == "pipe_merged.out"
    assert manifest["ground_truth_file"] == "ground_truth.yaml"


def test_audit_ready_when_manifest_output_and_truth_exist(tmp_path: Path):
    _write_ground_truth(tmp_path / "ground_truth.yaml")
    (tmp_path / "pipe_merged.out").write_bytes(b"placeholder")
    (tmp_path / "sim_pipe_001_manifest.json").write_text(
        json.dumps(
            {
                "schema": "gprmax_dataset_manifest_v1",
                "scenario_id": "sim_pipe_001",
                "primary_out_file": "pipe_merged.out",
                "ground_truth_file": "ground_truth.yaml",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit = audit_gprmax_path(tmp_path)

    assert audit["ready_for_mygpr_smoke"] is True
    assert audit["can_write_manifest"] is False
    assert audit["manifest_files"] == [str(tmp_path / "sim_pipe_001_manifest.json")]


def test_root_audit_finds_ground_truth_datasets(tmp_path: Path):
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    _write_ground_truth(ready_dir / "ground_truth.yaml")
    (ready_dir / "pipe_merged.out").write_bytes(b"placeholder")
    (ready_dir / "ready_manifest.json").write_text(
        json.dumps(
            {
                "schema": "gprmax_dataset_manifest_v1",
                "scenario_id": "ready",
                "primary_out_file": "pipe_merged.out",
                "ground_truth_file": "ground_truth.yaml",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    _write_ground_truth(incomplete_dir / "ground_truth.yaml", output_file="missing.out")

    audit = audit_gprmax_path(tmp_path)

    assert audit["mode"] == "root"
    assert audit["dataset_count"] == 2
    assert audit["ready_count"] == 1
    assert {Path(item["dataset_dir"]).name for item in audit["datasets"]} == {
        "ready",
        "incomplete",
    }
