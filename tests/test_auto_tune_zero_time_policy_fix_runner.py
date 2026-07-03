#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AT-006 zero-time policy fix runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.auto_tune_validation.run_zero_time_policy_fix import run_policy_fix


def test_zero_time_policy_fix_runner_writes_required_outputs(tmp_path: Path):
    evidence_root = tmp_path / "AT-006"
    result = run_policy_fix(
        evidence_root=evidence_root,
        dataset="cylinder_single_v1",
        source_repo="https://github.com/CYberkra/MyGPR",
        source_branch="codex/research-gprmax-autotune",
        source_commit="test-source-commit",
    )

    assert result["source_commit"] == "test-source-commit"
    required = [
        "reports/zero_time_policy_fix_report.md",
        "reports/zero_time_policy_fix_report.html",
        "manifests/evidence_manifest.json",
        "manifests/zero_time_policy_summary.json",
        "tables/zero_time_policy_checks.csv",
        "figures/zero_time_policy_before_after.png",
    ]
    for rel in required:
        assert (evidence_root / rel).exists(), rel

    summary = json.loads((evidence_root / "manifests/zero_time_policy_summary.json").read_text(encoding="utf-8"))
    assert "zero_time_policy" in summary
    assert "legacy_default_behavior" in summary
    assert "guarded_behavior" in summary
