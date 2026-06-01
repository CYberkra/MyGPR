#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for preflight generated artifact staging guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import preflight_check


def test_generated_artifact_path_detection_blocks_native_outputs() -> None:
    assert preflight_check._is_generated_artifact_path(
        "experiments/gprmax/GX-007/models/scene/raw_with_target1.out"
    )
    assert preflight_check._is_generated_artifact_path(
        "some/external/native_output.vti"
    )
    assert preflight_check._is_generated_artifact_path(
        "experiments/gprmax/GX-007/scene/paired_outputs/target_response.npy"
    )


def test_generated_artifact_path_detection_allows_curated_fixtures() -> None:
    assert not preflight_check._is_generated_artifact_path(
        "tests/fixtures/gprmax_campaign/pairing/raw_valid.csv"
    )
    assert not preflight_check._is_generated_artifact_path(
        "docs/ui_smoke_001_screenshots/wide_basic.png"
    )
    assert not preflight_check._is_generated_artifact_path(
        "experiments/gprmax/GX-007/gx007_complete_2d_run_audit.md"
    )


def test_staged_generated_artifacts_raise_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="experiments/gprmax/GX-007/scene/paired_outputs/preview.png\n",
        )

    monkeypatch.setattr(preflight_check.subprocess, "run", _fake_run)
    with pytest.raises(AssertionError, match="Generated gprMax/native artifacts"):
        preflight_check.check_staged_generated_artifacts()
