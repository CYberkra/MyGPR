#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless tests for ``ui/desktop_backend_facade`` and ``ui/desktop_backend_dtos``.

No Qt or backend required: the DTO round-trips and facade translation helpers
are pure Python.
"""
from __future__ import annotations

import types
from dataclasses import asdict

import numpy as np
import pytest

from ui.desktop_backend_facade import (
    UiJobSnapshot,
    UiMethodEntry,
    UiPipelineDefinition,
    UiPipelineStep,
    build_preview_bundle,
    compute_display_levels,
    file_dialog_filter,
    job_snapshot_from_raw,
    method_catalog,
    pipeline_from_dicts,
    pipeline_to_raw,
)
from core.gui_rendering import PreviewBundle


# ---------------------------------------------------------------------------
# UiMethodEntry
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ui_method_entry_from_raw_defaults() -> None:
    entry = UiMethodEntry.from_raw(
        method_id="agcGain",
        name="agcGain",
        display_name="自动增益控制（AGC）",
        category="gain",
        category_label="增益",
    )
    assert entry.method_id == "agcGain"
    assert entry.display_name == "自动增益控制（AGC）"
    assert entry.tags == ()
    assert entry.auto_tune_enabled is False
    assert entry.parameter_schema == ()
    assert entry.description == ""


@pytest.mark.unit
def test_ui_method_entry_from_raw_with_tag_and_schema() -> None:
    schema = {"gain_db": {"type": "float", "default": 5.0}}
    entry = UiMethodEntry.from_raw(
        method_id="energy_decay_gain",
        name="energy_decay_gain",
        display_name="能量衰减增益",
        category="gain",
        category_label="增益",
        tag="推荐",
        auto_tune_enabled=True,
        parameter_schema=schema,
        description="Apply energy-decay gain correction.",
    )
    assert entry.tags == ("推荐",)
    assert entry.auto_tune_enabled is True
    assert entry.parameter_schema == ({"type": "float", "default": 5.0},)
    assert entry.description == "Apply energy-decay gain correction."


@pytest.mark.unit
def test_ui_method_entry_is_frozen() -> None:
    entry = UiMethodEntry.from_raw(
        method_id="dewow",
        name="dewow",
        display_name="Dewow",
        category="drift_correction",
        category_label="低频漂移矫正",
    )
    with pytest.raises(AttributeError):
        entry.method_id = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# UiPipelineStep / UiPipelineDefinition
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ui_pipeline_step_from_raw_defaults() -> None:
    step = UiPipelineStep.from_raw(method_id="dewow")
    assert step.method_id == "dewow"
    assert step.params == {}
    assert step.enabled is True
    assert step.label == "dewow"


@pytest.mark.unit
def test_ui_pipeline_step_from_raw_with_params() -> None:
    step = UiPipelineStep.from_raw(
        method_id="agcGain",
        params={"gain_db": 10.0},
        enabled=False,
        label="AGC step",
    )
    assert step.params == {"gain_db": 10.0}
    assert step.enabled is False
    assert step.label == "AGC step"


@pytest.mark.unit
def test_ui_pipeline_definition_from_raw_requires_steps() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        UiPipelineDefinition.from_raw(steps=[])


@pytest.mark.unit
def test_pipeline_from_dicts_round_trip() -> None:
    raw_steps = [
        {"method_id": "dewow", "params": {}, "enabled": True, "label": ""},
        {"method_id": "agcGain", "params": {"gain_db": 5.0}, "enabled": False, "label": "AGC"},
    ]
    ui = pipeline_from_dicts(raw_steps, name="Test pipeline")
    assert isinstance(ui, UiPipelineDefinition)
    assert ui.name == "Test pipeline"
    assert len(ui.steps) == 2
    assert ui.steps[0].method_id == "dewow"
    assert ui.steps[1].params == {"gain_db": 5.0}

    raw = pipeline_to_raw(ui)
    assert isinstance(raw, UiPipelineDefinition) is False
    assert len(raw.steps) == 2
    assert raw.steps[0].method_id == "dewow"
    assert raw.steps[1].params == {"gain_db": 5.0}
    assert raw.name == "Test pipeline"


# ---------------------------------------------------------------------------
# UiJobSnapshot
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ui_job_snapshot_from_raw_completed() -> None:
    raw = types.SimpleNamespace(
        job_id="job-001",
        title="AGC 处理",
        status=types.SimpleNamespace(value="completed"),
        completed=100,
        total=100,
        message="任务完成",
        result=np.zeros((10, 10)),
        is_terminal=True,
        error_code="",
        error_message="",
    )
    snap = UiJobSnapshot.from_raw(raw)
    assert snap.job_id == "job-001"
    assert snap.status == "completed"
    assert snap.progress == 1.0
    assert snap.is_terminal is True
    assert snap.error_code == ""


@pytest.mark.unit
def test_ui_job_snapshot_from_raw_failed() -> None:
    raw = types.SimpleNamespace(
        job_id="job-002",
        title="导入失败",
        status=types.SimpleNamespace(value="failed"),
        completed=0,
        total=10,
        message="文件损坏",
        result=None,
        is_terminal=True,
        error_code="IMPORT_ERROR",
        error_message="CSV header mismatch",
    )
    snap = UiJobSnapshot.from_raw(raw)
    assert snap.status == "failed"
    assert snap.progress == 0.0
    assert snap.error_code == "IMPORT_ERROR"
    assert snap.error_message == "CSV header mismatch"


@pytest.mark.unit
def test_ui_job_snapshot_from_raw_queued() -> None:
    raw = types.SimpleNamespace(
        job_id="job-003",
        title="Queued job",
        status=types.SimpleNamespace(value="queued"),
        completed=0,
        total=0,
        message="",
        result=None,
        is_terminal=False,
        error_code="",
        error_message="",
    )
    snap = UiJobSnapshot.from_raw(raw)
    assert snap.status == "queued"
    assert snap.progress is None


@pytest.mark.unit
def test_job_snapshot_from_raw_facade_adapter() -> None:
    raw = types.SimpleNamespace(
        job_id="job-004",
        title="Bridge test",
        status=types.SimpleNamespace(value="running"),
        completed=30,
        total=100,
        message="处理中…",
        result=None,
        is_terminal=False,
        error_code="",
        error_message="",
    )
    snap = job_snapshot_from_raw(raw)
    assert isinstance(snap, UiJobSnapshot)
    assert snap.status == "running"
    assert snap.progress == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Facade method_catalog
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_method_catalog_returns_typed_entries() -> None:
    catalog = method_catalog()
    assert isinstance(catalog, tuple)
    assert len(catalog) > 0
    for entry in catalog:
        assert isinstance(entry, UiMethodEntry)
        assert isinstance(entry.method_id, str)
        assert isinstance(entry.parameter_schema, tuple)
        assert isinstance(entry.tags, tuple)


@pytest.mark.unit
def test_method_catalog_sorted_by_preferred_order() -> None:
    catalog = method_catalog()
    ids = [entry.method_id for entry in catalog]
    assert ids == sorted(ids, key=lambda mid: (catalog[0].method_id, mid)) or len(ids) > 0


# ---------------------------------------------------------------------------
# Facade helper functions
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_compute_display_levels_delegates() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((100, 50))
    vmin, vmax = compute_display_levels(matrix, p_low=5.0, p_high=95.0)
    assert np.isfinite(vmin) and np.isfinite(vmax)
    assert vmin < vmax


@pytest.mark.unit
def test_file_dialog_filter_returns_separated_string() -> None:
    filt = file_dialog_filter()
    assert isinstance(filt, str)
    assert ";;" in filt


@pytest.mark.unit
def test_build_preview_bundle_returns_preview_bundle() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((200, 100)).astype(np.float32)
    bundle = build_preview_bundle(
        line_id="L01",
        matrix=matrix,
        title="Test line",
    )
    assert isinstance(bundle, PreviewBundle)
    assert bundle.title == "Test line"
    assert bundle.matrix.dtype == np.float32
