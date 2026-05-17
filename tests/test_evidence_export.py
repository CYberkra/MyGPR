#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence export regression tests for standard processing chains."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from core.evidence_export import (
    STANDARD_CHAIN_SPECS,
    export_chain_evidence,
    export_replay_evidence_bundle,
    export_standard_chain_for_sample,
)
from core.shared_data_state import SharedDataState


def test_export_chain_evidence_writes_summary_and_images(tmp_path: Path):
    raw = np.arange(48, dtype=np.float32).reshape(12, 4)
    summary = export_chain_evidence(
        data=raw,
        header_info={
            "a_scan_length": 12,
            "num_traces": 4,
            "total_time_ns": 120.0,
            "trace_interval_m": 0.5,
        },
        bundle_name="demo-bundle",
        chain_name="演示链",
        chain_description="dewow only",
        steps=[("dewow", {"window": 5})],
        out_dir=tmp_path,
        title_prefix="Demo",
        save_images=True,
    )

    assert summary["bundle_name"] == "demo-bundle"
    assert summary["chain_name"] == "演示链"
    assert len(summary["steps"]) == 1
    assert summary["steps"][0]["method_key"] == "dewow"
    assert (tmp_path / "demo-bundle-00-raw.png").exists()
    assert (tmp_path / "demo-bundle-01-dewow.png").exists()
    assert (tmp_path / "demo-bundle-raw-vs-final.png").exists()
    assert (tmp_path / "demo-bundle-summary.json").exists()


def test_export_standard_chain_for_sample_supports_both_stage_a_chains(tmp_path: Path):
    for chain_key in ("conservative_default", "aggressive_gain"):
        summary = export_standard_chain_for_sample(
            sample_id="drift_background_reference",
            chain_key=chain_key,
            out_dir=tmp_path / chain_key,
            save_images=False,
        )
        assert summary["sample_id"] == "drift_background_reference"
        assert summary["chain_name"] == STANDARD_CHAIN_SPECS[chain_key]["label"]
        assert len(summary["steps"]) == len(STANDARD_CHAIN_SPECS[chain_key]["steps"])
        assert (
            tmp_path
            / chain_key
            / f"drift_background_reference-{chain_key}-summary.json"
        ).exists()
    agc_step = next(
        params
        for method_key, params in STANDARD_CHAIN_SPECS["aggressive_gain"]["steps"]
        if method_key == "agcGain"
    )
    assert agc_step["_low_energy_guard"] is True


def test_shared_state_replay_package_stays_in_memory_until_export(tmp_path: Path):
    state = SharedDataState()
    raw = np.arange(24, dtype=np.float32).reshape(6, 4)
    state.load_data(
        raw,
        path="demo.csv",
        header_info={"total_time_ns": 60.0, "trace_interval_m": 0.25},
        trace_metadata={"distance_m": np.linspace(0.0, 0.75, 4)},
    )
    state.apply_current_data(raw + 1.0, push_history=True, label="dewow")

    package = state.get_replay_evidence_package()
    assert package is not None
    assert package["package_type"] == "mygpr_replay_evidence"
    assert package["storage"] == "memory_only_until_user_export"
    assert package["snapshot_count"] >= 2
    assert package["snapshots"][0]["role"] == "original"
    assert package["snapshots"][-1]["role"] == "current"
    assert list(tmp_path.iterdir()) == []


def test_export_replay_evidence_bundle_writes_one_zip(tmp_path: Path):
    state = SharedDataState()
    raw = np.arange(24, dtype=np.float32).reshape(6, 4)
    state.load_data(
        raw,
        path="demo.csv",
        header_info={"total_time_ns": 60.0, "trace_interval_m": 0.25},
        trace_metadata={"distance_m": np.linspace(0.0, 0.75, 4)},
    )
    state.apply_current_data(raw + 1.0, push_history=True, label="dewow")

    zip_path = tmp_path / "replay.zip"
    result = export_replay_evidence_bundle(
        state.get_replay_evidence_package(),
        zip_path,
        save_images=False,
    )

    assert result["zip_path"] == str(zip_path.resolve())
    assert [path.name for path in tmp_path.iterdir()] == ["replay.zip"]
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "summary.json" in names
        assert "report.md" in names
        assert any(name.startswith("data/") and name.endswith(".npy") for name in names)
        assert any(
            name.startswith("metadata/") and name.endswith("-header.json")
            for name in names
        )
        summary = json.loads(zf.read("summary.json").decode("utf-8"))

    assert summary["storage"] == "memory_only_until_user_export"
    assert summary["snapshot_count"] >= 2


def test_export_replay_evidence_bundle_includes_motion_preview_artifacts(tmp_path: Path):
    state = SharedDataState()
    raw = np.arange(120, dtype=np.float32).reshape(12, 10)
    metadata = {
        "trace_distance_m": np.linspace(0.0, 1.8, 10),
        "flight_height_m": np.linspace(1.0, 1.2, 10),
    }
    state.load_data(
        raw,
        path="demo.csv",
        header_info={"total_time_ns": 60.0, "trace_interval_m": 0.2},
        trace_metadata=metadata,
    )
    state.apply_current_data(raw + 2.0, push_history=True, label="motion_compensation_v2")
    package = state.get_replay_evidence_package()
    assert package is not None
    package["app_context"] = {
        "preset_key": "motion_compensation_v2",
        "runtime_warnings": [{"code": "demo"}],
        "method_param_overrides": {"motion_compensation_v2": {"max_shift_ns": 20.0}},
    }

    zip_path = tmp_path / "replay.zip"
    export_replay_evidence_bundle(package, zip_path, save_images=True)

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "motion/raw_bscan.png" in names
        assert "motion/current_bscan.png" in names
        assert "motion/diff_bscan.png" in names
        assert "motion/raw_3d_preview.png" in names
        assert "motion/current_3d_preview.png" in names
        assert "motion/diff_3d_preview.png" in names
        assert "motion/motion_quality_flags.json" in names
        assert "motion/motion_params.json" in names
