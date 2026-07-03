#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-tune comparison research artifact export tests."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from core.auto_tune_comparison import run_auto_tune_comparison
from core.auto_tune_comparison_export import (
    _locked_display_spec,
    export_auto_tune_comparison_artifacts,
)


def _build_export_fixture(samples: int = 84, traces: int = 24) -> np.ndarray:
    rng = np.random.default_rng(29)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(-1.0, 1.0, traces, dtype=np.float64)[None, :]
    data = 0.36 * np.sin(2.0 * np.pi * 0.7 * t)
    data += 0.12 * np.sin(2.0 * np.pi * 9.0 * t)
    data = np.repeat(data, traces, axis=1)
    data += 0.04 * rng.normal(size=(samples, traces))
    hyperbola = 28 + np.round(10.0 * np.square(x)).astype(int).reshape(-1)
    for trace_idx, row in enumerate(hyperbola):
        data[row : row + 3, trace_idx] += np.array([0.25, 0.85, 0.25])
    data[56:60, 7:18] += 0.18 * np.hanning(4)[:, None]
    return data.astype(np.float32)


def _truth_manifest() -> dict:
    return {
        "schema": "mygpr_gprmax_ground_truth_v1",
        "scenario_id": "export_truth_demo",
        "analysis_roi": {
            "time_start_idx": 18,
            "time_end_idx": 68,
            "dist_start_idx": 3,
            "dist_end_idx": 21,
        },
        "targets": [
            {
                "target_id": "pipe_01",
                "type": "pipe",
                "material": "metal",
                "depth_m": 0.35,
                "must_preserve": True,
                "roi": {
                    "time_start_idx": 24,
                    "time_end_idx": 40,
                    "dist_start_idx": 6,
                    "dist_end_idx": 18,
                },
            }
        ],
        "background_rois": [
            {
                "time_start_idx": 8,
                "time_end_idx": 18,
                "dist_start_idx": 2,
                "dist_end_idx": 8,
            }
        ],
        "metrics_contract": {"purpose": "test"},
        "source_paths": {
            "manifest_file": "case001_manifest.json",
            "ground_truth_file": "ground_truth.yaml",
        },
        "raw_sidecar": {
            "schema": "gprmax_ground_truth_v1",
            "dataset_id": "export_truth_demo",
        },
        "conversion_warnings": ["test warning"],
    }


def test_export_auto_tune_comparison_artifacts_writes_research_bundle(tmp_path: Path):
    raw = _build_export_fixture()
    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        roi_spec={
            "mode": "manual",
            "bounds": {
                "time_start_idx": 18,
                "time_end_idx": 68,
                "dist_start_idx": 3,
                "dist_end_idx": 21,
            },
            "label": "synthetic-target-roi",
        },
        search_mode="fast",
    )

    bundle = export_auto_tune_comparison_artifacts(
        result,
        out_dir=tmp_path,
        bundle_name="case001",
        input_ref="synthetic://case001",
        notes=["GPRMAX forward-model cases will reuse this export contract."],
    )

    expected = {
        "summary_json",
        "manual_png",
        "auto_png",
        "side_by_side_png",
        "params_csv",
        "metrics_csv",
        "report_md",
    }
    assert expected <= set(bundle["artifacts"])

    for key in expected:
        path = Path(bundle["artifacts"][key])
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    summary = json.loads(
        Path(bundle["artifacts"]["summary_json"]).read_text(encoding="utf-8")
    )
    assert summary["input_ref"] == "synthetic://case001"
    assert summary["verdict"] == result.verdict
    assert summary["roi_info"]["label"] == "synthetic-target-roi"
    assert summary["display_spec"]["locked_scale"] is True
    assert "result" not in summary["manual"]
    assert "result" not in summary["automatic"]

    with Path(bundle["artifacts"]["params_csv"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        params_rows = list(csv.DictReader(handle))
    assert {
        (row["candidate"], row["method_key"], row["param_name"])
        for row in params_rows
    } >= {
        ("manual", "dewow", "window"),
        ("automatic", "dewow", "window"),
    }

    with Path(bundle["artifacts"]["metrics_csv"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        metrics_rows = list(csv.DictReader(handle))
    assert "comparison_score" in {row["metric"] for row in metrics_rows}

    report_text = Path(bundle["artifacts"]["report_md"]).read_text(encoding="utf-8")
    assert "# AutoTune gprMax Evidence Report" in report_text
    assert "## 5. Trial Summary" in report_text
    assert "## 7. Reproducibility" in report_text
    assert "## 8. Research Boundary" in report_text
    assert "synthetic://case001" in report_text


def test_export_auto_tune_comparison_artifacts_writes_truth_evidence_bundle(tmp_path: Path):
    raw = _build_export_fixture()
    result = run_auto_tune_comparison(
        raw,
        header_info={"ground_truth": _truth_manifest()},
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )

    bundle = export_auto_tune_comparison_artifacts(
        result,
        out_dir=tmp_path,
        bundle_name="truth_case",
        input_ref="synthetic://truth_case",
    )

    required = {
        "summary_json": "comparison_summary.json",
        "evidence_manifest_json": "evidence_manifest.json",
        "converted_ground_truth_json": "converted_ground_truth.json",
        "raw_ground_truth_json": "raw_ground_truth.json",
        "truth_metrics_json": "truth_metrics.json",
        "workflow_params_json": "workflow_params.json",
        "trial_table_csv": "trial_table.csv",
        "trial_table_json": "trial_table.json",
        "params_csv": "params_table.csv",
        "metrics_csv": "metrics_table.csv",
        "manual_png": "manual_bscan.png",
        "auto_png": "auto_bscan.png",
        "side_by_side_png": "side_by_side.png",
        "report_md": "comparison_report.md",
        "evidence_zip": "evidence_bundle.zip",
    }
    for key, file_name in required.items():
        path = Path(bundle["artifacts"][key])
        assert path.name == file_name
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    manifest = json.loads(
        Path(bundle["artifacts"]["evidence_manifest_json"]).read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "mygpr_autotune_evidence_v1"
    assert manifest["ground_truth"]["enabled"] is True
    assert manifest["ground_truth"]["scenario_id"] == "export_truth_demo"
    assert manifest["ground_truth"]["has_background_rois"] is True
    assert manifest["artifacts"]["converted_ground_truth_json"]["status"] == "available"
    assert manifest["artifacts"]["evidence_zip"]["status"] == "available"

    converted = json.loads(
        Path(bundle["artifacts"]["converted_ground_truth_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert converted["scenario_id"] == "export_truth_demo"
    assert converted["raw_sidecar"]["schema"] == "gprmax_ground_truth_v1"

    truth_metrics = json.loads(
        Path(bundle["artifacts"]["truth_metrics_json"]).read_text(encoding="utf-8")
    )
    assert truth_metrics["enabled"] is True
    assert truth_metrics["manual"]["truth_score"] is not None
    assert truth_metrics["automatic"]["truth_score"] is not None

    workflow = json.loads(
        Path(bundle["artifacts"]["workflow_params_json"]).read_text(encoding="utf-8")
    )
    assert workflow["pipeline"] == ["dewow"]
    assert workflow["manual_params_by_method"]["dewow"]["window"] == 1
    assert "dewow" in workflow["automatic_params_by_method"]

    with Path(bundle["artifacts"]["trial_table_csv"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        trial_rows = list(csv.DictReader(handle))
    assert any(row["branch"] == "manual" and row["selected"] == "True" for row in trial_rows)
    assert any(row["branch"] == "automatic" for row in trial_rows)

    trial_payload = json.loads(
        Path(bundle["artifacts"]["trial_table_json"]).read_text(encoding="utf-8")
    )
    assert trial_payload["schema"] == "mygpr_autotune_trial_table_v1"
    assert "dewow" in trial_payload["methods"]

    with zipfile.ZipFile(bundle["artifacts"]["evidence_zip"]) as zf:
        names = set(zf.namelist())
    assert {
        "comparison_summary.json",
        "evidence_manifest.json",
        "converted_ground_truth.json",
        "truth_metrics.json",
        "workflow_params.json",
        "trial_table.csv",
        "trial_table.json",
        "side_by_side.png",
    } <= names


def test_export_auto_tune_comparison_artifacts_marks_missing_truth_files(tmp_path: Path):
    raw = _build_export_fixture(samples=72, traces=18)
    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )

    bundle = export_auto_tune_comparison_artifacts(
        result,
        out_dir=tmp_path,
        bundle_name="no_truth_case",
    )

    manifest = json.loads(
        Path(bundle["artifacts"]["evidence_manifest_json"]).read_text(encoding="utf-8")
    )
    assert manifest["ground_truth"]["enabled"] is False
    assert manifest["artifacts"]["converted_ground_truth_json"]["status"] == "missing"
    assert manifest["artifacts"]["raw_ground_truth_json"]["status"] == "missing"

    truth_metrics = json.loads(
        Path(bundle["artifacts"]["truth_metrics_json"]).read_text(encoding="utf-8")
    )
    assert truth_metrics == {
        "enabled": False,
        "reason": "ground truth unavailable",
    }


def test_export_auto_tune_comparison_artifacts_serializes_nonfinite_metrics(tmp_path: Path):
    raw = _build_export_fixture(samples=72, traces=18)
    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )
    result.manual.metrics["nan_metric"] = np.float64(np.nan)
    result.automatic.metrics["inf_metric"] = np.array([np.inf])

    bundle = export_auto_tune_comparison_artifacts(
        result,
        out_dir=tmp_path,
        bundle_name="nonfinite_case",
    )

    summary = json.loads(
        Path(bundle["artifacts"]["summary_json"]).read_text(encoding="utf-8")
    )
    assert summary["manual"]["metrics"]["nan_metric"] is None
    assert summary["automatic"]["metrics"]["inf_metric"] == [None]

    with Path(bundle["artifacts"]["metrics_csv"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}
    assert rows["nan_metric"]["manual_value"] == ""
    assert rows["inf_metric"]["auto_value"] == "[null]"


def test_export_auto_tune_comparison_accepts_numpy_scalar_percentile_clip(tmp_path: Path):
    raw = _build_export_fixture(samples=72, traces=18)
    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        display_spec={"percentile_clip": np.array([95.0])},
        search_mode="fast",
    )

    bundle = export_auto_tune_comparison_artifacts(
        result,
        out_dir=tmp_path,
        bundle_name="numpy_clip_case",
    )

    summary = json.loads(
        Path(bundle["artifacts"]["summary_json"]).read_text(encoding="utf-8")
    )
    assert summary["display_spec"]["percentile_clip"] == [95.0]
    assert np.isfinite(summary["display_spec"]["vmin"])
    assert np.isfinite(summary["display_spec"]["vmax"])


def test_export_auto_tune_comparison_closes_figures_on_save_error(
    tmp_path: Path,
    monkeypatch,
):
    import matplotlib.pyplot as plt

    raw = _build_export_fixture(samples=72, traces=18)
    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        search_mode="fast",
    )
    before = set(plt.get_fignums())

    def _raise_save_error(*_args, **_kwargs):
        raise RuntimeError("savefig failed")

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", _raise_save_error)

    with pytest.raises(RuntimeError, match="savefig failed"):
        export_auto_tune_comparison_artifacts(
            result,
            out_dir=tmp_path,
            bundle_name="save_error_case",
        )

    assert set(plt.get_fignums()) == before


def test_locked_display_spec_ignores_nonfinite_values_when_scaling():
    manual = np.array([[np.nan, -2.0], [np.inf, 1.0]], dtype=np.float32)
    auto = np.array([[0.0, 3.0], [-np.inf, 4.0]], dtype=np.float32)

    spec = _locked_display_spec(
        manual,
        auto,
        {"percentile_clip": 100.0},
        cmap="gray",
    )

    assert spec["vmin"] == -4.0
    assert spec["vmax"] == 4.0


def test_export_preserves_autotune_v1_candidate_space_manifest_fields(tmp_path: Path):
    raw = _build_export_fixture(samples=72, traces=18)
    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        roi_spec={
            "mode": "manual",
            "bounds": {
                "time_start_idx": 10,
                "time_end_idx": 50,
                "dist_start_idx": 2,
                "dist_end_idx": 14,
            },
            "label": "manual-review-roi",
        },
        search_mode="fast",
    )
    selected = dict(result.automatic.params_by_method.get("dewow") or {"window": 7})
    result.automatic.auto_tune_results = {
        "dewow": {
            "best_score": 0.88,
            "best_reason": "selected from V1 bounded candidate space",
            "recommended_params": selected,
            "all_trials": [
                {
                    "trial_index": 0,
                    "stage": "coarse",
                    "params": selected,
                    "score": 0.88,
                    "reason": "selected from V1 bounded candidate space",
                    "warnings": [],
                    "valid": True,
                    "candidate_space_hash": "sha256:demo-space",
                    "candidate_space_profile_id": "landslide_bedrock_sliding_surface",
                    "candidate_space_config_version": "autotune_v1_profiles.v1",
                    "candidate_space_recipe_ids": ["landslide_bedrock_interface"],
                    "candidate_id": "dewow.window.7",
                    "candidate_source": "adaptive",
                    "candidate_group": "dewow_window",
                    "candidate_parameters": {"window": selected.get("window", 7)},
                    "score_version": "autotune_scoring_v2",
                }
            ],
            "execution_stats": {"total_trial_count": 1},
        }
    }

    bundle = export_auto_tune_comparison_artifacts(
        result,
        out_dir=tmp_path,
        bundle_name="v1_manifest_case",
        input_ref="field://no-prior-demo",
    )

    manifest = json.loads(
        Path(bundle["artifacts"]["evidence_manifest_json"]).read_text(encoding="utf-8")
    )
    v1 = manifest["autotune_v1"]
    assert v1["candidate_space_hashes"] == ["sha256:demo-space"]
    assert v1["profile_ids"] == ["landslide_bedrock_sliding_surface"]
    assert v1["scoring_boundary"] == "field_no_prior_proxy"
    assert v1["manual_review_required"] is True
    assert "RMSE" in v1["forbidden_metrics"]

    trial_payload = json.loads(
        Path(bundle["artifacts"]["trial_table_json"]).read_text(encoding="utf-8")
    )
    assert trial_payload["autotune_v1_evidence"]["candidate_ids"] == ["dewow.window.7"]

    with Path(bundle["artifacts"]["trial_table_csv"]).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected_rows = [row for row in rows if row["branch"] == "automatic" and row["selected"] == "True"]
    assert selected_rows
    row = selected_rows[0]
    assert row["candidate_space_hash"] == "sha256:demo-space"
    assert row["candidate_space_profile_id"] == "landslide_bedrock_sliding_surface"
    assert row["candidate_id"] == "dewow.window.7"
    assert row["scoring_boundary"] == "field_no_prior_proxy"
    assert row["manual_review_required"] == "True"

    workflow = json.loads(
        Path(bundle["artifacts"]["workflow_params_json"]).read_text(encoding="utf-8")
    )
    assert workflow["autotune_v1_candidate_space"]["candidate_space_hashes"] == ["sha256:demo-space"]

    report_text = Path(bundle["artifacts"]["report_md"]).read_text(encoding="utf-8")
    assert "## 5.1 AutoTune V1 Evidence Boundary" in report_text
    assert "field_no_prior_proxy" in report_text
