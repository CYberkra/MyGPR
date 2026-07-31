#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public orchestration API for manual-baseline versus auto-tune evidence export."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.auto_tune_comparison import AutoTuneComparisonRun, to_summary_dict
from core.auto_tune_comparison_export_common import _json_safe, _safe_bundle_name
from core.auto_tune_comparison_export_evidence import (
    _build_evidence_manifest, _write_csv_rows, _write_evidence_zip,
)
from core.auto_tune_comparison_export_images import (
    _locked_display_spec, _save_side_by_side, _save_single_bscan,
)
from core.auto_tune_comparison_export_markdown import _build_report_markdown
from core.auto_tune_comparison_export_tables import (
    _build_metrics_rows, _build_params_rows, _build_trial_table,
    _build_truth_metrics, _build_workflow_params,
)

def export_auto_tune_comparison_artifacts(
    result: AutoTuneComparisonRun,
    *,
    out_dir: str | Path,
    bundle_name: str | None = None,
    input_ref: str | None = None,
    notes: list[str] | None = None,
    cmap: str = "gray",
) -> dict[str, Any]:
    """Export a complete manual-vs-auto research evidence bundle."""
    output_root = Path(out_dir) / _safe_bundle_name(bundle_name)
    output_root.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_bundle_name(bundle_name)

    paths = {
        "summary_json": output_root / "comparison_summary.json",
        "evidence_manifest_json": output_root / "evidence_manifest.json",
        "converted_ground_truth_json": output_root / "converted_ground_truth.json",
        "raw_ground_truth_json": output_root / "raw_ground_truth.json",
        "truth_metrics_json": output_root / "truth_metrics.json",
        "workflow_params_json": output_root / "workflow_params.json",
        "trial_table_csv": output_root / "trial_table.csv",
        "trial_table_json": output_root / "trial_table.json",
        "manual_png": output_root / "manual_bscan.png",
        "auto_png": output_root / "auto_bscan.png",
        "side_by_side_png": output_root / "side_by_side.png",
        "params_csv": output_root / "params_table.csv",
        "metrics_csv": output_root / "metrics_table.csv",
        "report_md": output_root / "comparison_report.md",
        "evidence_zip": output_root / "evidence_bundle.zip",
    }

    manual_arr = np.asarray(result.manual.result, dtype=np.float32)
    auto_arr = np.asarray(result.automatic.result, dtype=np.float32)
    display_spec = _locked_display_spec(
        manual_arr,
        auto_arr,
        result.display_spec,
        cmap=cmap,
    )

    _save_single_bscan(
        manual_arr,
        paths["manual_png"],
        title="Manual baseline",
        display_spec=display_spec,
    )
    _save_single_bscan(
        auto_arr,
        paths["auto_png"],
        title="Auto-tuned",
        display_spec=display_spec,
    )
    _save_side_by_side(
        manual_arr,
        auto_arr,
        paths["side_by_side_png"],
        display_spec=display_spec,
    )

    summary = to_summary_dict(result)
    summary["input_ref"] = input_ref
    summary["notes"] = list(notes or [])
    summary["display_spec"] = {
        **dict(summary.get("display_spec") or {}),
        **display_spec,
    }
    summary["exported_at"] = datetime.now().isoformat(timespec="seconds")
    summary["artifacts"] = {
        key: str(path.resolve())
        for key, path in paths.items()
        if key not in {"evidence_manifest_json", "evidence_zip"}
    }

    _write_csv_rows(
        paths["params_csv"],
        _build_params_rows(summary),
        fieldnames=[
            "candidate",
            "source",
            "method_key",
            "stage_index",
            "param_name",
            "param_value",
        ],
    )
    _write_csv_rows(
        paths["metrics_csv"],
        _build_metrics_rows(summary),
        fieldnames=[
            "metric",
            "manual_value",
            "auto_value",
            "delta",
        ],
    )
    trial_rows, trial_payload, trial_warnings = _build_trial_table(summary)
    _write_csv_rows(
        paths["trial_table_csv"],
        trial_rows,
        fieldnames=[
            "branch",
            "method_key",
            "trial_index",
            "selected",
            "params_json",
            "score",
            "comparison_score",
            "truth_score",
            "truth_target_energy_preservation",
            "truth_target_saliency_gain",
            "truth_background_energy_reduction",
            "truth_false_positive_ratio",
            "candidate_space_hash",
            "candidate_space_profile_id",
            "candidate_space_config_version",
            "candidate_space_recipe_ids_json",
            "candidate_id",
            "candidate_source",
            "candidate_group",
            "candidate_parameters_json",
            "scoring_boundary",
            "manual_review_required",
            "score_version",
            "reason",
            "warnings_json",
        ],
    )
    paths["trial_table_json"].write_text(
        json.dumps(_json_safe(trial_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary.setdefault("evidence_warnings", [])
    summary["evidence_warnings"].extend(trial_warnings)
    paths["report_md"].write_text(
        _build_report_markdown(summary),
        encoding="utf-8",
    )
    ground_truth = getattr(result, "ground_truth", None)
    if isinstance(ground_truth, dict):
        paths["converted_ground_truth_json"].write_text(
            json.dumps(_json_safe(ground_truth), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raw_sidecar = ground_truth.get("raw_sidecar")
        if isinstance(raw_sidecar, dict):
            paths["raw_ground_truth_json"].write_text(
                json.dumps(_json_safe(raw_sidecar), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    paths["truth_metrics_json"].write_text(
        json.dumps(_build_truth_metrics(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["workflow_params_json"].write_text(
        json.dumps(_build_workflow_params(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["summary_json"].write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["evidence_zip"].touch()
    manifest = _build_evidence_manifest(
        summary,
        paths,
        output_root=output_root,
        input_ref=input_ref,
        notes=notes or [],
    )
    paths["evidence_manifest_json"].write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_evidence_zip(paths["evidence_zip"], paths, output_root)

    return _json_safe(
        {
            "bundle_name": safe_name,
            "output_dir": str(output_root.resolve()),
            "artifacts": {
                key: str(path.resolve()) for key, path in paths.items() if path.exists()
            },
            "summary": summary,
        }
    )

__all__ = ["export_auto_tune_comparison_artifacts", "_locked_display_spec"]
