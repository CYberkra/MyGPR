#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a minimal gprMax -> AutoTune -> Evidence smoke workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.auto_tune_comparison import run_auto_tune_comparison, to_summary_dict
from core.auto_tune_comparison_export import export_auto_tune_comparison_artifacts
from core.gprmax_dataset_contract import load_gprmax_dataset_contract


TRUTH_METRIC_KEYS = (
    "truth_score",
    "truth_target_energy_preservation",
    "truth_target_saliency_gain",
    "truth_background_energy_reduction",
    "truth_false_positive_ratio",
)


def run_smoke(
    dataset: str | Path,
    output: str | Path,
    *,
    bundle_name: str = "gprmax_autotune_evidence_smoke",
    search_mode: str = "fast",
    pipeline: list[str] | None = None,
) -> dict[str, Any]:
    """Run the smoke workflow and return a compact JSON-safe summary."""
    package = load_gprmax_dataset_contract(dataset)
    if package.data.size == 0:
        raise ValueError("gprMax dataset contains empty B-scan data")
    ground_truth = package.header_info.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError("gprMax dataset did not attach header_info['ground_truth']")

    pipeline_order = list(pipeline or ["dewow"])
    manual_params = _default_manual_params(pipeline_order)
    roi_spec = _roi_spec_from_ground_truth(ground_truth)
    comparison = run_auto_tune_comparison(
        package.data,
        header_info=package.header_info,
        trace_metadata=package.trace_metadata,
        pipeline=pipeline_order,
        manual_params_by_method=manual_params,
        roi_spec=roi_spec,
        search_mode=search_mode,
    )
    summary = to_summary_dict(comparison)
    bundle = export_auto_tune_comparison_artifacts(
        comparison,
        out_dir=output,
        bundle_name=bundle_name,
        input_ref=str(package.primary_out_file),
        notes=[
            "Minimal gprMax AutoTune Evidence smoke.",
            "Ground truth is used for validation and Evidence, not search guidance.",
        ],
    )

    artifacts = bundle.get("artifacts") or {}
    manual_metrics = ((summary.get("manual") or {}).get("metrics") or {})
    auto_metrics = ((summary.get("automatic") or {}).get("metrics") or {})
    result = {
        "scenario_id": package.scenario_id,
        "dataset": str(package.dataset_dir),
        "manifest": str(package.manifest_path),
        "evidence_output_dir": bundle.get("output_dir"),
        "evidence_bundle_zip": artifacts.get("evidence_zip"),
        "comparison_report_md": artifacts.get("report_md"),
        "comparison_summary_json": artifacts.get("summary_json"),
        "manual": _selected_metrics(manual_metrics),
        "automatic": _selected_metrics(auto_metrics),
    }
    return result


def _default_manual_params(pipeline: list[str]) -> dict[str, dict[str, Any]]:
    defaults = {
        "dewow": {"window": 3},
        "subtracting_average_2D": {"ntraces": 3},
        "agcGain": {"window": 5},
    }
    return {method: dict(defaults.get(method, {})) for method in pipeline}


def _roi_spec_from_ground_truth(ground_truth: dict[str, Any]) -> dict[str, Any]:
    analysis_roi = ground_truth.get("analysis_roi")
    if isinstance(analysis_roi, dict):
        return {
            "mode": "manual",
            "bounds": dict(analysis_roi),
            "label": f"ground_truth:{ground_truth.get('scenario_id', 'analysis_roi')}",
        }
    return {"mode": "full", "label": "full"}


def _selected_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    selected = {
        "comparison_score": metrics.get("comparison_score"),
    }
    for key in TRUTH_METRIC_KEYS:
        selected[key] = metrics.get(key)
    return selected


def _print_report(result: dict[str, Any]) -> None:
    manual = result.get("manual") or {}
    auto = result.get("automatic") or {}
    print(f"Scenario: {result.get('scenario_id')}")
    print(f"Evidence output: {result.get('evidence_output_dir')}")
    print(f"Evidence bundle: {result.get('evidence_bundle_zip')}")
    print(f"Report: {result.get('comparison_report_md')}")
    print(f"truth_score manual: {manual.get('truth_score')}")
    print(f"truth_score AutoTune: {auto.get('truth_score')}")
    print(f"comparison_score manual: {manual.get('comparison_score')}")
    print(f"comparison_score AutoTune: {auto.get('comparison_score')}")
    print("JSON:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal gprMax AutoTune Evidence smoke workflow.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a gprMax dataset directory or manifest JSON.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the Evidence smoke bundle will be written.",
    )
    parser.add_argument(
        "--bundle-name",
        default="gprmax_autotune_evidence_smoke",
        help="Evidence bundle folder name.",
    )
    parser.add_argument(
        "--search-mode",
        default="fast",
        choices=["fast", "standard", "thorough"],
        help="AutoTune search mode.",
    )
    parser.add_argument(
        "--pipeline",
        default="dewow",
        help="Comma-separated processing pipeline, default: dewow.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pipeline = [item.strip() for item in str(args.pipeline).split(",") if item.strip()]
    result = run_smoke(
        args.dataset,
        args.output,
        bundle_name=args.bundle_name,
        search_mode=args.search_mode,
        pipeline=pipeline,
    )
    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
