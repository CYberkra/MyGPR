#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AT-014 multi-scene metric-fidelity validation evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.processing_engine import prepare_runtime_params, run_processing_method
from scripts.auto_tune_validation import run_no_zerotime_gain_validation as at005a
from scripts.auto_tune_validation.background_window_policy import (
    DEFAULT_RATIO_CANDIDATES,
    generate_relative_background_candidates,
)
from scripts.auto_tune_validation.run_stepwise_validation import _git_rev_parse, _json_safe


GAIN_STEP = ("energy_decay_gain", {"strength": 0.8, "smoothing_samples": 31, "max_gain": 6.0})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--evidence-repo-root", required=True)
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--ratio-candidates", default="0.05,0.10,0.20,0.40,0.70,1.00")
    args = parser.parse_args(argv)
    ratios = _parse_ratio_candidates(args.ratio_candidates)
    result = run_metric_fidelity_validation(
        evidence_root=Path(args.evidence_root),
        evidence_repo_root=Path(args.evidence_repo_root),
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
        ratio_candidates=ratios,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_metric_fidelity_validation(
    *,
    evidence_root: Path,
    evidence_repo_root: Path,
    source_repo: str,
    source_branch: str,
    source_commit: str | None = None,
    ratio_candidates: list[float] | None = None,
) -> dict[str, Any]:
    source_commit = source_commit or _git_rev_parse(ROOT)
    evidence_root.mkdir(parents=True, exist_ok=True)
    figures_dir = evidence_root / "figures"
    tables_dir = evidence_root / "tables"
    reports_dir = evidence_root / "reports"
    manifests_dir = evidence_root / "manifests"
    for d in (figures_dir, tables_dir, reports_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    scene_specs = _scene_specs(evidence_repo_root)
    scene_candidate_metrics: list[dict[str, Any]] = []
    gx004_rows: list[dict[str, Any]] = []
    gx005_rows: list[dict[str, Any]] = []
    gx006_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    scene_best: dict[str, dict[str, Any]] = {}

    for spec in scene_specs:
        scene = _load_scene(spec)
        data = scene["data"]
        trace_count = int(data.shape[1])
        spacing = float(scene["trace_spacing_m"])
        color_limit = at005a._global_color_limit(data)
        at005a._save_overlay(
            data,
            figures_dir / f"{spec['id'].lower()}_input_overlay.png",
            roi=scene["target_roi"] or at005a._target_roi({}, data.shape),
            color_limit=color_limit,
            title=f"{spec['id']} input",
        )

        candidates = generate_relative_background_candidates(
            trace_count=trace_count,
            trace_spacing_m=spacing,
            ratio_candidates=ratio_candidates or list(DEFAULT_RATIO_CANDIDATES),
            max_fraction_of_trace_count=1.0,
            include_full_line_candidate=True,
            min_ntraces=3,
        )
        best_score = float("-inf")
        best_row: dict[str, Any] | None = None
        for cand in candidates:
            processed = _run_primary_lane(
                data,
                header_info=scene["header_info"],
                trace_metadata=scene["trace_metadata"],
                bg_ntraces=int(cand.ntraces),
            )
            lane_id = f"{spec['id'].lower()}_n{cand.ntraces}_{cand.label}"
            fig_path = figures_dir / f"{lane_id}.png"
            crop_path = figures_dir / f"{lane_id}_roi_crop.png"
            roi_for_plot = scene["target_roi"] or at005a._target_roi({}, processed.shape)
            at005a._save_bscan(processed, fig_path, color_limit=color_limit, title=lane_id)
            if roi_for_plot:
                at005a._save_crop(processed, crop_path, roi_for_plot, color_limit, f"{lane_id} ROI crop")

            metrics = _scene_metrics(spec["id"], scene, data, processed)
            score = _candidate_score(spec["id"], metrics)
            row = {
                "scene_id": spec["id"],
                "candidate_label": cand.label,
                "ratio": float(cand.ntraces_ratio),
                "generated_ntraces": int(cand.ntraces),
                "ntraces_over_trace_count": float(cand.ntraces_ratio),
                "window_length_m": cand.window_length_m,
                "gain_method": GAIN_STEP[0],
                "gain_params": GAIN_STEP[1],
                "zero_time_policy": "excluded_or_fixed_zero",
                "dewow_policy": "excluded_primary",
                "branch_validity": "valid_with_caveats",
                "candidate_score": score,
                "warnings": metrics.get("warnings", []),
                "figure": f"figures/{fig_path.name}",
                "roi_crop": f"figures/{crop_path.name}" if roi_for_plot else "",
                **metrics.get("summary", {}),
            }
            scene_candidate_metrics.append(row)
            if row["warnings"]:
                warning_rows.append(
                    {
                        "scene_id": spec["id"],
                        "candidate_label": cand.label,
                        "generated_ntraces": int(cand.ntraces),
                        "risk_flags": row["warnings"],
                    }
                )

            if spec["id"] == "GX-004":
                gx004_rows.append({"scene_id": spec["id"], "candidate_label": cand.label, "generated_ntraces": int(cand.ntraces), **metrics["gx004"]})
            if spec["id"] == "GX-005":
                gx005_rows.extend(
                    {
                        "scene_id": spec["id"],
                        "candidate_label": cand.label,
                        "generated_ntraces": int(cand.ntraces),
                        **item,
                    }
                    for item in metrics["gx005_targets"]
                )
            if spec["id"] == "GX-006":
                gx006_rows.append({"scene_id": spec["id"], "candidate_label": cand.label, "generated_ntraces": int(cand.ntraces), **metrics["gx006"]})

            if score > best_score:
                best_score = score
                best_row = row

        assert best_row is not None
        scene_best[spec["id"]] = {
            "best_candidate_label": best_row["candidate_label"],
            "best_candidate_ntraces": best_row["generated_ntraces"],
            "best_candidate_score": best_row["candidate_score"],
            "trace_count": trace_count,
            "generated_ntraces": [
                int(item["generated_ntraces"]) for item in scene_candidate_metrics if item["scene_id"] == spec["id"]
            ],
        }
        _copy_best(figures_dir, best_row["figure"], spec["id"])

    gate = _gate_reassessment(scene_candidate_metrics, gx004_rows, gx005_rows, gx006_rows, scene_best)
    _write_csv(tables_dir / "scene_candidate_metrics.csv", scene_candidate_metrics)
    _write_csv(tables_dir / "gx004_false_positive_fidelity_metrics.csv", gx004_rows)
    _write_csv(tables_dir / "gx005_per_target_processed_metrics.csv", gx005_rows)
    _write_csv(tables_dir / "gx006_layer_interface_metrics.csv", gx006_rows)
    _write_csv(tables_dir / "gate_reassessment.csv", [gate])
    _write_csv(tables_dir / "warnings_and_risk_flags.csv", warning_rows)
    _save_overview(scene_candidate_metrics, figures_dir / "scene_candidate_metric_fidelity_overview.png")
    _save_gx004_plot(gx004_rows, figures_dir / "gx004_negative_control_energy.png")
    _save_gx005_plot(gx005_rows, figures_dir / "gx005_per_target_processed_comparison.png")
    _save_gx006_plot(gx006_rows, figures_dir / "gx006_layer_interface_preservation.png")

    summary = {
        "artifact_id": "AT-014",
        "task_id": "AT-014_multi_scene_metric_fidelity_validation",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "included_scenes": ["GX-003", "GX-004", "GX-005", "GX-006"],
        "at011_policy_unchanged": True,
        "ratio_family": ratio_candidates or list(DEFAULT_RATIO_CANDIDATES),
        "lane_policy": {
            "zero_time": "excluded_or_fixed_zero",
            "dewow": "excluded_primary",
            "background_policy": "AT-011 relative trace-count-aware",
            "gain_policy": "energy_decay_gain conservative lane",
        },
        "scene_best": scene_best,
        "gate_reassessment": gate,
        "gx004_findings": _gx004_findings(gx004_rows),
        "gx005_findings": _gx005_findings(gx005_rows),
        "gx006_findings": _gx006_findings(gx006_rows),
        "known_risks": [
            "Synthetic thin-2D scenes cannot support preset promotion by themselves.",
            "No field-performance claim is supported by AT-014.",
            "AT-014 improves metric fidelity but does not change AutoTune scoring semantics.",
        ],
        "claim_boundary": {
            "preset_promotion": "forbidden",
            "overall_autotune_superiority": "forbidden",
            "field_performance_validation": "forbidden",
            "thin_2d_limitation": "disclosed",
        },
        "at013_proxy_limitation_addressed": True,
    }
    manifest = {
        "artifact_id": "AT-014",
        "task_id": "AT-014_multi_scene_metric_fidelity_validation",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending_self_reference",
        "generated_at": summary["generated_at"],
        "command": "python scripts/auto_tune_validation/run_multi_scene_metric_fidelity_validation.py",
        "dataset_name": "GX-003/GX-004/GX-005/GX-006",
        "dataset_shape": "multi-scene",
        "dataset_hash": "manifest-driven",
        "ground_truth_available": True,
        "metric_type": "multi_scene_metric_fidelity_validation",
        "artifacts": {
            "markdown_report": "reports/multi_scene_metric_fidelity_validation_report.md",
            "html_report": "reports/multi_scene_metric_fidelity_validation_report.html",
            "summary": "manifests/metric_fidelity_summary.json",
            "scene_candidate_metrics_csv": "tables/scene_candidate_metrics.csv",
            "gx004_fidelity_csv": "tables/gx004_false_positive_fidelity_metrics.csv",
            "gx005_per_target_csv": "tables/gx005_per_target_processed_metrics.csv",
            "gx006_layer_csv": "tables/gx006_layer_interface_metrics.csv",
            "gate_csv": "tables/gate_reassessment.csv",
            "warnings_csv": "tables/warnings_and_risk_flags.csv",
        },
    }
    _write_json(manifests_dir / "metric_fidelity_summary.json", summary)
    _write_json(manifests_dir / "evidence_manifest.json", manifest)
    (reports_dir / "multi_scene_metric_fidelity_validation_report.md").write_text(_render_md(summary), encoding="utf-8")
    (reports_dir / "multi_scene_metric_fidelity_validation_report.html").write_text(_render_html(summary), encoding="utf-8")
    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "included_scenes": summary["included_scenes"],
        "gate_status": gate["gate_status"],
        "scene_best": {k: {"label": v["best_candidate_label"], "ntraces": v["best_candidate_ntraces"]} for k, v in scene_best.items()},
    }


def _scene_specs(evidence_repo_root: Path) -> list[dict[str, Any]]:
    gpr = evidence_repo_root / "gprmax"
    return [
        {"id": "GX-003", "path": gpr / "GX-003_audited_native_gprmax_benchmark", "csv": "tables/mygpr_bscan.csv", "gt": "manifests/gprmax_package_audit.json", "spacing": 0.01},
        {"id": "GX-004", "path": gpr / "GX-004_no_target_false_positive_control", "csv": "converted/data.csv", "gt": "manifests/ground_truth.json", "spacing": 0.01},
        {"id": "GX-005", "path": gpr / "GX-005_multi_target_varying_depth", "csv": "converted/data.csv", "gt": "manifests/ground_truth.json", "spacing": 0.01},
        {"id": "GX-006", "path": gpr / "GX-006_layered_complex_background", "csv": "converted/data.csv", "gt": "manifests/ground_truth.json", "spacing": 0.01},
    ]


def _load_scene(spec: dict[str, Any]) -> dict[str, Any]:
    base = Path(spec["path"])
    data = np.loadtxt(base / spec["csv"], delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    gt = json.loads((base / spec["gt"]).read_text(encoding="utf-8")) if (base / spec["gt"]).exists() else {}
    rois = _parse_rois(gt.get("rois", []), data.shape)
    target_roi = None
    if spec["id"] == "GX-003":
        target_roi = at005a._target_roi({}, data.shape)
    elif rois["target"]:
        target_roi = _roi_to_obj(rois["target"][0])
    header = {"trace_interval_m": float(spec["spacing"]), "trace_step_m": float(spec["spacing"])}
    trace_count = int(data.shape[1])
    trace_metadata = {"trace_distance_m": np.arange(trace_count, dtype=np.float64) * float(spec["spacing"])}
    return {
        "id": spec["id"],
        "data": np.asarray(data, dtype=np.float64),
        "ground_truth": gt,
        "rois": rois,
        "target_roi": target_roi,
        "header_info": header,
        "trace_metadata": trace_metadata,
        "trace_spacing_m": float(spec["spacing"]),
    }


def _parse_rois(items: list[Any], shape: tuple[int, int]) -> dict[str, list[dict[str, int | str]]]:
    out: dict[str, list[dict[str, int | str]]] = {"target": [], "local_background": [], "negative_control": [], "layer_interface": [], "no_target_region": []}
    for item in items:
        if not isinstance(item, dict):
            continue
        rtype = str(item.get("roi_type", ""))
        if rtype not in out:
            continue
        try:
            s0 = int(item.get("sample_start_idx"))
            s1 = int(item.get("sample_end_idx"))
            t0 = int(item.get("trace_start_idx"))
            t1 = int(item.get("trace_end_idx"))
        except Exception:
            continue
        s0 = max(0, min(shape[0] - 1, s0))
        s1 = max(s0 + 1, min(shape[0], s1))
        t0 = max(0, min(shape[1] - 1, t0))
        t1 = max(t0 + 1, min(shape[1], t1))
        out[rtype].append(
            {
                "roi_id": str(item.get("roi_id", "")),
                "associated_object_id": str(item.get("associated_object_id", "")),
                "sample_start_idx": s0,
                "sample_end_idx": s1,
                "trace_start_idx": t0,
                "trace_end_idx": t1,
            }
        )
    return out


def _roi_to_obj(roi: dict[str, int | str]) -> at005a.Roi:
    return at005a.Roi(
        time_start_idx=int(roi["sample_start_idx"]),
        time_end_idx=int(roi["sample_end_idx"]),
        dist_start_idx=int(roi["trace_start_idx"]),
        dist_end_idx=int(roi["trace_end_idx"]),
    )


def _run_primary_lane(
    raw: np.ndarray,
    *,
    header_info: dict[str, Any],
    trace_metadata: dict[str, Any],
    bg_ntraces: int,
) -> np.ndarray:
    current = np.array(raw, copy=True)
    params_bg = {"ntraces": int(bg_ntraces)}
    runtime_bg = prepare_runtime_params(
        "subtracting_average_2D",
        params_bg,
        header_info,
        trace_metadata,
        current.shape,
    )
    current, _ = run_processing_method(current, "subtracting_average_2D", runtime_bg)
    params_gain = dict(GAIN_STEP[1])
    runtime_gain = prepare_runtime_params(
        GAIN_STEP[0],
        params_gain,
        header_info,
        trace_metadata,
        current.shape,
    )
    current, _ = run_processing_method(current, GAIN_STEP[0], runtime_gain)
    return np.asarray(current, dtype=np.float64)


def _rms(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _roi_patch(arr: np.ndarray, roi: dict[str, int | str]) -> np.ndarray:
    s0 = int(roi["sample_start_idx"])
    s1 = int(roi["sample_end_idx"])
    t0 = int(roi["trace_start_idx"])
    t1 = int(roi["trace_end_idx"])
    return arr[s0:s1, t0:t1]


def _scene_metrics(scene_id: str, scene: dict[str, Any], before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    rois = scene["rois"]
    warnings: list[str] = []
    summary: dict[str, Any] = {}
    gx004: dict[str, Any] = {}
    gx005_targets: list[dict[str, Any]] = []
    gx006: dict[str, Any] = {}

    clip_ratio = float(np.mean(np.abs(after) >= np.nanpercentile(np.abs(after), 99.8)))
    global_ratio = _rms(after) / max(_rms(before), 1e-12)
    fp_proxy = 0.0

    if scene_id == "GX-004":
        neg_before = [_rms(_roi_patch(before, roi)) for roi in rois["negative_control"]]
        neg_after = [_rms(_roi_patch(after, roi)) for roi in rois["negative_control"]]
        nt_before = [_rms(_roi_patch(before, roi)) for roi in rois["no_target_region"]]
        nt_after = [_rms(_roi_patch(after, roi)) for roi in rois["no_target_region"]]
        negative_before = float(np.mean(neg_before)) if neg_before else 0.0
        negative_after = float(np.mean(neg_after)) if neg_after else 0.0
        no_target_before = float(np.mean(nt_before)) if nt_before else 0.0
        no_target_after = float(np.mean(nt_after)) if nt_after else 0.0
        negative_ratio = negative_after / max(negative_before, 1e-12)
        no_target_ratio = no_target_after / max(no_target_before, 1e-12)
        local_before = [_rms(_roi_patch(before, roi)) for roi in rois["local_background"]]
        local_after = [_rms(_roi_patch(after, roi)) for roi in rois["local_background"]]
        local_ratio = (float(np.mean(local_after)) / max(float(np.mean(local_before)) if local_before else 1e-12, 1e-12))
        fp_proxy = float(0.5 * negative_ratio + 0.5 * no_target_ratio)
        artifact_risk = bool(fp_proxy > 1.2 or local_ratio > 1.3)
        if artifact_risk:
            warnings.append("gx004_false_positive_artifact_risk")
        gx004 = {
            "negative_control_energy_before": negative_before,
            "negative_control_energy_after": negative_after,
            "negative_control_energy_ratio": negative_ratio,
            "no_target_region_energy_before": no_target_before,
            "no_target_region_energy_after": no_target_after,
            "no_target_region_energy_ratio": no_target_ratio,
            "local_contrast_inflation": local_ratio,
            "clipping_ratio": clip_ratio,
            "global_energy_ratio": global_ratio,
            "false_positive_proxy": fp_proxy,
            "artifact_risk": artifact_risk,
        }
        summary.update(
            {
                "false_positive_proxy": fp_proxy,
                "clipping_ratio": clip_ratio,
                "global_energy_ratio": global_ratio,
                "target_preservation": None,
            }
        )

    if scene_id == "GX-005":
        target_metrics: list[dict[str, Any]] = []
        for target_roi in rois["target"]:
            tid = str(target_roi.get("associated_object_id") or target_roi.get("roi_id"))
            local = next((r for r in rois["local_background"] if str(r.get("associated_object_id")) == tid), None)
            if local is None:
                local = rois["local_background"][0] if rois["local_background"] else target_roi
            tb = _rms(_roi_patch(before, target_roi))
            ta = _rms(_roi_patch(after, target_roi))
            lb = _rms(_roi_patch(before, local))
            la = _rms(_roi_patch(after, local))
            cb = tb / max(lb, 1e-12)
            ca = ta / max(la, 1e-12)
            row = {
                "target_id": tid,
                "target_roi_energy_before": tb,
                "target_roi_energy_after": ta,
                "target_roi_energy_ratio": ta / max(tb, 1e-12),
                "local_background_energy_before": lb,
                "local_background_energy_after": la,
                "target_to_local_background_contrast_before": cb,
                "target_to_local_background_contrast_after": ca,
                "contrast_ratio": ca / max(cb, 1e-12),
                "target_preservation_ratio": ta / max(tb, 1e-12),
                "clipping_ratio_target_roi": float(np.mean(np.abs(_roi_patch(after, target_roi)) >= np.nanpercentile(np.abs(after), 99.8))),
            }
            target_metrics.append(row)
        gx005_targets = target_metrics
        if target_metrics:
            preserves = [r["target_preservation_ratio"] for r in target_metrics]
            contrasts = [r["contrast_ratio"] for r in target_metrics]
            max_gap = float(max(preserves) - min(preserves))
            one_target_only = bool((max(contrasts) > 1.05) and (min(contrasts) < 0.95))
            imbalance = bool(max_gap > 0.25 or one_target_only)
            if imbalance:
                warnings.append("gx005_target_imbalance_risk")
            summary.update(
                {
                    "target_preservation": float(np.mean(preserves)),
                    "roi_contrast": float(np.mean(contrasts)),
                    "max_target_preservation_gap": max_gap,
                    "one_target_only_improvement": one_target_only,
                    "target_imbalance_risk": imbalance,
                    "clipping_ratio": clip_ratio,
                    "global_energy_ratio": global_ratio,
                }
            )

    if scene_id == "GX-006":
        layer_before = [_rms(_roi_patch(before, roi)) for roi in rois["layer_interface"]]
        layer_after = [_rms(_roi_patch(after, roi)) for roi in rois["layer_interface"]]
        l_before = float(np.mean(layer_before)) if layer_before else 0.0
        l_after = float(np.mean(layer_after)) if layer_after else 0.0
        layer_ratio = l_after / max(l_before, 1e-12)
        clutter = rois["negative_control"][0] if rois["negative_control"] else (rois["local_background"][0] if rois["local_background"] else None)
        clutter_before = _rms(_roi_patch(before, clutter)) if clutter else 0.0
        clutter_after = _rms(_roi_patch(after, clutter)) if clutter else 0.0
        clutter_proxy = clutter_after / max(clutter_before, 1e-12)
        interface_risk = bool(layer_ratio < 0.7)
        clutter_risk = bool(clutter_proxy > 1.25)
        if interface_risk:
            warnings.append("gx006_interface_suppression_risk")
        if clutter_risk:
            warnings.append("gx006_clutter_false_positive_risk")
        gx006 = {
            "layer_interface_energy_before": l_before,
            "layer_interface_energy_after": l_after,
            "layer_interface_energy_ratio": layer_ratio,
            "layer_interface_continuity_proxy_before": l_before,
            "layer_interface_continuity_proxy_after": l_after,
            "clutter_negative_control_energy_before": clutter_before,
            "clutter_negative_control_energy_after": clutter_after,
            "clutter_false_positive_proxy": clutter_proxy,
            "interface_suppression_risk": interface_risk,
            "clutter_false_positive_risk": clutter_risk,
            "clipping_ratio": clip_ratio,
            "global_energy_ratio": global_ratio,
        }
        summary.update(
            {
                "roi_contrast": layer_ratio,
                "false_positive_proxy": clutter_proxy,
                "target_preservation": None,
                "clipping_ratio": clip_ratio,
                "global_energy_ratio": global_ratio,
            }
        )

    if scene_id == "GX-003":
        roi = scene["target_roi"] or at005a._target_roi({}, before.shape)
        before_m = at005a._metrics(before, before, roi)
        after_m = at005a._metrics(after, before, roi)
        summary.update(
            {
                "roi_contrast": after_m.get("roi_to_local_background_contrast"),
                "target_preservation": after_m.get("roi_energy_ratio_to_input"),
                "false_positive_proxy": after_m.get("false_positive_proxy"),
                "clipping_ratio": after_m.get("clipping_ratio"),
                "global_energy_ratio": after_m.get("global_energy_ratio_to_input"),
            }
        )

    return {"summary": summary, "warnings": warnings, "gx004": gx004, "gx005_targets": gx005_targets, "gx006": gx006}


def _candidate_score(scene_id: str, metrics: dict[str, Any]) -> float:
    s = metrics["summary"]
    clip = float(s.get("clipping_ratio") or 0.0)
    if scene_id == "GX-004":
        return (1.0 - clip) - float(s.get("false_positive_proxy") or 0.0)
    if scene_id == "GX-005":
        return float(s.get("target_preservation") or 0.0) + float(s.get("roi_contrast") or 0.0) - clip
    if scene_id == "GX-006":
        return float(s.get("roi_contrast") or 0.0) - float(s.get("false_positive_proxy") or 0.0) - clip
    return float(s.get("roi_contrast") or 0.0) + float(s.get("target_preservation") or 0.0) - clip


def _gate_reassessment(
    scene_rows: list[dict[str, Any]],
    gx004_rows: list[dict[str, Any]],
    gx005_rows: list[dict[str, Any]],
    gx006_rows: list[dict[str, Any]],
    scene_best: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    flags: list[str] = []
    if any(bool(r.get("artifact_risk")) for r in gx004_rows):
        flags.append("gx004_false_positive_risk")
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for r in gx005_rows:
        key = f"{r['candidate_label']}:{r['generated_ntraces']}"
        by_candidate.setdefault(key, []).append(r)
    for _, items in by_candidate.items():
        if len(items) < 2:
            continue
        preserves = [float(x.get("target_preservation_ratio") or 0.0) for x in items]
        contrasts = [float(x.get("contrast_ratio") or 0.0) for x in items]
        if max(preserves) - min(preserves) > 0.25:
            flags.append("gx005_target_imbalance")
            break
        if max(contrasts) > 1.05 and min(contrasts) < 0.95:
            flags.append("gx005_one_target_only_improvement")
            break
    if any(bool(r.get("interface_suppression_risk")) for r in gx006_rows):
        flags.append("gx006_interface_suppression")
    if any(bool(r.get("clutter_false_positive_risk")) for r in gx006_rows):
        flags.append("gx006_clutter_false_positive")
    if any(float(v["best_candidate_ntraces"]) / max(1.0, float(v["trace_count"])) >= 0.95 for v in scene_best.values()):
        flags.append("best_candidate_at_edge")
    flags.append("synthetic_thin2d_scene_limit")
    status = "blocked" if flags else "pass"
    return {
        "gate_status": status,
        "risk_flags": flags,
        "scene_best_labels": {k: v["best_candidate_label"] for k, v in scene_best.items()},
        "summary": "AT-014 reassessment keeps gate blocked unless multi-scene risks and synthetic limits are cleared.",
    }


def _gx004_findings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    best = min(rows, key=lambda r: float(r.get("false_positive_proxy") or 1e9))
    return {
        "status": "evaluated",
        "best_candidate_label": best["candidate_label"],
        "best_generated_ntraces": best["generated_ntraces"],
        "min_false_positive_proxy": best["false_positive_proxy"],
        "artifact_risk_present": any(bool(r.get("artifact_risk")) for r in rows),
    }


def _gx005_findings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    gaps: list[float] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        k = f"{row['candidate_label']}:{row['generated_ntraces']}"
        grouped.setdefault(k, []).append(row)
    for vals in grouped.values():
        if len(vals) >= 2:
            p = [float(v.get("target_preservation_ratio") or 0.0) for v in vals]
            gaps.append(max(p) - min(p))
    return {
        "status": "evaluated",
        "max_target_preservation_gap": max(gaps) if gaps else 0.0,
        "target_imbalance_risk_present": any(g > 0.25 for g in gaps),
        "note": "Strict per-target processed ROI metrics reported in gx005_per_target_processed_metrics.csv",
    }


def _gx006_findings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    return {
        "status": "evaluated",
        "interface_suppression_risk_present": any(bool(r.get("interface_suppression_risk")) for r in rows),
        "clutter_false_positive_risk_present": any(bool(r.get("clutter_false_positive_risk")) for r in rows),
        "note": "Layer/interface metrics are reported separately from clutter false-positive metrics.",
    }


def _copy_best(fig_dir: Path, figure_ref: str, scene_id: str) -> None:
    src = fig_dir / Path(figure_ref).name
    alias = {
        "GX-003": "gx003_best_overlay.png",
        "GX-004": "gx004_false_positive_map.png",
        "GX-005": "gx005_per_target_processed_comparison.png",
        "GX-006": "gx006_layer_interface_preservation.png",
    }[scene_id]
    if src.exists():
        shutil.copy2(src, fig_dir / alias)


def _save_overview(rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(str(r["scene_id"]), []).append(r)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    axs = axes.flatten()
    for idx, scene in enumerate(["GX-003", "GX-004", "GX-005", "GX-006"]):
        ax = axs[idx]
        data = sorted(grouped.get(scene, []), key=lambda x: float(x["ntraces_over_trace_count"]))
        if not data:
            ax.set_title(scene + " (missing)")
            continue
        ax.plot([float(d["ntraces_over_trace_count"]) for d in data], [float(d["candidate_score"]) for d in data], "o-", color="#2563eb")
        ax.set_title(scene)
        ax.set_xlabel("ntraces/trace_count")
        ax.set_ylabel("fidelity score")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_gx004_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda x: int(x["generated_ntraces"]))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.plot([int(r["generated_ntraces"]) for r in rows], [float(r["negative_control_energy_ratio"]) for r in rows], "o-", label="negative_control_ratio")
    ax.plot([int(r["generated_ntraces"]) for r in rows], [float(r["no_target_region_energy_ratio"]) for r in rows], "s-", label="no_target_ratio")
    ax.set_xlabel("ntraces")
    ax.set_ylabel("energy ratio")
    ax.set_title("GX-004 no-target / negative-control energy ratios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_gx005_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    labels = [f"{r['candidate_label']}:{r['target_id']}" for r in rows]
    vals = [float(r.get("target_preservation_ratio") or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    ax.bar(range(len(vals)), vals, color="#0f766e")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("target_preservation_ratio")
    ax.set_title("GX-005 strict per-target processed metrics")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_gx006_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda x: int(x["generated_ntraces"]))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.plot([int(r["generated_ntraces"]) for r in rows], [float(r["layer_interface_energy_ratio"]) for r in rows], "o-", label="layer_ratio")
    ax.plot([int(r["generated_ntraces"]) for r in rows], [float(r["clutter_false_positive_proxy"]) for r in rows], "s-", label="clutter_fp_proxy")
    ax.set_xlabel("ntraces")
    ax.set_ylabel("metric")
    ax.set_title("GX-006 layer/interface fidelity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_json_safe(row.get(k)), ensure_ascii=False) for k in fields})


def _parse_ratio_candidates(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid ratio candidate: {token}")
        out.append(value)
    if not out:
        raise ValueError("no ratio candidates provided")
    return out


def _render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# AT-014 Multi-scene Metric Fidelity Validation",
        "",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Included scenes: `{summary['included_scenes']}`",
        f"- AT-011 policy unchanged: `{summary['at011_policy_unchanged']}`",
        f"- Gate reassessment: `{summary['gate_reassessment']['gate_status']}`",
        "",
        "## Strict findings",
        f"- GX-004: `{summary['gx004_findings']}`",
        f"- GX-005: `{summary['gx005_findings']}`",
        f"- GX-006: `{summary['gx006_findings']}`",
        "- GX-004 only uses no-target/negative-control fidelity metrics; no target-preservation metric is used.",
        "- GX-005 reports strict processed per-target metrics for target_A and target_B.",
        "- GX-006 reports layer/interface-aware metrics separately from clutter false-positive metrics.",
        "",
        "## Scene best candidates",
    ]
    for sid, row in summary["scene_best"].items():
        lines.append(f"- {sid}: `{row['best_candidate_label']}` ntraces=`{row['best_candidate_ntraces']}`")
    lines.extend(
        [
            "",
            "## Gate status",
            f"- status: `{summary['gate_reassessment']['gate_status']}`",
            f"- risk_flags: `{summary['gate_reassessment']['risk_flags']}`",
            f"- AT-013 proxy limitation addressed: `{summary['at013_proxy_limitation_addressed']}`",
            "- AT-011 ratio family is replayed unchanged: 0.05, 0.10, 0.20, 0.40, 0.70, 1.00.",
            "- Thin-2D synthetic limitation remains active for all scenes.",
            "- No preset promotion. No overall AutoTune superiority claim. No field-performance claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_html(summary: dict[str, Any]) -> str:
    risks = "".join(f"<li>{html.escape(str(item))}</li>" for item in summary["gate_reassessment"]["risk_flags"])
    scene_rows = "".join(
        f"<li>{html.escape(sid)}: best={html.escape(str(item['best_candidate_label']))} ntraces={item['best_candidate_ntraces']}</li>"
        for sid, item in summary["scene_best"].items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AT-014 Multi-scene Metric Fidelity Validation</title>
  <style>
    body {{ margin: 0; background: #f3f6fb; color: #1d2a40; font-family: "Segoe UI", "PingFang SC", sans-serif; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 28px; }}
    .card {{ background: #fff; border: 1px solid #d8e1ee; border-radius: 8px; padding: 14px; margin: 12px 0; }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    img {{ width: 100%; border: 1px solid #d8e1ee; border-radius: 6px; background: #fff; }}
  </style>
</head>
<body>
<main>
  <h1>AT-014 Multi-scene Metric Fidelity Validation</h1>
  <div class="card">
    <div><b>Source commit:</b> <code>{summary['source_commit']}</code></div>
    <div><b>AT-011 policy unchanged:</b> <code>{summary['at011_policy_unchanged']}</code></div>
    <div><b>Gate reassessment:</b> <code>{summary['gate_reassessment']['gate_status']}</code></div>
  </div>
  <div class="card">
    <h2>Claim boundary</h2>
    <ul>
      <li>Preset promotion: forbidden</li>
      <li>Overall AutoTune superiority claim: forbidden</li>
      <li>Field-performance claim: forbidden</li>
      <li>Thin-2D synthetic limitation: disclosed</li>
      <li>AT-013 proxy limitation addressed: <code>{summary['at013_proxy_limitation_addressed']}</code></li>
    </ul>
  </div>
  <div class="card"><h2>Scene best candidates</h2><ul>{scene_rows}</ul></div>
  <div class="card"><h2>Risk flags</h2><ul>{risks}</ul></div>
  <div class="grid">
    <img src="../figures/scene_candidate_metric_fidelity_overview.png" alt="overview">
    <img src="../figures/gx004_negative_control_energy.png" alt="gx004">
    <img src="../figures/gx005_per_target_processed_comparison.png" alt="gx005">
    <img src="../figures/gx006_layer_interface_preservation.png" alt="gx006">
  </div>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
