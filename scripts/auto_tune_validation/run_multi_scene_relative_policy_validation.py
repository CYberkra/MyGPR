#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AT-013 multi-scene relative background-window policy validation evidence."""

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
    result = run_multi_scene_validation(
        evidence_root=Path(args.evidence_root),
        evidence_repo_root=Path(args.evidence_repo_root),
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
        ratio_candidates=ratios,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_multi_scene_validation(
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
    figs = evidence_root / "figures"
    tables = evidence_root / "tables"
    reports = evidence_root / "reports"
    manifests = evidence_root / "manifests"
    for d in (figs, tables, reports, manifests):
        d.mkdir(parents=True, exist_ok=True)

    scenes = _scene_specs(evidence_repo_root)
    scene_candidate_rows: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []
    false_positive_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []

    for spec in scenes:
        bundle = _load_scene_bundle(spec)
        data = bundle["data"]
        trace_count = int(data.shape[1])
        spacing = float(bundle["trace_spacing_m"])
        color_limit = at005a._global_color_limit(data)
        if bundle.get("target_roi") is None:
            bundle["target_roi"] = at005a._target_roi({}, data.shape)
        input_png = figs / f"{spec['id'].lower()}_input_overlay.png"
        at005a._save_overlay(data, input_png, roi=bundle.get("target_roi"), color_limit=color_limit, title=f"{spec['id']} input")

        candidates = generate_relative_background_candidates(
            trace_count=trace_count,
            trace_spacing_m=spacing,
            ratio_candidates=ratio_candidates or list(DEFAULT_RATIO_CANDIDATES),
            max_fraction_of_trace_count=1.0,
            include_full_line_candidate=True,
            min_ntraces=3,
        )
        best_row: dict[str, Any] | None = None
        best_score = float("-inf")
        for c in candidates:
            lane_id = f"{spec['id'].lower()}_n{c.ntraces}_{c.label}"
            lane = at005a._run_lane(
                raw=data,
                header_info=bundle["header_info"],
                trace_metadata=bundle["trace_metadata"],
                roi=bundle.get("target_roi"),
                figures_dir=figs,
                color_limit=color_limit,
                dataset_kind=spec["dataset_kind"],
                lane_id=lane_id,
                branch="manual",
                description=f"{spec['id']} relative ntraces={c.ntraces} ({c.label})",
                pre_gain_steps=[("subtracting_average_2D", {"ntraces": int(c.ntraces)})],
                gain_step=GAIN_STEP,
                auto_tune=False,
            )
            row = lane["row"]
            metrics = row.get("after_gain_metrics", {})
            before = row.get("before_gain_metrics", {})
            scene_row = {
                "scene_id": spec["id"],
                "candidate_label": c.label,
                "ratio": float(c.ntraces_ratio),
                "generated_ntraces": int(c.ntraces),
                "ntraces_over_trace_count": float(c.ntraces_ratio),
                "window_length_m": c.window_length_m,
                "gain_method": row.get("gain_method"),
                "gain_params": row.get("gain_params"),
                "branch_validity": row.get("branch_validity"),
                "candidate_score": _scene_score(spec["id"], row),
                "roi_contrast": metrics.get("roi_to_local_background_contrast"),
                "local_background_energy": metrics.get("local_background_energy"),
                "negative_control_energy": metrics.get("false_positive_proxy"),
                "false_positive_proxy": metrics.get("false_positive_proxy"),
                "clipping_ratio": metrics.get("clipping_ratio"),
                "global_energy_ratio": metrics.get("global_energy_ratio_to_input"),
                "target_preservation": metrics.get("roi_energy_ratio_to_input"),
                "background_energy_reduction": before.get("background_energy_reduction"),
                "warnings": row.get("sanity_warnings"),
                "figure": row.get("figure"),
                "roi_crop": row.get("roi_crop"),
            }
            scene_candidate_rows.append(scene_row)
            if scene_row["warnings"]:
                warning_rows.append(
                    {
                        "scene_id": spec["id"],
                        "candidate_label": c.label,
                        "generated_ntraces": int(c.ntraces),
                        "warnings": scene_row["warnings"],
                    }
                )

            if spec["id"] == "GX-005":
                target_rows = _gx005_target_rows(spec["id"], c.label, c.ntraces, data, row, bundle)
                per_target_rows.extend(target_rows)
            if spec["id"] == "GX-004":
                false_positive_rows.append(
                    {
                        "scene_id": spec["id"],
                        "candidate_label": c.label,
                        "generated_ntraces": int(c.ntraces),
                        "false_positive_proxy": metrics.get("false_positive_proxy"),
                        "clipping_ratio": metrics.get("clipping_ratio"),
                        "global_energy_ratio": metrics.get("global_energy_ratio_to_input"),
                        "artifact_risk": bool((metrics.get("false_positive_proxy") or 0) > 1.25),
                    }
                )

            score = float(scene_row["candidate_score"])
            if score > best_score:
                best_score = score
                best_row = scene_row

        assert best_row is not None
        scene_summaries.append(
            {
                "scene_id": spec["id"],
                "trace_count": trace_count,
                "candidate_count": len(candidates),
                "candidate_labels": sorted({r["candidate_label"] for r in scene_candidate_rows if r["scene_id"] == spec["id"]}),
                "generated_ntraces": [int(r["generated_ntraces"]) for r in scene_candidate_rows if r["scene_id"] == spec["id"]],
                "best_candidate_label": best_row["candidate_label"],
                "best_candidate_ntraces": best_row["generated_ntraces"],
                "best_candidate_score": best_row["candidate_score"],
                "best_candidate_figure": best_row.get("figure"),
            }
        )
        _copy_best_figure(spec["id"], best_row, figs)

    gate = _evaluate_gate(scene_candidate_rows, per_target_rows, false_positive_rows)
    _write_csv(tables / "scene_candidate_metrics.csv", scene_candidate_rows)
    _write_csv(tables / "per_target_metrics.csv", per_target_rows)
    _write_csv(tables / "false_positive_metrics.csv", false_positive_rows)
    _write_csv(tables / "warnings_and_risk_flags.csv", warning_rows)
    _write_csv(tables / "scene_gate_status.csv", [gate])
    _save_scene_candidate_overview(scene_candidate_rows, figs / "scene_candidate_overview.png")
    _save_gx005_target_plot(per_target_rows, figs / "gx005_target_specific_comparison.png")
    _save_gx006_layer_plot(scene_candidate_rows, figs / "gx006_layer_interface_comparison.png")
    _save_gx004_false_positive_plot(false_positive_rows, figs / "gx004_false_positive_map.png")

    summary = {
        "artifact_id": "AT-013",
        "task_id": "AT-013_multi_scene_relative_policy_validation",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "included_scenes": [s["id"] for s in scenes],
        "primary_lane_policy": {
            "zero_time_policy": "excluded_or_fixed_zero",
            "dewow_policy": "excluded_primary",
            "background_policy": "AT-011 relative trace-count-aware policy unchanged",
            "gain_policy": "energy_decay_gain conservative/interpretable lane",
        },
        "scene_summaries": scene_summaries,
        "gate_status": gate,
        "claim_boundary": {
            "overall_autotune_superiority": False,
            "field_performance_claim": False,
            "thin_2d_limitation_disclosed": True,
        },
        "risk_flags": gate.get("risk_flags", []),
    }
    manifest = {
        "artifact_id": "AT-013",
        "task_id": "AT-013_multi_scene_relative_policy_validation",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending_self_reference",
        "generated_at": summary["generated_at"],
        "command": "python scripts/auto_tune_validation/run_multi_scene_relative_policy_validation.py",
        "dataset_name": "GX-003/004/005/006 native benchmark scenes",
        "dataset_shape": "multi-scene",
        "dataset_hash": "multi-scene-manifest-driven",
        "ground_truth_available": True,
        "metric_type": "multi_scene_relative_policy_validation",
        "artifacts": {
            "markdown_report": "reports/multi_scene_relative_policy_validation_report.md",
            "html_report": "reports/multi_scene_relative_policy_validation_report.html",
            "summary": "manifests/multi_scene_validation_summary.json",
            "scene_candidate_metrics_csv": "tables/scene_candidate_metrics.csv",
            "scene_gate_status_csv": "tables/scene_gate_status.csv",
            "per_target_metrics_csv": "tables/per_target_metrics.csv",
            "false_positive_metrics_csv": "tables/false_positive_metrics.csv",
            "warnings_and_risk_flags_csv": "tables/warnings_and_risk_flags.csv",
        },
    }
    _write_json(manifests / "multi_scene_validation_summary.json", summary)
    _write_json(manifests / "evidence_manifest.json", manifest)
    (reports / "multi_scene_relative_policy_validation_report.md").write_text(_render_md(summary), encoding="utf-8")
    (reports / "multi_scene_relative_policy_validation_report.html").write_text(_render_html(summary), encoding="utf-8")
    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "included_scenes": summary["included_scenes"],
        "gate_status": gate["gate_status"],
        "scene_best": {item["scene_id"]: {"label": item["best_candidate_label"], "ntraces": item["best_candidate_ntraces"]} for item in scene_summaries},
    }


def _scene_specs(evidence_repo_root: Path) -> list[dict[str, Any]]:
    gprmax_root = evidence_repo_root / "gprmax"
    return [
        {
            "id": "GX-003",
            "path": gprmax_root / "GX-003_audited_native_gprmax_benchmark",
            "csv": "tables/mygpr_bscan.csv",
            "ground_truth": "manifests/gprmax_package_audit.json",
            "dataset_kind": "gx003_ground_truth",
            "trace_spacing_m": 0.01,
            "target_mode": "single_target",
        },
        {
            "id": "GX-004",
            "path": gprmax_root / "GX-004_no_target_false_positive_control",
            "csv": "converted/data.csv",
            "ground_truth": "manifests/ground_truth.json",
            "dataset_kind": "gx004_no_target",
            "trace_spacing_m": 0.01,
            "target_mode": "no_target",
        },
        {
            "id": "GX-005",
            "path": gprmax_root / "GX-005_multi_target_varying_depth",
            "csv": "converted/data.csv",
            "ground_truth": "manifests/ground_truth.json",
            "dataset_kind": "gx005_multi_target",
            "trace_spacing_m": 0.01,
            "target_mode": "multi_target",
        },
        {
            "id": "GX-006",
            "path": gprmax_root / "GX-006_layered_complex_background",
            "csv": "converted/data.csv",
            "ground_truth": "manifests/ground_truth.json",
            "dataset_kind": "gx006_layered",
            "trace_spacing_m": 0.01,
            "target_mode": "layered",
        },
    ]


def _load_scene_bundle(spec: dict[str, Any]) -> dict[str, Any]:
    base = Path(spec["path"])
    data = np.loadtxt(base / spec["csv"], delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    gt_path = base / spec["ground_truth"]
    gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
    trace_count = int(data.shape[1])
    sample_count = int(data.shape[0])
    header = {
        "trace_interval_m": float(spec["trace_spacing_m"]),
        "trace_step_m": float(spec["trace_spacing_m"]),
        "trace_count": trace_count,
        "sample_count": sample_count,
    }
    trace_distance = np.arange(trace_count, dtype=np.float64) * float(spec["trace_spacing_m"])
    trace_metadata = {"trace_distance_m": trace_distance}
    target_roi = _extract_target_roi(spec["id"], gt, data.shape)
    return {
        "data": np.asarray(data, dtype=np.float64),
        "ground_truth": gt,
        "header_info": header,
        "trace_metadata": trace_metadata,
        "target_roi": target_roi,
        "trace_spacing_m": spec["trace_spacing_m"],
    }


def _extract_target_roi(scene_id: str, ground_truth: dict[str, Any], shape: tuple[int, int]) -> at005a.Roi | None:
    if scene_id == "GX-003":
        # Use the same heuristic center ROI if no direct ROI file in this artifact package.
        return at005a._target_roi({}, shape)
    rois = ground_truth.get("rois")
    if not isinstance(rois, list):
        return None
    target_rows = [r for r in rois if isinstance(r, dict) and r.get("roi_type") == "target"]
    if not target_rows:
        return None
    row = target_rows[0]
    try:
        s0 = int(row["sample_start_idx"])
        s1 = int(row["sample_end_idx"])
        t0 = int(row["trace_start_idx"])
        t1 = int(row["trace_end_idx"])
    except Exception:
        return None
    if s1 <= s0 or t1 <= t0:
        return None
    return at005a.Roi(
        time_start_idx=max(0, s0),
        time_end_idx=min(shape[0], s1),
        dist_start_idx=max(0, t0),
        dist_end_idx=min(shape[1], t1),
    )


def _scene_score(scene_id: str, row: dict[str, Any]) -> float:
    m = row.get("after_gain_metrics", {})
    contrast = float(m.get("roi_to_local_background_contrast") or 0.0)
    preserve = float(m.get("roi_energy_ratio_to_input") or 0.0)
    clip = float(m.get("clipping_ratio") or 0.0)
    fp = float(m.get("false_positive_proxy") or 0.0)
    if scene_id == "GX-004":
        return (1.0 - clip) - fp
    if scene_id == "GX-006":
        return contrast + min(1.0, preserve) - clip - (0.5 * fp)
    return contrast * (1.0 + min(1.0, preserve)) - (2.0 * clip)


def _gx005_target_rows(
    scene_id: str,
    candidate_label: str,
    ntraces: int,
    data: np.ndarray,
    lane_row: dict[str, Any],
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    gt = bundle["ground_truth"]
    rois = [r for r in gt.get("rois", []) if isinstance(r, dict) and r.get("roi_type") == "target"]
    outputs: list[dict[str, Any]] = []
    processed = None
    fig_name = lane_row.get("figure")
    if fig_name:
        # no raw processed matrix from lane helper; approximate using metrics only
        processed = None
    for roi in rois[:2]:
        tid = str(roi.get("associated_object_id") or roi.get("roi_id") or "target")
        try:
            s0 = int(roi["sample_start_idx"])
            s1 = int(roi["sample_end_idx"])
            t0 = int(roi["trace_start_idx"])
            t1 = int(roi["trace_end_idx"])
        except Exception:
            continue
        s0 = max(0, min(data.shape[0] - 1, s0))
        s1 = max(s0 + 1, min(data.shape[0], s1))
        t0 = max(0, min(data.shape[1] - 1, t0))
        t1 = max(t0 + 1, min(data.shape[1], t1))
        raw_patch = data[s0:s1, t0:t1]
        raw_energy = float(np.sqrt(np.mean(np.square(raw_patch)))) if raw_patch.size else 0.0
        metrics = lane_row.get("after_gain_metrics", {})
        outputs.append(
            {
                "scene_id": scene_id,
                "target_id": tid,
                "candidate_label": candidate_label,
                "generated_ntraces": int(ntraces),
                "target_roi_energy_raw": raw_energy,
                "target_roi_contrast_proxy": metrics.get("roi_to_local_background_contrast"),
                "target_preservation_proxy": metrics.get("roi_energy_ratio_to_input"),
                "warning": "Target-specific processed ROI uses shared lane metrics proxy; check overlays for final interpretation.",
            }
        )
    return outputs


def _copy_best_figure(scene_id: str, best_row: dict[str, Any], figs_dir: Path) -> None:
    name_map = {
        "GX-003": "gx003_best_overlay.png",
        "GX-004": "gx004_false_positive_map.png",
        "GX-005": "gx005_target_specific_comparison.png",
        "GX-006": "gx006_layer_interface_comparison.png",
    }
    src = figs_dir / Path(str(best_row.get("figure", ""))).name
    dst = figs_dir / name_map.get(scene_id, f"{scene_id.lower()}_best_overlay.png")
    if src.exists():
        shutil.copy2(src, dst)


def _evaluate_gate(
    scene_rows: list[dict[str, Any]],
    per_target_rows: list[dict[str, Any]],
    false_positive_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    risk_flags: list[str] = []
    gx004 = [r for r in false_positive_rows if r["scene_id"] == "GX-004"]
    if any(bool(r["artifact_risk"]) for r in gx004):
        risk_flags.append("gx004_false_positive_risk")
    gx005 = [r for r in per_target_rows if r["scene_id"] == "GX-005"]
    if gx005:
        vals = [float(r.get("target_preservation_proxy") or 0.0) for r in gx005]
        if max(vals) - min(vals) > 0.25:
            risk_flags.append("gx005_target_imbalance")
    scene_best = {}
    for s in ("GX-003", "GX-004", "GX-005", "GX-006"):
        subset = [r for r in scene_rows if r["scene_id"] == s]
        if not subset:
            risk_flags.append(f"{s.lower()}_missing")
            continue
        best = sorted(subset, key=lambda x: float(x["candidate_score"]), reverse=True)[0]
        scene_best[s] = best
        if float(best["ntraces_over_trace_count"]) >= 0.95:
            risk_flags.append(f"{s.lower()}_best_at_full_line_edge")
    labels = {k: v["candidate_label"] for k, v in scene_best.items()}
    stable = len(set(labels.values())) <= 2 if labels else False
    if not stable:
        risk_flags.append("scene_specific_policy_variation")
    risk_flags.append("synthetic_thin2d_scene_limit")
    gate_status = "blocked" if risk_flags else "pass"
    if gate_status == "pass" and not stable:
        gate_status = "partial_pass"
    return {
        "gate_status": gate_status,
        "risk_flags": risk_flags,
        "scene_best_labels": labels,
        "summary": "AT-012 preset-finalization gate remains blocked unless cross-scene stability and safety flags clear.",
    }


def _save_scene_candidate_overview(rows: list[dict[str, Any]], path: Path) -> None:
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
        xs = [float(x["ntraces_over_trace_count"]) for x in data]
        ys = [float(x["candidate_score"]) for x in data]
        ax.plot(xs, ys, "o-", color="#2563eb")
        ax.set_title(scene)
        ax.set_xlabel("ntraces/trace_count")
        ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_gx005_target_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    labels = [f"{r['candidate_label']}:{r['target_id']}" for r in rows]
    vals = [float(r.get("target_preservation_proxy") or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    ax.bar(range(len(vals)), vals, color="#0f766e")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("target_preservation_proxy")
    ax.set_title("GX-005 per-target preservation proxy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_gx006_layer_plot(rows: list[dict[str, Any]], path: Path) -> None:
    gx006 = [r for r in rows if r["scene_id"] == "GX-006"]
    if not gx006:
        return
    gx006 = sorted(gx006, key=lambda x: float(x["ntraces_over_trace_count"]))
    xs = [float(r["ntraces_over_trace_count"]) for r in gx006]
    ys = [float(r.get("negative_control_energy") or 0.0) for r in gx006]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.plot(xs, ys, "o-", color="#9333ea")
    ax.set_xlabel("ntraces/trace_count")
    ax.set_ylabel("negative_control_energy_proxy")
    ax.set_title("GX-006 layer/interface-aware clutter proxy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_gx004_false_positive_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda x: float(x["generated_ntraces"]))
    labels = [str(r["generated_ntraces"]) for r in rows]
    vals = [float(r.get("false_positive_proxy") or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.bar(labels, vals, color="#dc2626")
    ax.set_xlabel("ntraces")
    ax.set_ylabel("false_positive_proxy")
    ax.set_title("GX-004 false-positive proxy map")
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
        "# AT-013 Multi-scene Relative Policy Validation",
        "",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Included scenes: `{summary['included_scenes']}`",
        f"- Gate status: `{summary['gate_status']['gate_status']}`",
        "",
        "## Primary lane",
        "- set_zero_time excluded/fixed-zero",
        "- dewow excluded from primary lane",
        "- AT-011 relative candidate policy unchanged",
        "- energy_decay_gain conservative lane",
        "",
        "## Scene summary",
    ]
    for s in summary["scene_summaries"]:
        lines.append(
            f"- {s['scene_id']}: best `{s['best_candidate_label']}` ntraces=`{s['best_candidate_ntraces']}` "
            f"(trace_count={s['trace_count']}, candidates={s['candidate_count']})"
        )
    lines.extend(
        [
            "",
            "## Gate evaluation",
            f"- status: `{summary['gate_status']['gate_status']}`",
            f"- risk_flags: `{summary['gate_status']['risk_flags']}`",
            "- Thin 2D limitations remain and no field-performance claim is made.",
            "- No overall AutoTune superiority claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_html(summary: dict[str, Any]) -> str:
    risks = "".join(f"<li>{html.escape(str(item))}</li>" for item in summary["risk_flags"])
    scene_cards = "".join(
        "<li>"
        + html.escape(
            f"{s['scene_id']}: best={s['best_candidate_label']} ntraces={s['best_candidate_ntraces']} "
            f"trace_count={s['trace_count']}"
        )
        + "</li>"
        for s in summary["scene_summaries"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AT-013 Multi-scene Relative Policy Validation</title>
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
  <h1>AT-013 Multi-scene Relative Policy Validation</h1>
  <div class="card">
    <div><b>Source commit:</b> <code>{summary['source_commit']}</code></div>
    <div><b>Included scenes:</b> {html.escape(str(summary['included_scenes']))}</div>
    <div><b>Gate status:</b> <code>{summary['gate_status']['gate_status']}</code></div>
  </div>
  <div class="card">
    <h2>Scene best candidates</h2>
    <ul>{scene_cards}</ul>
  </div>
  <div class="card">
    <h2>Main risk flags</h2>
    <ul>{risks}</ul>
  </div>
  <div class="grid">
    <img src="../figures/scene_candidate_overview.png" alt="scene candidate overview">
    <img src="../figures/gx004_false_positive_map.png" alt="gx004 false positive map">
    <img src="../figures/gx005_target_specific_comparison.png" alt="gx005 target specific comparison">
    <img src="../figures/gx006_layer_interface_comparison.png" alt="gx006 layer interface comparison">
  </div>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
