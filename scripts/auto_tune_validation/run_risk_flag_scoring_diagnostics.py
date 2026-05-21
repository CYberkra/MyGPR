#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AT-016 risk-flag and scoring diagnostics from AT-014 evidence tables."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.auto_tune_validation.run_stepwise_validation import _git_rev_parse, _json_safe


AT014_TABLES = {
    "scene_candidate_metrics": "tables/scene_candidate_metrics.csv",
    "gx004": "tables/gx004_false_positive_fidelity_metrics.csv",
    "gx005": "tables/gx005_per_target_processed_metrics.csv",
    "gx006": "tables/gx006_layer_interface_metrics.csv",
    "gate": "tables/gate_reassessment.csv",
    "warnings": "tables/warnings_and_risk_flags.csv",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at014-root", required=True, help="Path to AT-014 evidence artifact root.")
    parser.add_argument("--evidence-root", required=True, help="Path to AT-016 evidence artifact root.")
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args(argv)
    result = run_risk_flag_scoring_diagnostics(
        at014_root=Path(args.at014_root),
        evidence_root=Path(args.evidence_root),
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    return 0


def run_risk_flag_scoring_diagnostics(
    *,
    at014_root: Path,
    evidence_root: Path,
    source_repo: str,
    source_branch: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    source_commit = source_commit or _git_rev_parse(ROOT)
    tables_dir = evidence_root / "tables"
    figures_dir = evidence_root / "figures"
    reports_dir = evidence_root / "reports"
    manifests_dir = evidence_root / "manifests"
    for d in (tables_dir, figures_dir, reports_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    scene_rows = _read_json_csv_rows(at014_root / AT014_TABLES["scene_candidate_metrics"])
    gx004_rows = _read_json_csv_rows(at014_root / AT014_TABLES["gx004"])
    gx005_rows = _read_json_csv_rows(at014_root / AT014_TABLES["gx005"])
    gx006_rows = _read_json_csv_rows(at014_root / AT014_TABLES["gx006"])
    gate_rows = _read_json_csv_rows(at014_root / AT014_TABLES["gate"])
    warnings_rows = _read_json_csv_rows(at014_root / AT014_TABLES["warnings"])
    at014_summary = json.loads((at014_root / "manifests" / "metric_fidelity_summary.json").read_text(encoding="utf-8"))

    ranking_breakdown = _build_ranking_breakdown(scene_rows)
    gx005_diag = _build_gx005_diag(gx005_rows, ranking_breakdown)
    gx006_diag = _build_gx006_diag(gx006_rows, ranking_breakdown)
    gx004_diag = _build_gx004_diag(gx004_rows, ranking_breakdown)
    risk_vs_score = _build_risk_vs_score_matrix(
        ranking_breakdown=ranking_breakdown,
        gx004_diag=gx004_diag,
        gx005_diag=gx005_diag,
        gx006_diag=gx006_diag,
        gate_rows=gate_rows,
        warnings_rows=warnings_rows,
    )
    gate_explain = _build_gate_blocker_explanation(
        gate_rows=gate_rows,
        gx005_diag=gx005_diag,
        gx006_diag=gx006_diag,
    )
    next_actions = _build_recommended_next_actions()

    _write_csv(tables_dir / "ranking_breakdown.csv", ranking_breakdown)
    _write_csv(tables_dir / "risk_vs_score_matrix.csv", risk_vs_score)
    _write_csv(tables_dir / "gx004_no_target_safety_diagnostics.csv", gx004_diag)
    _write_csv(tables_dir / "gx005_target_balance_diagnostics.csv", gx005_diag)
    _write_csv(tables_dir / "gx006_interface_suppression_diagnostics.csv", gx006_diag)
    _write_csv(tables_dir / "gate_blocker_explanation.csv", gate_explain)
    _write_csv(tables_dir / "recommended_next_actions.csv", next_actions)

    _save_risk_vs_score_plot(risk_vs_score, figures_dir / "risk_vs_score_overview.png")
    _save_gx005_plot(gx005_diag, figures_dir / "gx005_target_balance_vs_score.png")
    _save_gx006_plot(gx006_diag, figures_dir / "gx006_interface_ratio_vs_score.png")
    _save_gx004_plot(gx004_diag, figures_dir / "gx004_false_positive_proxy_vs_score.png")

    summary = {
        "artifact_id": "AT-016",
        "task_id": "AT-016_risk_flag_scoring_diagnostics",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_base_artifact": "AT-014_multi_scene_metric_fidelity_validation",
        "at014_source_commit": at014_summary.get("source_commit"),
        "at014_rerun": False,
        "scoring_changed": False,
        "algorithms_changed": False,
        "included_scenes": at014_summary.get("included_scenes", []),
        "gate_status": at014_summary.get("gate_reassessment", {}).get("gate_status"),
        "risk_flags": at014_summary.get("gate_reassessment", {}).get("risk_flags", []),
        "scene_best": at014_summary.get("scene_best", {}),
        "local_n9_interpretation": (
            "local n=9 ranks best in AT-014 candidate scoring for all scenes, "
            "but gate blockers remain active and preset promotion is forbidden."
        ),
        "gx004_safety_summary": _summarize_gx004(gx004_diag),
        "gx005_balance_summary": _summarize_gx005(gx005_diag),
        "gx006_interface_summary": _summarize_gx006(gx006_diag),
        "diagnostic_conclusion": (
            "Current issue is not explained by score ordering alone. "
            "Ranking and risk gate are intentionally separate; "
            "blockers indicate candidate-space and scene-safety constraints remain unresolved."
        ),
        "recommended_immediate_next_task": "AT-017 scoring/risk-penalty refinement (diagnostic design first, no semantic scoring switch yet)",
        "claim_boundary": {
            "preset_promotion": "forbidden",
            "overall_autotune_superiority": "forbidden",
            "field_performance_validation": "forbidden",
            "thin_2d_limitation": "disclosed",
        },
    }
    manifest = {
        "artifact_id": "AT-016",
        "task_id": "AT-016_risk_flag_scoring_diagnostics",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending_self_reference",
        "generated_at": summary["generated_at"],
        "command": "python scripts/auto_tune_validation/run_risk_flag_scoring_diagnostics.py",
        "dataset_name": "AT-014 diagnostic tables",
        "dataset_shape": "multi-scene tabular diagnostics",
        "dataset_hash": "manifest-driven",
        "ground_truth_available": True,
        "metric_type": "risk_flag_scoring_diagnostics",
        "artifacts": {
            "markdown_report": "reports/risk_flag_scoring_diagnostics_report.md",
            "html_report": "reports/risk_flag_scoring_diagnostics_report.html",
            "summary_json": "manifests/risk_flag_scoring_diagnostics_summary.json",
            "ranking_breakdown_csv": "tables/ranking_breakdown.csv",
            "risk_vs_score_csv": "tables/risk_vs_score_matrix.csv",
            "gx004_diag_csv": "tables/gx004_no_target_safety_diagnostics.csv",
            "gx005_diag_csv": "tables/gx005_target_balance_diagnostics.csv",
            "gx006_diag_csv": "tables/gx006_interface_suppression_diagnostics.csv",
            "gate_explain_csv": "tables/gate_blocker_explanation.csv",
            "recommended_next_actions_csv": "tables/recommended_next_actions.csv",
        },
    }

    _write_json(manifests_dir / "risk_flag_scoring_diagnostics_summary.json", summary)
    _write_json(manifests_dir / "evidence_manifest.json", manifest)
    (reports_dir / "risk_flag_scoring_diagnostics_report.md").write_text(_render_md(summary), encoding="utf-8")
    (reports_dir / "risk_flag_scoring_diagnostics_report.html").write_text(_render_html(summary), encoding="utf-8")

    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "gate_status": summary["gate_status"],
        "risk_flags": summary["risk_flags"],
        "at014_rerun": False,
        "scoring_changed": False,
        "algorithms_changed": False,
    }


def _read_json_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                parsed[k] = _parse_cell(v)
            rows.append(parsed)
    return rows


def _parse_cell(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    if text.lower() == "null":
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return default
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return default
        return int(round(value))
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _candidate_key(scene_id: str, label: str, ntraces: int) -> str:
    return f"{scene_id}|{label}|{ntraces}"


def _build_ranking_breakdown(scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scene_rows:
        scene = str(row.get("scene_id"))
        grouped.setdefault(scene, []).append(row)

    out: list[dict[str, Any]] = []
    for scene, rows in grouped.items():
        ordered = sorted(rows, key=lambda x: _to_float(x.get("candidate_score")), reverse=True)
        scores = [_to_float(r.get("candidate_score")) for r in ordered]
        best = scores[0] if scores else 0.0
        second = scores[1] if len(scores) > 1 else best
        max_n = max(_to_int(r.get("generated_ntraces")) for r in ordered) if ordered else 0
        for idx, row in enumerate(ordered):
            score = _to_float(row.get("candidate_score"))
            gap_to_best = best - score
            gap_best_second = best - second if idx == 0 else None
            rank = idx + 1
            n = _to_int(row.get("generated_ntraces"))
            near_tie = bool(abs(gap_to_best) <= 1e-6) if rank > 1 else bool(abs(best - second) <= 0.01)
            main_pos, main_neg = _metric_contributors(scene, row)
            out.append(
                {
                    "scene_id": scene,
                    "candidate_label": str(row.get("candidate_label")),
                    "generated_ntraces": n,
                    "candidate_score": score,
                    "rank": rank,
                    "score_gap_to_best": gap_to_best,
                    "score_gap_best_to_second": gap_best_second,
                    "best_clearly_separated": bool((best - second) > 0.05) if rank == 1 else None,
                    "near_tie": near_tie,
                    "best_at_domain_edge": bool(rank == 1 and n >= max_n),
                    "main_positive_metric_contributors": main_pos,
                    "main_negative_metric_contributors": main_neg,
                    "key": _candidate_key(scene, str(row.get("candidate_label")), n),
                }
            )
    return out


def _metric_contributors(scene: str, row: dict[str, Any]) -> tuple[list[str], list[str]]:
    clip = _to_float(row.get("clipping_ratio"))
    fp = _to_float(row.get("false_positive_proxy"))
    roi = _to_float(row.get("roi_contrast"))
    tp = _to_float(row.get("target_preservation"))
    if scene == "GX-004":
        return (
            [f"lower_false_positive_proxy={fp:.4f}", f"lower_clipping_ratio={clip:.4f}"],
            [f"global_energy_ratio={_to_float(row.get('global_energy_ratio')):.4f}"],
        )
    if scene == "GX-005":
        return (
            [f"higher_target_preservation={tp:.4f}", f"higher_roi_contrast={roi:.4f}"],
            [f"clipping_ratio_penalty={clip:.4f}", f"imbalance_gap={_to_float(row.get('max_target_preservation_gap')):.4f}"],
        )
    if scene == "GX-006":
        return (
            [f"higher_layer_ratio_proxy={roi:.4f}"],
            [f"clutter_fp_proxy={fp:.4f}", f"clipping_ratio={clip:.4f}"],
        )
    return (
        [f"roi_contrast={roi:.4f}", f"target_preservation={tp:.4f}"],
        [f"clipping_ratio={clip:.4f}"],
    )


def _build_gx005_diag(
    gx005_rows: list[dict[str, Any]],
    ranking_breakdown: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    score_map = {r["key"]: r for r in ranking_breakdown}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in gx005_rows:
        key = (str(row.get("candidate_label")), _to_int(row.get("generated_ntraces")))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (label, n), rows in sorted(grouped.items(), key=lambda x: x[0][1]):
        scene = "GX-005"
        rk = score_map.get(_candidate_key(scene, label, n), {})
        target_map = {str(r.get("target_id")): r for r in rows}
        a = target_map.get("target_A", {})
        b = target_map.get("target_B", {})
        a_pres = _to_float(a.get("target_preservation_ratio"))
        b_pres = _to_float(b.get("target_preservation_ratio"))
        a_con = _to_float(a.get("contrast_ratio"))
        b_con = _to_float(b.get("contrast_ratio"))
        preserve_gap = abs(a_pres - b_pres)
        one_target_only = bool((a_con > 1.0 and b_con < 1.0) or (b_con > 1.0 and a_con < 1.0))
        imbalance = bool(preserve_gap > 0.25 or one_target_only)
        out.append(
            {
                "scene_id": scene,
                "candidate_label": label,
                "generated_ntraces": n,
                "candidate_score": _to_float(rk.get("candidate_score")),
                "candidate_rank": _to_int(rk.get("rank")),
                "target_A_preservation_ratio": a_pres,
                "target_B_preservation_ratio": b_pres,
                "target_A_contrast_ratio": a_con,
                "target_B_contrast_ratio": b_con,
                "preservation_gap": preserve_gap,
                "one_target_only_improvement": one_target_only,
                "target_imbalance_risk": imbalance,
                "local_n9_candidate": bool(label == "local" and n == 9),
            }
        )
    return out


def _build_gx006_diag(
    gx006_rows: list[dict[str, Any]],
    ranking_breakdown: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    score_map = {r["key"]: r for r in ranking_breakdown}
    out: list[dict[str, Any]] = []
    for row in sorted(gx006_rows, key=lambda x: _to_int(x.get("generated_ntraces"))):
        label = str(row.get("candidate_label"))
        n = _to_int(row.get("generated_ntraces"))
        rk = score_map.get(_candidate_key("GX-006", label, n), {})
        layer_ratio = _to_float(row.get("layer_interface_energy_ratio"))
        clutter_proxy = _to_float(row.get("clutter_false_positive_proxy"))
        interface_risk = bool(row.get("interface_suppression_risk"))
        out.append(
            {
                "scene_id": "GX-006",
                "candidate_label": label,
                "generated_ntraces": n,
                "candidate_score": _to_float(rk.get("candidate_score")),
                "candidate_rank": _to_int(rk.get("rank")),
                "layer_interface_energy_ratio": layer_ratio,
                "clutter_false_positive_proxy": clutter_proxy,
                "interface_suppression_risk": interface_risk,
                "clutter_false_positive_risk": bool(row.get("clutter_false_positive_risk")),
                "interface_suppression_common_across_candidates": None,
                "local_n9_candidate": bool(label == "local" and n == 9),
            }
        )
    common = all(bool(r["interface_suppression_risk"]) for r in out) if out else False
    for r in out:
        r["interface_suppression_common_across_candidates"] = common
    return out


def _build_gx004_diag(
    gx004_rows: list[dict[str, Any]],
    ranking_breakdown: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    score_map = {r["key"]: r for r in ranking_breakdown}
    proxies = [_to_float(r.get("false_positive_proxy")) for r in gx004_rows]
    median_proxy = float(np.median(np.array(proxies))) if proxies else 0.0
    out: list[dict[str, Any]] = []
    for row in sorted(gx004_rows, key=lambda x: _to_int(x.get("generated_ntraces"))):
        label = str(row.get("candidate_label"))
        n = _to_int(row.get("generated_ntraces"))
        rk = score_map.get(_candidate_key("GX-004", label, n), {})
        proxy = _to_float(row.get("false_positive_proxy"))
        out.append(
            {
                "scene_id": "GX-004",
                "candidate_label": label,
                "generated_ntraces": n,
                "candidate_score": _to_float(rk.get("candidate_score")),
                "candidate_rank": _to_int(rk.get("rank")),
                "false_positive_proxy": proxy,
                "artifact_risk": bool(row.get("artifact_risk")),
                "clipping_ratio": _to_float(row.get("clipping_ratio")),
                "global_energy_ratio": _to_float(row.get("global_energy_ratio")),
                "safer_than_scene_median_proxy": bool(proxy <= median_proxy),
                "local_n9_candidate": bool(label == "local" and n == 9),
            }
        )
    return out


def _build_risk_vs_score_matrix(
    *,
    ranking_breakdown: list[dict[str, Any]],
    gx004_diag: list[dict[str, Any]],
    gx005_diag: list[dict[str, Any]],
    gx006_diag: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    warnings_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gx004_map = {_candidate_key("GX-004", str(r["candidate_label"]), _to_int(r["generated_ntraces"])): r for r in gx004_diag}
    gx005_map = {_candidate_key("GX-005", str(r["candidate_label"]), _to_int(r["generated_ntraces"])): r for r in gx005_diag}
    gx006_map = {_candidate_key("GX-006", str(r["candidate_label"]), _to_int(r["generated_ntraces"])): r for r in gx006_diag}
    warn_map: dict[str, list[str]] = {}
    for w in warnings_rows:
        k = _candidate_key(str(w.get("scene_id")), str(w.get("candidate_label")), _to_int(w.get("generated_ntraces")))
        rf = w.get("risk_flags") if isinstance(w.get("risk_flags"), list) else []
        warn_map[k] = [str(x) for x in rf]
    gate_flags = []
    if gate_rows:
        raw = gate_rows[0].get("risk_flags")
        gate_flags = [str(x) for x in raw] if isinstance(raw, list) else []
    synthetic_flag_global = "synthetic_thin2d_scene_limit" in gate_flags

    out: list[dict[str, Any]] = []
    for row in ranking_breakdown:
        k = str(row["key"])
        scene = str(row["scene_id"])
        fp_risk = bool(gx004_map.get(k, {}).get("artifact_risk", False))
        imb_risk = bool(gx005_map.get(k, {}).get("target_imbalance_risk", False))
        int_risk = bool(gx006_map.get(k, {}).get("interface_suppression_risk", False))
        local_warn = warn_map.get(k, [])
        risk_count = int(fp_risk) + int(imb_risk) + int(int_risk) + int(synthetic_flag_global)
        hard_blocker = bool(imb_risk or int_risk or synthetic_flag_global)
        pass_score_only = _to_int(row.get("rank")) == 1
        pass_gate = bool(pass_score_only and not hard_blocker)
        out.append(
            {
                "scene_id": scene,
                "candidate_label": row["candidate_label"],
                "generated_ntraces": _to_int(row["generated_ntraces"]),
                "candidate_score": _to_float(row["candidate_score"]),
                "candidate_rank": _to_int(row["rank"]),
                "false_positive_risk": fp_risk,
                "target_imbalance_risk": imb_risk,
                "interface_suppression_risk": int_risk,
                "synthetic_limit_flag": synthetic_flag_global,
                "risk_count": risk_count,
                "hard_blocker_present": hard_blocker,
                "would_pass_if_score_only": pass_score_only,
                "would_pass_if_gate_applied": pass_gate,
                "warning_flags_raw": local_warn,
            }
        )
    return out


def _build_gate_blocker_explanation(
    *,
    gate_rows: list[dict[str, Any]],
    gx005_diag: list[dict[str, Any]],
    gx006_diag: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags = gate_rows[0].get("risk_flags") if gate_rows else []
    flags = [str(x) for x in flags] if isinstance(flags, list) else []
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "blocker": "gx005_target_imbalance",
            "present": "gx005_target_imbalance" in flags,
            "meaning": "Candidate behavior is imbalanced between target_A and target_B.",
            "why_blocking": "Preset cannot be finalized if one target improves while the other degrades.",
            "clear_evidence_needed": "Per-candidate and cross-scene target-balance metrics converge under accepted thresholds.",
            "hard_blocker": True,
        }
    )
    rows.append(
        {
            "blocker": "gx006_interface_suppression",
            "present": "gx006_interface_suppression" in flags,
            "meaning": "Layer/interface responses are over-suppressed in layered scenes.",
            "why_blocking": "Structurally meaningful interfaces cannot be removed without safety controls.",
            "clear_evidence_needed": "Interface-aware safety rule or validated candidate domain preserving interface continuity.",
            "hard_blocker": True,
        }
    )
    rows.append(
        {
            "blocker": "synthetic_thin2d_scene_limit",
            "present": "synthetic_thin2d_scene_limit" in flags,
            "meaning": "Evidence base is synthetic thin-2D only.",
            "why_blocking": "Preset finalization requires broader validity than current synthetic regime.",
            "clear_evidence_needed": "Additional scene diversity and/or field no-prior QC evidence chain.",
            "hard_blocker": True,
        }
    )
    rows.append(
        {
            "blocker": "ranking_vs_gate_relation",
            "present": True,
            "meaning": "Candidate ranking and gate safety are separate decision layers.",
            "why_blocking": "Top score is insufficient when hard blockers remain.",
            "clear_evidence_needed": "Risk blockers cleared first, then ranking can support recommendation.",
            "hard_blocker": False,
        }
    )
    return rows


def _build_recommended_next_actions() -> list[dict[str, Any]]:
    return [
        {
            "priority": 1,
            "task": "AT-017 scoring/risk-penalty refinement design",
            "reason": "AT-016 shows ranking alone cannot satisfy gate safety constraints.",
            "scope_note": "Design diagnostics-first weighting proposals without immediate semantic scoring switch.",
            "recommended_now": True,
        },
        {
            "priority": 2,
            "task": "Interface-aware background suppression safety rule",
            "reason": "GX-006 interface suppression risk is broad across candidates.",
            "scope_note": "Candidate filtering/safety guard before preset promotion.",
            "recommended_now": False,
        },
        {
            "priority": 3,
            "task": "Additional benchmark scene expansion",
            "reason": "Thin-2D synthetic limitation remains a hard blocker.",
            "scope_note": "Use expanded scene diversity to reduce transfer risk.",
            "recommended_now": False,
        },
        {
            "priority": 4,
            "task": "DS-002 external decay-profile comparison",
            "reason": "Field-shallow/deep decay context can calibrate interpretation risk.",
            "scope_note": "No immediate scoring change; context-building for later claims.",
            "recommended_now": False,
        },
    ]


def _summarize_gx004(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    best = min(rows, key=lambda r: _to_float(r.get("false_positive_proxy"), default=1e9))
    any_artifact = any(bool(r.get("artifact_risk")) for r in rows)
    return {
        "artifact_risk_present": any_artifact,
        "lowest_false_positive_candidate": {
            "label": best.get("candidate_label"),
            "ntraces": best.get("generated_ntraces"),
            "false_positive_proxy": best.get("false_positive_proxy"),
        },
    }


def _summarize_gx005(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    local9 = next((r for r in rows if bool(r.get("local_n9_candidate"))), None)
    broad = all(bool(r.get("target_imbalance_risk")) for r in rows)
    return {
        "target_imbalance_common_across_candidates": broad,
        "local_n9_preservation_gap": _to_float(local9.get("preservation_gap")) if local9 else None,
        "max_preservation_gap": max(_to_float(r.get("preservation_gap")) for r in rows),
    }


def _summarize_gx006(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    common = all(bool(r.get("interface_suppression_risk")) for r in rows)
    local9 = next((r for r in rows if bool(r.get("local_n9_candidate"))), None)
    return {
        "interface_suppression_common_across_candidates": common,
        "local_n9_layer_interface_ratio": _to_float(local9.get("layer_interface_energy_ratio")) if local9 else None,
        "min_layer_interface_ratio": min(_to_float(r.get("layer_interface_energy_ratio")) for r in rows),
        "max_layer_interface_ratio": max(_to_float(r.get("layer_interface_energy_ratio")) for r in rows),
    }


def _save_risk_vs_score_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    scenes = ["GX-003", "GX-004", "GX-005", "GX-006"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    axs = axes.flatten()
    for idx, scene in enumerate(scenes):
        ax = axs[idx]
        data = [r for r in rows if str(r.get("scene_id")) == scene]
        data.sort(key=lambda x: _to_int(x.get("generated_ntraces")))
        if not data:
            ax.set_title(scene + " (missing)")
            continue
        xs = [_to_int(r.get("generated_ntraces")) for r in data]
        ys = [_to_float(r.get("candidate_score")) for r in data]
        cs = [_to_int(r.get("risk_count")) for r in data]
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=42)
        ax.plot(xs, ys, color="#64748b", linewidth=1.0, alpha=0.8)
        ax.set_title(scene)
        ax.set_xlabel("ntraces")
        ax.set_ylabel("candidate_score")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="risk_count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_gx005_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda x: _to_int(x.get("generated_ntraces")))
    x = [_to_int(r.get("generated_ntraces")) for r in rows]
    a = [_to_float(r.get("target_A_preservation_ratio")) for r in rows]
    b = [_to_float(r.get("target_B_preservation_ratio")) for r in rows]
    gap = [_to_float(r.get("preservation_gap")) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    ax.plot(x, a, "o-", label="target_A_preservation")
    ax.plot(x, b, "s-", label="target_B_preservation")
    ax.plot(x, gap, "^-", label="preservation_gap")
    ax.axhline(0.25, color="#ef4444", linestyle="--", linewidth=1.0, label="imbalance_threshold")
    ax.set_xlabel("ntraces")
    ax.set_ylabel("ratio")
    ax.set_title("GX-005 target balance vs candidate space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_gx006_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda x: _to_int(x.get("generated_ntraces")))
    x = [_to_int(r.get("generated_ntraces")) for r in rows]
    layer = [_to_float(r.get("layer_interface_energy_ratio")) for r in rows]
    clutter = [_to_float(r.get("clutter_false_positive_proxy")) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    ax.plot(x, layer, "o-", label="layer_interface_energy_ratio")
    ax.plot(x, clutter, "s-", label="clutter_false_positive_proxy")
    ax.axhline(0.7, color="#ef4444", linestyle="--", linewidth=1.0, label="interface_risk_threshold")
    ax.set_xlabel("ntraces")
    ax.set_ylabel("metric")
    ax.set_title("GX-006 interface suppression vs score space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_gx004_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda x: _to_int(x.get("generated_ntraces")))
    x = [_to_int(r.get("generated_ntraces")) for r in rows]
    fp = [_to_float(r.get("false_positive_proxy")) for r in rows]
    score = [_to_float(r.get("candidate_score")) for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 4), dpi=140)
    ax1.plot(x, fp, "o-", color="#0f766e", label="false_positive_proxy")
    ax1.set_xlabel("ntraces")
    ax1.set_ylabel("false_positive_proxy", color="#0f766e")
    ax2 = ax1.twinx()
    ax2.plot(x, score, "s--", color="#2563eb", label="candidate_score")
    ax2.set_ylabel("candidate_score", color="#2563eb")
    ax1.set_title("GX-004 no-target safety vs score")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")
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


def _render_md(summary: dict[str, Any]) -> str:
    gx004 = summary.get("gx004_safety_summary", {})
    gx005 = summary.get("gx005_balance_summary", {})
    gx006 = summary.get("gx006_interface_summary", {})
    lines = [
        "# AT-016 Risk-flag and Scoring Diagnostics",
        "",
        "## 1. Executive summary",
        "- AT-016 is diagnostics-only and is based on AT-014 outputs.",
        "- Candidate ranking and gate blocking are intentionally separate layers.",
        f"- Gate status remains `{summary.get('gate_status')}`.",
        "",
        "## 2. Why local n=9 ranks best",
        "- In AT-014 candidate scoring, local candidates retain higher per-scene score under the current metric composition.",
        "- For GX-003/004/005/006 the top-ranked candidate is local n=9.",
        "- Score leadership does not imply safety clearance or preset eligibility.",
        "",
        "## 3. Why gate remains blocked",
        f"- Active blockers: `{summary.get('risk_flags')}`.",
        "- Ranking answers 'which candidate scores higher'.",
        "- Gate answers 'is candidate safe enough for preset finalization'.",
        "- Current blockers prevent preset promotion even when score rank is 1.",
        "",
        "## 4. GX-004 finding",
        f"- artifact_risk_present: `{gx004.get('artifact_risk_present')}`.",
        "- No-target false-positive control remains comparatively stable in this diagnostic set.",
        "",
        "## 5. GX-005 finding",
        f"- target_imbalance_common_across_candidates: `{gx005.get('target_imbalance_common_across_candidates')}`.",
        f"- local_n9_preservation_gap: `{gx005.get('local_n9_preservation_gap')}`.",
        "- Imbalance risk is not reduced to zero by choosing local n=9.",
        "",
        "## 6. GX-006 finding",
        f"- interface_suppression_common_across_candidates: `{gx006.get('interface_suppression_common_across_candidates')}`.",
        f"- local_n9_layer_interface_ratio: `{gx006.get('local_n9_layer_interface_ratio')}`.",
        "- Interface suppression risk persists and is not unique to one candidate.",
        "",
        "## 7. Recommendation",
        "- Recommended immediate next task: **AT-017 scoring/risk-penalty refinement design**.",
        "- Keep scoring semantics unchanged in this phase; first produce bounded design diagnostics.",
        "",
        "## 8. Claim boundary",
        "- No preset promotion.",
        "- No overall AutoTune superiority claim.",
        "- No field-performance claim.",
        "- Thin-2D synthetic limitation remains.",
    ]
    return "\n".join(lines) + "\n"


def _render_html(summary: dict[str, Any]) -> str:
    risk_items = "".join(f"<li>{html.escape(str(x))}</li>" for x in summary.get("risk_flags", []))
    scene_best = summary.get("scene_best", {})
    best_items = "".join(
        (
            f"<li>{html.escape(scene)}: "
            f"label={html.escape(str(payload.get('best_candidate_label')))} "
            f"n={html.escape(str(payload.get('best_candidate_ntraces')))}</li>"
        )
        for scene, payload in scene_best.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AT-016 Risk-flag and Scoring Diagnostics</title>
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
  <h1>AT-016 Risk-flag and Scoring Diagnostics</h1>
  <div class="card">
    <div><b>Source commit:</b> <code>{html.escape(str(summary.get("source_commit")))}</code></div>
    <div><b>Diagnostic base:</b> <code>{html.escape(str(summary.get("diagnostic_base_artifact")))}</code></div>
    <div><b>Gate status:</b> <code>{html.escape(str(summary.get("gate_status")))}</code></div>
    <div><b>AT-014 rerun:</b> <code>{html.escape(str(summary.get("at014_rerun")))}</code></div>
    <div><b>Scoring changed:</b> <code>{html.escape(str(summary.get("scoring_changed")))}</code></div>
  </div>
  <div class="card">
    <h2>Active risk flags</h2>
    <ul>{risk_items}</ul>
  </div>
  <div class="card">
    <h2>Scene best candidates (from AT-014 base)</h2>
    <ul>{best_items}</ul>
  </div>
  <div class="card">
    <h2>Interpretation</h2>
    <p>{html.escape(str(summary.get("diagnostic_conclusion")))}</p>
    <p><b>Recommended immediate next task:</b> {html.escape(str(summary.get("recommended_immediate_next_task")))}</p>
  </div>
  <div class="grid">
    <img src="../figures/risk_vs_score_overview.png" alt="risk_vs_score_overview">
    <img src="../figures/gx004_false_positive_proxy_vs_score.png" alt="gx004_false_positive_proxy_vs_score">
    <img src="../figures/gx005_target_balance_vs_score.png" alt="gx005_target_balance_vs_score">
    <img src="../figures/gx006_interface_ratio_vs_score.png" alt="gx006_interface_ratio_vs_score">
  </div>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

