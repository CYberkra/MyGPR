#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AT-019 YingShan Line9 full AutoTune diagnostics evidence."""

from __future__ import annotations

import csv
import json
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

from core.auto_tune import AutoTuneError, auto_tune_method
from core.no_prior_qc_policy import build_no_prior_qc_policy
from core.processing_engine import prepare_runtime_params, run_processing_method
from read_file_data import readcsv
from core.gpr_io import extract_airborne_csv_payload


FIELD_CSV = Path(r"C:\Users\17844\Desktop\02_Preprocessed_Standard\2025-09_营山\Line9origin(36).csv")


def _git_head(repo: Path) -> str:
    import subprocess

    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True)
        .strip()
    )


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for r in rows for k in r.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if not keys:
            return
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), (dict, list)) else r.get(k) for k in keys})


def _read_header(path: Path) -> dict[str, Any]:
    info: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(6):
            line = f.readline()
            if not line:
                break
            if "=" not in line:
                continue
            left, right = line.split("=", 1)
            key = left.strip().lower()
            val = right.split(",")[0].strip()
            try:
                num = float(val)
            except ValueError:
                continue
            info[key] = num
    samples = int(info.get("number of samples", 0))
    traces = int(info.get("number of traces", 0))
    time_ns = float(info.get("time windows (ns)", info.get("time windows", 0.0)))
    trace_interval = float(info.get("trace interval (m)", info.get("trace interval", 0.0)))
    if samples <= 0 or traces <= 0:
        return {}
    return {
        "a_scan_length": samples,
        "num_traces": traces,
        "total_time_ns": time_ns,
        "trace_interval_m": trace_interval,
    }


def _plot_bscan(data: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(10, 4))
    vmax = np.percentile(np.abs(data), 99)
    plt.imshow(data, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.title(title)
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> int:
    evidence_root = Path(r"D:\CDUT-UavGPR-Controller\MyGPR-Evidence\autotune\AT-019_yingshan_line9_full_autotune_diagnostics")
    figures = evidence_root / "figures"
    tables = evidence_root / "tables"
    reports = evidence_root / "reports"
    manifests = evidence_root / "manifests"
    for d in [figures, tables, reports, manifests]:
        d.mkdir(parents=True, exist_ok=True)

    raw_csv = readcsv(str(FIELD_CSV))
    header = _read_header(FIELD_CSV)
    data, trace_metadata, header_info = extract_airborne_csv_payload(raw_csv, header)
    data = np.asarray(data, dtype=np.float32)
    header_info = dict(header_info or {})
    trace_metadata = dict(trace_metadata or {})

    _plot_bscan(data, figures / "raw_bscan_preview.png", "AT-019 Raw B-scan (diagnostic only)")

    stages = ["set_zero_time", "dewow", "frequency_filter_1d", "subtracting_average_2D", "energy_decay_gain"]
    stage_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    dewow_rows: list[dict[str, Any]] = []
    background_rows: list[dict[str, Any]] = []

    current = np.array(data, copy=True)
    first_warn_stage = None
    for i, method in enumerate(stages, start=1):
        try:
            result = auto_tune_method(current, method, header_info=header_info, trace_metadata=trace_metadata)
        except AutoTuneError as exc:
            stage_rows.append({"stage_index": i, "method_key": method, "status": "error", "error": str(exc)})
            if first_warn_stage is None:
                first_warn_stage = method
            break
        warnings = result.get("constraint_warnings", [])
        if warnings and first_warn_stage is None:
            first_warn_stage = method
        stage_rows.append({
            "stage_index": i,
            "method_key": method,
            "status": "ok",
            "best_params": result.get("best_params", {}),
            "selection_margin": result.get("selection_margin"),
            "selection_confidence": result.get("selection_confidence"),
            "constraint_warning_count": len(warnings),
            "risk_flags": result.get("risk_flags", []),
        })
        for w in warnings:
            constraint_rows.append({"stage": method, **w})
        for t in result.get("all_trials", []):
            row = {
                "stage": method,
                "trial_label": t.get("label"),
                "requested_params": t.get("requested_params"),
                "effective_params": t.get("params"),
                "constraint_adjusted": t.get("constraint_adjusted"),
                "score": t.get("score"),
                "valid": t.get("valid", True),
                "reason": t.get("reason"),
                "metrics": t.get("metrics"),
                "penalties": t.get("penalties"),
                "runtime_warning_count": len(t.get("runtime_warnings", []) or []),
            }
            trial_rows.append(row)
            if method == "dewow":
                dewow_rows.append(row)
            if method in {"subtracting_average_2D", "median_background_2D", "running_average_2D", "svd_bg"}:
                background_rows.append(row)
            for rw in t.get("runtime_warnings", []) or []:
                runtime_rows.append({"stage": method, "trial_label": t.get("label"), **rw})

        best_params = dict(result.get("best_params") or {})
        runtime_params = prepare_runtime_params(method, best_params, header_info, trace_metadata, current.shape)
        current, meta = run_processing_method(current, method, runtime_params)
        for rw in meta.get("runtime_warnings", []) or []:
            runtime_rows.append({"stage": method, "trial_label": "selected_apply", **rw})

    bg_methods = ["subtracting_average_2D", "median_background_2D", "running_average_2D", "svd_bg"]
    bg_images = [("raw", data)]
    for m in bg_methods:
        try:
            r = auto_tune_method(data, m, header_info=header_info, trace_metadata=trace_metadata)
            best = dict(r.get("best_params") or {})
            rp = prepare_runtime_params(m, best, header_info, trace_metadata, data.shape)
            out, _ = run_processing_method(data, m, rp)
            bg_images.append((m, out))
            for t in r.get("all_trials", []):
                background_rows.append({"stage": m, "trial_label": t.get("label"), "requested_params": t.get("requested_params"), "effective_params": t.get("params"), "score": t.get("score"), "reason": t.get("reason")})
        except Exception as exc:
            stage_rows.append({"stage_index": 100 + len(stage_rows), "method_key": f"bg_diag_{m}", "status": "error", "error": str(exc)})

    fig, axes = plt.subplots(1, len(bg_images), figsize=(4 * len(bg_images), 4), squeeze=False)
    for idx, (name, arr) in enumerate(bg_images):
        ax = axes[0, idx]
        vmax = np.percentile(np.abs(arr), 99)
        ax.imshow(arr, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("AT-019 Background candidates (diagnostic only)")
    fig.tight_layout()
    fig.savefig(figures / "background_candidate_panel.png", dpi=140)
    plt.close(fig)

    _csv_write(tables / "stage_summary.csv", stage_rows)
    _csv_write(tables / "trial_diagnostics.csv", trial_rows)
    _csv_write(tables / "constraint_warnings.csv", constraint_rows)
    _csv_write(tables / "runtime_warnings.csv", runtime_rows)
    _csv_write(tables / "background_candidate_diagnostics.csv", background_rows)
    _csv_write(tables / "dewow_candidate_diagnostics.csv", dewow_rows)

    no_prior = build_no_prior_qc_policy(no_prior_level="high_risk", target_prior_available=False, roi_available=False)
    no_prior_rows = [{
        "file": FIELD_CSV.name,
        "no_prior_level": no_prior.no_prior_level,
        "safe_auto_recommendation_allowed": no_prior.safe_auto_recommendation_allowed,
        "aggressive_background_suppression_allowed": no_prior.aggressive_background_suppression_allowed,
        "amplitude_claim_allowed": no_prior.amplitude_claim_allowed,
        "manual_review_required": no_prior.manual_review_required,
        "recommended_initial_policy": no_prior.recommended_initial_policy,
    }]
    _csv_write(tables / "no_prior_guard_interaction.csv", no_prior_rows)

    summary = {
        "artifact_id": "AT-019",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_head(ROOT),
        "field_csv": str(FIELD_CSV),
        "data_shape": [int(data.shape[0]), int(data.shape[1])],
        "first_warning_or_failure_stage": first_warn_stage,
        "full_autotune_completed": all(r.get("status", "ok") == "ok" for r in stage_rows if not str(r.get("method_key", "")).startswith("bg_diag_")),
        "dewow_window_256_clamped_to_250_expected": True,
        "no_prior_level": no_prior.no_prior_level,
        "claim_boundary": "diagnostic only; no target detection; no underground correctness; no preset promotion; no scoring change",
    }
    manifest = {
        "artifact_id": "AT-019",
        "task_id": "AT-019_yingshan_line9_full_autotune_diagnostics",
        "source_commit": summary["source_commit"],
        "artifacts": {
            "report_md": "reports/yingshan_line9_full_autotune_diagnostics_report.md",
            "report_html": "reports/yingshan_line9_full_autotune_diagnostics_report.html",
            "summary_json": "manifests/yingshan_line9_autotune_diagnostics_summary.json",
        },
    }
    _json_write(manifests / "yingshan_line9_autotune_diagnostics_summary.json", summary)
    _json_write(manifests / "evidence_manifest.json", manifest)

    md = [
        "# AT-019 YingShan Line9 Full AutoTune Diagnostics",
        "",
        "## Executive summary",
        f"- Full AutoTune pipeline status: {'completed' if summary['full_autotune_completed'] else 'partial/fail'}",
        f"- First warning/failure stage: {summary['first_warning_or_failure_stage']}",
        "- Dewow window=256 is expected to clamp to 250 for 501-sample data; this is constraint warning, not algorithm crash when run continues.",
        "- Background stage diagnostics generated across subtracting/median/running-average/svd methods.",
        "- No-prior high-risk policy implies UI should block aggressive auto recommendation paths and require manual review.",
        "",
        "## Data summary",
        f"- Field CSV: `{FIELD_CSV}`",
        f"- Parsed shape: `{data.shape}`",
        "- ROI: none (no-prior field diagnostics)",
        "",
        "## Claim boundary",
        "- Field diagnostic only; no ground-truth validation.",
        "- No target detection claim.",
        "- No underground correctness claim.",
        "- No AutoTune superiority claim.",
        "- No preset promotion.",
    ]
    (reports / "yingshan_line9_full_autotune_diagnostics_report.md").write_text("\n".join(md), encoding="utf-8")
    html = "<html><body><h1>AT-019</h1><p>See markdown report.</p></body></html>"
    (reports / "yingshan_line9_full_autotune_diagnostics_report.html").write_text(html, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
