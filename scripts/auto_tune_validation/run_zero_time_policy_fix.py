#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AT-006 zero-time policy fix evidence."""

from __future__ import annotations

import argparse
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

from core.processing_engine import clone_header_info, clone_trace_metadata, prepare_runtime_params, run_processing_method
from scripts.auto_tune_validation.run_no_zerotime_gain_validation import run_validation as run_no_zerotime_validation
from scripts.auto_tune_validation.run_stepwise_validation import (
    _energy_ratio,
    _git_rev_parse,
    _infer_zero_time_policy,
    _load_dataset,
    _run_branch,
    _write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, help="AT-006 evidence output directory")
    parser.add_argument(
        "--dataset",
        default=str(
            ROOT.parent
            / "MyGPR-Evidence"
            / "gprmax"
            / "GX-003_audited_native_gprmax_benchmark"
        ),
        help="GX-003 evidence folder or dataset path",
    )
    parser.add_argument("--source-repo", default="https://github.com/CYberkra/MyGPR")
    parser.add_argument("--source-branch", default="codex/research-gprmax-autotune")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args(argv)

    result = run_policy_fix(
        evidence_root=Path(args.evidence_root),
        dataset=args.dataset,
        source_repo=args.source_repo,
        source_branch=args.source_branch,
        source_commit=args.source_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_policy_fix(
    *,
    evidence_root: Path,
    dataset: str,
    source_repo: str,
    source_branch: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    source_commit = source_commit or _git_rev_parse(ROOT)
    package = _load_dataset(dataset)
    zero_time_policy = _infer_zero_time_policy(package)

    evidence_root.mkdir(parents=True, exist_ok=True)
    figures_dir = evidence_root / "figures"
    manifests_dir = evidence_root / "manifests"
    reports_dir = evidence_root / "reports"
    tables_dir = evidence_root / "tables"
    for directory in (figures_dir, manifests_dir, reports_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = package["data"]
    header_info = package["header_info"]
    trace_metadata = package["trace_metadata"]
    dt_ns = float(header_info.get("time_step_s", 0.0)) * 1.0e9

    legacy = _run_legacy_default_zero_time(raw, header_info, trace_metadata)
    guarded = _run_guarded_fixed_zero(raw, header_info, trace_metadata, package, figures_dir / "_policy_debug")
    no_zero_lane = _run_no_zero_lane(evidence_root, dataset, source_repo, source_branch, source_commit)

    before_after_png = figures_dir / "zero_time_policy_before_after.png"
    _save_before_after_figure(
        raw=raw,
        legacy=legacy["data_after"],
        guarded=guarded["data_after"],
        output_path=before_after_png,
    )

    checks_rows = [
        {
            "check_id": "legacy_default_shift",
            "status": "warn" if legacy["shift_samples"] > 0 else "ok",
            "legacy_shift_samples": legacy["shift_samples"],
            "legacy_shift_ns": legacy["shift_ns"],
            "legacy_global_energy_ratio": legacy["global_energy_ratio"],
            "notes": "Legacy empty params path relies on method default (5.0 ns).",
        },
        {
            "check_id": "guarded_policy_shift",
            "status": "ok" if guarded["shift_samples"] == 0 else "warn",
            "guarded_shift_samples": guarded["shift_samples"],
            "guarded_shift_ns": guarded["shift_ns"],
            "guarded_global_energy_ratio": guarded["global_energy_ratio"],
            "notes": "Validation guard enforces fixed-zero when zero-time is implicit.",
        },
        {
            "check_id": "no_zerotime_preset_reusable",
            "status": "ok" if no_zero_lane["zero_time_policy"] == "excluded" else "warn",
            "no_zerotime_policy": no_zero_lane["zero_time_policy"],
            "notes": "AT-005A lane reused as formal no-zero-time validation preset.",
        },
    ]
    _write_csv(tables_dir / "zero_time_policy_checks.csv", checks_rows)

    summary = {
        "artifact_id": "AT-006",
        "task_id": "AT-006_zero_time_policy_fix",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": package["dataset_name"],
        "dataset_shape": package["dataset_shape"],
        "dataset_hash": package["dataset_hash"],
        "dataset_kind": _dataset_kind(package),
        "time_step_ns": dt_ns,
        "zero_time_policy": zero_time_policy,
        "legacy_default_behavior": {
            "new_zero_time_param": None,
            "effective_default_new_zero_time_ns": 5.0,
            "shift_samples": legacy["shift_samples"],
            "shift_ns": legacy["shift_ns"],
            "global_energy_ratio": legacy["global_energy_ratio"],
            "runtime_warnings": legacy["runtime_warnings"],
        },
        "guarded_behavior": {
            "policy": zero_time_policy,
            "effective_new_zero_time_ns": 0.0,
            "shift_samples": guarded["shift_samples"],
            "shift_ns": guarded["shift_ns"],
            "global_energy_ratio": guarded["global_energy_ratio"],
            "runtime_warnings": guarded["runtime_warnings"],
            "policy_notes": guarded["policy_notes"],
        },
        "no_zerotime_preset": {
            "reused_from": "AT-005A",
            "zero_time_policy": no_zero_lane["zero_time_policy"],
            "report_markdown": str(no_zero_lane["report_markdown"]),
            "report_html": str(no_zero_lane["report_html"]),
        },
        "root_cause": (
            "Validation branches that call set_zero_time with empty params fall back to method default "
            "new_zero_time=5.0 ns. On GX-003 dt this maps to ~423 samples and can collapse global energy."
        ),
        "fix": (
            "Validation runners now infer native_gprmax_converted context and apply zero_time_policy="
            "explicit_only_fixed_zero, forcing missing zero-time params to 0.0 ns and disabling implicit tuning."
        ),
        "recommended_next": "Rerun AT-002 and AT-003 under the guarded policy, preserving prior conclusions as historical evidence.",
        "limitations": [
            "This task hardens validation policy only; it does not redesign set_zero_time algorithm behavior globally.",
            "No claim of overall AutoTune superiority is made here.",
            "Ground truth ROI is unchanged from GX-003 and not replaced.",
        ],
    }
    _write_json(manifests_dir / "zero_time_policy_summary.json", summary)

    evidence_manifest = {
        "artifact_id": "AT-006",
        "task_id": "AT-006_zero_time_policy_fix",
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "evidence_repo": "https://github.com/CYberkra/MyGPR-Evidence",
        "evidence_commit": "pending_self_reference",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/auto_tune_validation/run_zero_time_policy_fix.py",
        "dataset_name": package["dataset_name"],
        "dataset_shape": package["dataset_shape"],
        "dataset_hash": package["dataset_hash"],
        "ground_truth_available": bool(package.get("ground_truth")),
        "metric_type": "ground_truth",
        "artifacts": {
            "summary": "manifests/zero_time_policy_summary.json",
            "checks_csv": "tables/zero_time_policy_checks.csv",
            "figure": "figures/zero_time_policy_before_after.png",
            "markdown_report": "reports/zero_time_policy_fix_report.md",
            "html_report": "reports/zero_time_policy_fix_report.html",
        },
        "limitations": summary["limitations"],
        "known_risks": [
            "User workflows outside validation runners still use configured set_zero_time behavior.",
            "Policy inference relies on dataset metadata and may need explicit override for uncommon inputs.",
        ],
    }
    _write_json(manifests_dir / "evidence_manifest.json", evidence_manifest)

    markdown = _render_markdown(summary, checks_rows)
    html = _render_html(summary, checks_rows)
    (reports_dir / "zero_time_policy_fix_report.md").write_text(markdown, encoding="utf-8")
    (reports_dir / "zero_time_policy_fix_report.html").write_text(html, encoding="utf-8")

    return {
        "evidence_root": str(evidence_root.resolve()),
        "source_commit": source_commit,
        "dataset_name": package["dataset_name"],
        "zero_time_policy": zero_time_policy,
        "legacy_shift_samples": legacy["shift_samples"],
        "guarded_shift_samples": guarded["shift_samples"],
        "report_markdown": str((reports_dir / "zero_time_policy_fix_report.md").resolve()),
        "report_html": str((reports_dir / "zero_time_policy_fix_report.html").resolve()),
    }


def _run_legacy_default_zero_time(
    raw: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
) -> dict[str, Any]:
    runtime_params = prepare_runtime_params(
        "set_zero_time",
        {},
        clone_header_info(header_info),
        clone_trace_metadata(trace_metadata),
        raw.shape,
    )
    after, meta = run_processing_method(raw, "set_zero_time", runtime_params)
    shift_samples = int((meta or {}).get("shift_samples") or 0)
    step_s = float((meta or {}).get("time_step_s") or 0.0)
    return {
        "data_after": after,
        "shift_samples": shift_samples,
        "shift_ns": float(shift_samples * step_s * 1.0e9) if step_s > 0.0 else 0.0,
        "global_energy_ratio": float(_energy_ratio(raw, after)),
        "runtime_warnings": _runtime_warning_codes(meta),
    }


def _run_guarded_fixed_zero(
    raw: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    package: dict[str, Any],
    preview_dir: Path,
) -> dict[str, Any]:
    policy = _infer_zero_time_policy(package)
    preview_dir.mkdir(parents=True, exist_ok=True)
    branch = _run_branch(
        branch="guarded_policy_check",
        raw=raw,
        header_info=header_info,
        trace_metadata=trace_metadata,
        ground_truth=package.get("ground_truth"),
        figures_dir=preview_dir,
        auto_tune=False,
        search_mode="fast",
        pipeline=["set_zero_time"],
        manual_params={},
        tune_methods=None,
        zero_time_policy=policy,
    )
    step = branch["steps"][0] if branch["steps"] else {}
    meta = step.get("result_meta", {}) if isinstance(step, dict) else {}
    shift_samples = int(meta.get("shift_samples") or 0)
    step_s = float(meta.get("time_step_s") or 0.0)
    return {
        "data_after": branch["result"],
        "shift_samples": shift_samples,
        "shift_ns": float(shift_samples * step_s * 1.0e9) if step_s > 0.0 else 0.0,
        "global_energy_ratio": float(_energy_ratio(raw, branch["result"])),
        "runtime_warnings": step.get("runtime_warnings", []),
        "policy_notes": step.get("zero_time_policy_notes", []),
    }


def _run_no_zero_lane(
    evidence_root: Path,
    dataset: str,
    source_repo: str,
    source_branch: str,
    source_commit: str,
) -> dict[str, Any]:
    no_zero_root = evidence_root / "_no_zerotime_lane"
    result = run_no_zerotime_validation(
        evidence_root=no_zero_root,
        dataset=dataset,
        source_repo=source_repo,
        source_branch=source_branch,
        source_commit=source_commit,
        include_field=False,
    )
    summary = json.loads((no_zero_root / "manifests" / "validation_summary.json").read_text(encoding="utf-8"))
    return {
        "result": result,
        "zero_time_policy": str(summary.get("zero_time_policy") or ""),
        "report_markdown": no_zero_root / "reports" / "no_zerotime_gain_validation_report.md",
        "report_html": no_zero_root / "reports" / "no_zerotime_gain_validation_report.html",
    }


def _dataset_kind(package: dict[str, Any]) -> str:
    scenario = package.get("scenario")
    if isinstance(scenario, dict):
        source = scenario.get("source")
        if isinstance(source, dict):
            value = source.get("kind")
            if value:
                return str(value)
    return "unknown"


def _runtime_warning_codes(meta: dict[str, Any] | None) -> list[str]:
    warnings = (meta or {}).get("runtime_warnings")
    if not isinstance(warnings, list):
        return []
    codes: list[str] = []
    for item in warnings:
        if isinstance(item, dict):
            code = item.get("code")
            if code:
                codes.append(str(code))
    return codes


def _save_before_after_figure(
    *,
    raw: np.ndarray,
    legacy: np.ndarray,
    guarded: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    stacked = np.concatenate([raw.ravel(), legacy.ravel(), guarded.ravel()])
    vmax = float(np.percentile(np.abs(stacked), 99.0))
    vmax = max(vmax, 1.0e-6)
    vmin = -vmax
    for ax, arr, title in (
        (axes[0], raw, "Input (raw)"),
        (axes[1], legacy, "Legacy default set_zero_time"),
        (axes[2], guarded, "Guarded set_zero_time (fixed 0)"),
    ):
        im = ax.imshow(arr, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _render_markdown(summary: dict[str, Any], checks_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# AT-006 Zero-Time Policy Fix",
        "",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Dataset: `{summary['dataset_name']}` (`{summary['dataset_kind']}`)",
        f"- Dataset shape: `{summary['dataset_shape']}`",
        f"- Time step: `{summary['time_step_ns']:.6f} ns`",
        f"- Zero-time policy: `{summary['zero_time_policy']}`",
        "",
        "## Root Cause",
        summary["root_cause"],
        "",
        "## Fix Implemented",
        summary["fix"],
        "",
        "## Policy Checks",
        "",
        "| Check | Status | Notes |",
        "| --- | --- | --- |",
    ]
    for row in checks_rows:
        lines.append(
            f"| `{row.get('check_id')}` | `{row.get('status')}` | {row.get('notes', '')} |"
        )
    lines.extend(
        [
            "",
            "## Before / After",
            f"- Legacy shift samples: `{summary['legacy_default_behavior']['shift_samples']}`",
            f"- Legacy shift ns: `{summary['legacy_default_behavior']['shift_ns']:.6f}`",
            f"- Legacy global energy ratio: `{summary['legacy_default_behavior']['global_energy_ratio']:.6f}`",
            f"- Guarded shift samples: `{summary['guarded_behavior']['shift_samples']}`",
            f"- Guarded shift ns: `{summary['guarded_behavior']['shift_ns']:.6f}`",
            f"- Guarded global energy ratio: `{summary['guarded_behavior']['global_energy_ratio']:.6f}`",
            "",
            "## No-Zero-Time Preset",
            f"- Reused lane policy: `{summary['no_zerotime_preset']['zero_time_policy']}`",
            f"- Report (md): `{summary['no_zerotime_preset']['report_markdown']}`",
            f"- Report (html): `{summary['no_zerotime_preset']['report_html']}`",
            "",
            "## Claim Boundary",
            "- This task does not prove overall AutoTune superiority.",
            "- This task does not replace GX-003 ROI.",
            "- This task only hardens validation-time zero-time policy.",
            "",
            "## Next Step",
            summary["recommended_next"],
        ]
    )
    return "\n".join(lines) + "\n"


def _render_html(summary: dict[str, Any], checks_rows: list[dict[str, Any]]) -> str:
    table_rows = "\n".join(
        f"<tr><td><code>{row.get('check_id')}</code></td><td>{row.get('status')}</td><td>{row.get('notes','')}</td></tr>"
        for row in checks_rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AT-006 Zero-Time Policy Fix</title>
  <style>
    body {{ font-family: "Segoe UI", "PingFang SC", sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-top: 1.2em; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 14px 16px; margin: 12px 0; background: #f9fafb; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
    .warn {{ color: #b45309; }}
  </style>
</head>
<body>
  <h1>AT-006 Zero-Time Policy Fix</h1>
  <p>Source commit: <code>{summary["source_commit"]}</code></p>
  <p>Dataset: <code>{summary["dataset_name"]}</code> (<code>{summary["dataset_kind"]}</code>) shape <code>{summary["dataset_shape"]}</code></p>
  <p>Zero-time policy: <code>{summary["zero_time_policy"]}</code></p>

  <h2>Root Cause</h2>
  <div class="card">{summary["root_cause"]}</div>

  <h2>Fix</h2>
  <div class="card">{summary["fix"]}</div>

  <h2>Policy Checks</h2>
  <table>
    <thead><tr><th>Check</th><th>Status</th><th>Notes</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>

  <h2>Before / After</h2>
  <div class="card">
    <p>Legacy shift samples: <code>{summary["legacy_default_behavior"]["shift_samples"]}</code>,
       shift ns: <code>{summary["legacy_default_behavior"]["shift_ns"]:.6f}</code>,
       global energy ratio: <code>{summary["legacy_default_behavior"]["global_energy_ratio"]:.6f}</code></p>
    <p>Guarded shift samples: <code>{summary["guarded_behavior"]["shift_samples"]}</code>,
       shift ns: <code>{summary["guarded_behavior"]["shift_ns"]:.6f}</code>,
       global energy ratio: <code>{summary["guarded_behavior"]["global_energy_ratio"]:.6f}</code></p>
    <p class="warn">This task does not claim overall AutoTune superiority.</p>
  </div>

  <h2>No-Zero-Time Preset</h2>
  <div class="card">
    <p>Policy: <code>{summary["no_zerotime_preset"]["zero_time_policy"]}</code></p>
    <p>Markdown report: <code>{summary["no_zerotime_preset"]["report_markdown"]}</code></p>
    <p>HTML report: <code>{summary["no_zerotime_preset"]["report_html"]}</code></p>
  </div>

  <h2>Next Step</h2>
  <div class="card">{summary["recommended_next"]}</div>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
