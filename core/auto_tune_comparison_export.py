#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research artifact export for manual-baseline vs auto-tune comparisons."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.auto_tune_comparison import AutoTuneComparisonRun, to_summary_dict


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
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_bundle_name(bundle_name)

    paths = {
        "summary_json": output_root / f"{safe_name}_comparison_summary.json",
        "manual_png": output_root / f"{safe_name}_manual_bscan.png",
        "auto_png": output_root / f"{safe_name}_auto_bscan.png",
        "side_by_side_png": output_root / f"{safe_name}_side_by_side.png",
        "params_csv": output_root / f"{safe_name}_params_table.csv",
        "metrics_csv": output_root / f"{safe_name}_metrics_table.csv",
        "report_md": output_root / f"{safe_name}_comparison_report.md",
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
    summary["artifacts"] = {
        key: str(path.resolve()) for key, path in paths.items()
    }
    summary["exported_at"] = datetime.now().isoformat(timespec="seconds")

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
    paths["report_md"].write_text(
        _build_report_markdown(summary),
        encoding="utf-8",
    )
    paths["summary_json"].write_text(
        json.dumps(
            _json_safe(summary),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return _json_safe(
        {
            "bundle_name": safe_name,
            "output_dir": str(output_root.resolve()),
            "artifacts": {
                key: str(path.resolve()) for key, path in paths.items()
            },
            "summary": summary,
        }
    )


def _safe_bundle_name(bundle_name: str | None) -> str:
    raw = str(bundle_name or datetime.now().strftime("auto_tune_comparison_%Y%m%d_%H%M%S"))
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw).strip("._-")
    return safe or "auto_tune_comparison"


def _locked_display_spec(
    manual_arr: np.ndarray,
    auto_arr: np.ndarray,
    source_spec: dict[str, Any] | None,
    *,
    cmap: str,
) -> dict[str, Any]:
    source = dict(source_spec or {})
    clip = source.get("percentile_clip")
    combined = np.concatenate(
        [
            np.ravel(np.asarray(manual_arr, dtype=np.float32)),
            np.ravel(np.asarray(auto_arr, dtype=np.float32)),
        ]
    )
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        limit = 1.0
    elif clip is not None:
        percentile = max(0.0, min(float(clip), 100.0))
        limit = float(np.percentile(np.abs(finite), percentile))
    else:
        limit = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    return {
        "locked_scale": True,
        "lock_color_scale": True,
        "normalize": False,
        "percentile_clip": clip,
        "cmap": str(cmap or "gray"),
        "vmin": -limit,
        "vmax": limit,
    }


def _save_single_bscan(
    data: np.ndarray,
    out_path: Path,
    *,
    title: str,
    display_spec: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=150)
    image = ax.imshow(
        np.asarray(data, dtype=np.float32),
        cmap=str(display_spec["cmap"]),
        aspect="auto",
        vmin=float(display_spec["vmin"]),
        vmax=float(display_spec["vmax"]),
    )
    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Sample")
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_side_by_side(
    manual_data: np.ndarray,
    auto_data: np.ndarray,
    out_path: Path,
    *,
    display_spec: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=150, constrained_layout=True)
    for ax, arr, title in [
        (axes[0], manual_data, "Manual baseline"),
        (axes[1], auto_data, "Auto-tuned"),
    ]:
        image = ax.imshow(
            np.asarray(arr, dtype=np.float32),
            cmap=str(display_spec["cmap"]),
            aspect="auto",
            vmin=float(display_spec["vmin"]),
            vmax=float(display_spec["vmax"]),
        )
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
    fig.savefig(out_path)
    plt.close(fig)


def _build_params_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_key, candidate_label in [
        ("manual", "manual"),
        ("automatic", "automatic"),
    ]:
        candidate = summary.get(candidate_key) or {}
        params_by_method = candidate.get("params_by_method") or {}
        pipeline = list(candidate.get("pipeline") or params_by_method.keys())
        source = str(candidate.get("source") or candidate_label)
        for stage_index, method_key in enumerate(pipeline, start=1):
            params = params_by_method.get(method_key) or {}
            if not params:
                rows.append(
                    {
                        "candidate": candidate_label,
                        "source": source,
                        "method_key": method_key,
                        "stage_index": stage_index,
                        "param_name": "",
                        "param_value": "",
                    }
                )
                continue
            for param_name, value in sorted(params.items()):
                rows.append(
                    {
                        "candidate": candidate_label,
                        "source": source,
                        "method_key": method_key,
                        "stage_index": stage_index,
                        "param_name": param_name,
                        "param_value": _csv_value(value),
                    }
                )
    return rows


def _build_metrics_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    manual_metrics = ((summary.get("manual") or {}).get("metrics") or {})
    auto_metrics = ((summary.get("automatic") or {}).get("metrics") or {})
    metric_delta = summary.get("metric_delta") or {}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(manual_metrics) | set(auto_metrics)):
        rows.append(
            {
                "metric": str(key),
                "manual_value": _csv_value(manual_metrics.get(key)),
                "auto_value": _csv_value(auto_metrics.get(key)),
                "delta": _csv_value(metric_delta.get(key)),
            }
        )
    return rows


def _write_csv_rows(
    out_path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _build_report_markdown(summary: dict[str, Any]) -> str:
    manual = summary.get("manual") or {}
    automatic = summary.get("automatic") or {}
    delta = summary.get("metric_delta") or {}
    roi = summary.get("roi_info") or {}
    artifacts = summary.get("artifacts") or {}
    notes = [str(item) for item in (summary.get("notes") or [])]
    lines = [
        "# Auto-Tune Comparison Report",
        "",
        f"- Input: {summary.get('input_ref') or '--'}",
        f"- Verdict: {summary.get('verdict') or '--'}",
        f"- ROI: {roi.get('label') or roi.get('source') or '--'}",
        f"- Baseline profile: {summary.get('baseline_profile_key') or '--'}",
        f"- Manual score: {_csv_value((manual.get('metrics') or {}).get('comparison_score'))}",
        f"- Auto score: {_csv_value((automatic.get('metrics') or {}).get('comparison_score'))}",
        f"- Delta score: {_csv_value(delta.get('comparison_score'))}",
        "",
        "## Pipeline",
        "",
        " -> ".join(str(item) for item in (manual.get("pipeline") or [])) or "--",
        "",
        "## Artifacts",
        "",
    ]
    for key, path in artifacts.items():
        lines.append(f"- {key}: `{path}`")
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Research Boundary",
            "",
            "This export proves that manual and automatic branches were compared on the same input, ROI, and locked display scale. It does not by itself prove real-field target preservation; GPRMAX forward-model cases and field data should be used as validation datasets.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_value(value: Any) -> str:
    safe = _json_safe(value)
    if isinstance(safe, (dict, list)):
        return json.dumps(safe, ensure_ascii=False, sort_keys=True)
    if safe is None:
        return ""
    return str(safe)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.floating):
        parsed = float(value.item())
        return parsed if np.isfinite(parsed) else None
    if isinstance(value, np.integer):
        return value.item()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, int):
        return int(value)
    return str(value)


__all__ = ["export_auto_tune_comparison_artifacts"]
