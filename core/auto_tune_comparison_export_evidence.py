#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manifest, archive and CSV writers for comparison evidence bundles."""
from __future__ import annotations

import csv
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from core.auto_tune_comparison_export_common import _json_safe
from core.auto_tune_comparison_export_tables import _build_autotune_v1_evidence_context

def _build_evidence_manifest(
    summary: dict[str, Any],
    paths: dict[str, Path],
    *,
    output_root: Path,
    input_ref: str | None,
    notes: list[str],
) -> dict[str, Any]:
    ground_truth = summary.get("ground_truth_info") or {"enabled": False}
    source_paths = ground_truth.get("source_paths") or {}
    artifacts = {
        key: {
            "path": _relative_artifact_path(path, output_root),
            "status": "available" if path.exists() else "missing",
        }
        for key, path in paths.items()
    }
    warnings_list = _summary_warnings(summary)
    conversion_warnings = ground_truth.get("conversion_warnings") or []
    warnings_list.extend(str(item) for item in conversion_warnings)
    autotune_v1_context = _build_autotune_v1_evidence_context(summary)
    if autotune_v1_context.get("manual_review_required"):
        warnings_list.append("AutoTune V1 export requires manual review under the recorded scoring boundary.")
    return {
        "schema": "mygpr_autotune_evidence_v1",
        "exported_at": summary.get("exported_at"),
        "project": "MyGPR",
        "git_commit": _safe_git_commit(warnings_list),
        "input": {
            "input_file": input_ref,
            "manifest_file": source_paths.get("manifest_file"),
            "ground_truth_file": source_paths.get("ground_truth_file"),
        },
        "ground_truth": {
            "enabled": bool(ground_truth.get("enabled")),
            "scenario_id": ground_truth.get("scenario_id"),
            "target_count": int(ground_truth.get("target_count") or 0),
            "has_background_rois": bool(ground_truth.get("has_background_rois")),
            "conversion_warnings": _json_safe(conversion_warnings),
        },
        "workflow": {
            "pipeline": list(
                ((summary.get("manual") or {}).get("pipeline"))
                or ((summary.get("automatic") or {}).get("pipeline"))
                or []
            ),
            "baseline_profile_key": summary.get("baseline_profile_key"),
            "roi_info": _json_safe(summary.get("roi_info") or {}),
        },
        "autotune_v1": _json_safe(autotune_v1_context),
        "artifacts": artifacts,
        "warnings": _json_safe(warnings_list),
        "notes": [str(item) for item in notes],
    }

def _summary_warnings(summary: dict[str, Any]) -> list[str]:
    warnings_list: list[str] = []
    for branch in ("manual", "automatic"):
        payload = summary.get(branch) or {}
        warnings_list.extend(str(item) for item in payload.get("warnings", []) or [])
    return warnings_list

def _safe_git_commit(warnings_list: list[str]) -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        warnings_list.append(f"git commit unavailable: {exc}")
        return None
    if result.returncode != 0:
        warnings_list.append(
            "git commit unavailable: " + (result.stderr.strip() or str(result.returncode))
        )
        return None
    commit = result.stdout.strip()
    return commit or None

def _relative_artifact_path(path: Path, output_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_root.resolve()))
    except ValueError:
        return str(path.resolve())

def _write_evidence_zip(
    zip_path: Path,
    paths: dict[str, Path],
    output_root: Path,
) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, path in paths.items():
            if key == "evidence_zip" or not path.exists():
                continue
            zf.write(path, _relative_artifact_path(path, output_root))

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

__all__ = ['_build_evidence_manifest', '_summary_warnings', '_safe_git_commit', '_relative_artifact_path', '_write_evidence_zip', '_write_csv_rows']
