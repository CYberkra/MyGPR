#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only research dashboard state for gprMax and AT-BG evidence."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "research_dashboard_defaults.json"


@dataclass
class ArtifactSummary:
    """Normalized read-only artifact status for the research console."""

    artifact_id: str
    artifact_role: str = ""
    display_name: str = ""
    scene_id: str = ""
    status: str = "missing"
    evidence_path: str = ""
    manifest_path: str = ""
    report_path: str = ""
    metrics_path: str = ""
    preview_paths: list[str] = field(default_factory=list)
    raw_shape: list[int] | None = None
    background_shape: list[int] | None = None
    target_response_shape: list[int] | None = None
    backend: str = ""
    source_commit: str = ""
    selected_trial_id: str = ""
    selected_method: str = ""
    selected_parameters: dict[str, Any] | None = None
    trial_count: int | None = None
    claim_boundary: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended_next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_role": self.artifact_role,
            "display_name": self.display_name,
            "scene_id": self.scene_id,
            "status": self.status,
            "evidence_path": self.evidence_path,
            "manifest_path": self.manifest_path,
            "report_path": self.report_path,
            "metrics_path": self.metrics_path,
            "preview_paths": self.preview_paths,
            "raw_shape": self.raw_shape,
            "background_shape": self.background_shape,
            "target_response_shape": self.target_response_shape,
            "backend": self.backend,
            "source_commit": self.source_commit,
            "selected_trial_id": self.selected_trial_id,
            "selected_method": self.selected_method,
            "selected_parameters": self.selected_parameters,
            "trial_count": self.trial_count,
            "claim_boundary": self.claim_boundary,
            "warnings": self.warnings,
            "recommended_next_action": self.recommended_next_action,
        }


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"missing json: {path}")
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"malformed json: {path}: {exc}")
        return {}
    if not isinstance(loaded, dict):
        warnings.append(f"json is not an object: {path}")
        return {}
    return loaded


def _read_json_list(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        warnings.append(f"missing json: {path}")
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"malformed json: {path}: {exc}")
        return []
    if isinstance(loaded, list):
        return [row for row in loaded if isinstance(row, dict)]
    warnings.append(f"json is not a list: {path}")
    return []


def _first_existing(base: Path, candidates: list[str]) -> str:
    for relative in candidates:
        path = base / relative
        if path.exists():
            return str(path)
    return ""


def load_dashboard_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load dashboard config, returning a safe empty config on invalid input."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    warnings: list[str] = []
    config = _read_json(path, warnings)
    config.setdefault("warnings", warnings)
    config.setdefault("evidence_root_candidates", [])
    config.setdefault("gprmax_artifacts", [])
    config.setdefault("at_bg_artifacts", [])
    config.setdefault("draft_scenes", [])
    return config


def resolve_evidence_root(config: dict[str, Any] | None = None) -> tuple[Path | None, list[str]]:
    """Resolve the first existing Evidence root from dashboard config."""
    config = config or load_dashboard_config()
    warnings: list[str] = []
    for candidate in config.get("evidence_root_candidates", []):
        path = Path(candidate)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if path.exists():
            return path, warnings
        warnings.append(f"evidence root candidate missing: {path}")
    return None, warnings or ["no evidence root candidates configured"]


def load_gprmax_artifact(path: str | Path, display_name: str = "", scene_id: str = "") -> ArtifactSummary:
    """Load a curated gprMax paired artifact summary without writing files."""
    base = Path(path)
    warnings: list[str] = []
    summary = ArtifactSummary(
        artifact_id=base.name,
        display_name=display_name or base.name,
        scene_id=scene_id,
        evidence_path=str(base),
        warnings=warnings,
        recommended_next_action="GX-008 standard metrics or AT-BG diagnostic follow-up",
    )
    if not base.exists():
        warnings.append(f"artifact missing: {base}")
        return summary

    manifest_path = base / "manifests" / "evidence_manifest.json"
    manifest = _read_json(manifest_path, warnings)
    summary.manifest_path = str(manifest_path) if manifest_path.exists() else ""
    summary.artifact_id = str(manifest.get("artifact_id") or summary.artifact_id)
    summary.artifact_role = str(manifest.get("artifact_role") or "")
    summary.scene_id = str(manifest.get("scene_id") or summary.scene_id)
    summary.raw_shape = manifest.get("raw_shape") if isinstance(manifest.get("raw_shape"), list) else None
    summary.background_shape = (
        manifest.get("background_shape") if isinstance(manifest.get("background_shape"), list) else None
    )
    summary.target_response_shape = (
        manifest.get("target_response_shape") if isinstance(manifest.get("target_response_shape"), list) else None
    )
    summary.backend = str(manifest.get("run_backend_final") or "")
    summary.source_commit = str(manifest.get("source_commit") or "")
    claim = manifest.get("claim_boundary")
    summary.claim_boundary = claim if isinstance(claim, list) else []

    summary.metrics_path = _first_existing(
        base,
        [
            "tables/standard_paired_metrics.json",
            "tables/paired_metrics.json",
        ],
    )
    summary.report_path = _first_existing(
        base,
        [
            "reports/standard_metrics_report.md",
            "reports/paired_target_response_report.md",
            "reports/gx008_scene003_paired_diagnostic_report.md",
            "reports/gx008_scene002_paired_diagnostic_report.md",
            "reports/gx008_scene001_paired_diagnostic_report.md",
        ],
    )
    summary.preview_paths = [
        str(base / relative)
        for relative in [
            "figures/raw_preview.png",
            "figures/background_preview.png",
            "figures/target_response_preview.png",
            "figures/paired_preview_panel.png",
        ]
        if (base / relative).exists()
    ]
    if not summary.preview_paths:
        warnings.append(f"preview files missing under: {base / 'figures'}")

    has_metrics = bool((base / "tables" / "standard_paired_metrics.json").exists())
    summary.status = "complete" if summary.manifest_path and manifest else "partial"
    if summary.status == "complete" and not has_metrics:
        summary.status = "evidence_done_metrics_pending"
        summary.recommended_next_action = "Apply standardized metrics before AT-BG diagnostic"
    return summary


def load_at_bg_artifact(path: str | Path, display_name: str = "") -> ArtifactSummary:
    """Load a curated AT-BG diagnostic artifact summary without writing files."""
    base = Path(path)
    warnings: list[str] = []
    summary = ArtifactSummary(
        artifact_id=base.name,
        display_name=display_name or base.name,
        status="missing",
        evidence_path=str(base),
        warnings=warnings,
        recommended_next_action="Compare with more GX-008 scenes",
    )
    if not base.exists():
        warnings.append(f"artifact missing: {base}")
        return summary

    manifest_path = base / "manifests" / "evidence_manifest.json"
    manifest = _read_json(manifest_path, warnings)
    summary.manifest_path = str(manifest_path) if manifest_path.exists() else ""
    summary.artifact_id = str(manifest.get("artifact_id") or summary.artifact_id)
    summary.artifact_role = str(manifest.get("artifact_role") or "")
    summary.scene_id = str(manifest.get("scene_id") or "")
    summary.source_commit = str(manifest.get("source_commit") or "")
    summary.trial_count = manifest.get("trial_count") if isinstance(manifest.get("trial_count"), int) else None
    summary.selected_trial_id = str(manifest.get("selected_trial_id") or "")
    claim = manifest.get("claim_boundary")
    summary.claim_boundary = claim if isinstance(claim, list) else []

    selected_path = base / "tables" / "selected_parameters.json"
    selected_doc = _read_json(selected_path, warnings) if selected_path.exists() else {}
    selected = selected_doc.get("selected") if isinstance(selected_doc.get("selected"), dict) else selected_doc
    if isinstance(selected, dict):
        summary.selected_trial_id = str(selected.get("trial_id") or summary.selected_trial_id)
        summary.selected_method = str(selected.get("method") or "")
        params = selected.get("parameter_set")
        summary.selected_parameters = params if isinstance(params, dict) else None

    if summary.trial_count is None:
        trial_table = _read_json_list(base / "tables" / "trial_table.json", warnings)
        if trial_table:
            summary.trial_count = len(trial_table)

    summary.report_path = _first_existing(
        base,
        [
            "reports/at_bg_004b_multi_scene_comparison_report.md",
            "reports/at_bg_004a_evidence_summary.md",
            "reports/at_bg_003_evidence_summary.md",
            "reports/background_suppression_autotune_report.md",
        ],
    )
    summary.metrics_path = _first_existing(
        base,
        [
            "tables/selected_parameter_comparison.json",
            "tables/method_rank_summary.json",
            "tables/trial_table.json",
        ],
    )
    summary.status = "complete" if summary.manifest_path and summary.metrics_path else "partial"
    return summary


def summarize_scene_status(gprmax: list[ArtifactSummary], at_bg: list[ArtifactSummary], draft_scenes: list[str]) -> list[dict[str, Any]]:
    """Build table-ready GX-008 scene status rows."""
    at_bg_by_scene = {item.scene_id: item for item in at_bg if item.scene_id}
    rows: list[dict[str, Any]] = []
    for item in gprmax:
        scene = item.scene_id
        at_bg_item = at_bg_by_scene.get(scene)
        rows.append(
            {
                "scene_id": scene,
                "display_name": item.display_name,
                "paired_evidence": "done" if item.status in {"complete", "evidence_done_metrics_pending"} else item.status,
                "standard_metrics": "done" if "standard_paired_metrics.json" in item.metrics_path else "pending",
                "at_bg": "done" if at_bg_item and at_bg_item.status == "complete" else "pending",
                "backend": item.backend or "-",
                "shape": item.target_response_shape or item.raw_shape,
                "claim": "synthetic paired diagnostic only",
                "warnings": item.warnings + ((at_bg_item.warnings if at_bg_item else []) if scene else []),
            }
        )
    for scene in draft_scenes:
        rows.append(
            {
                "scene_id": scene,
                "display_name": scene,
                "paired_evidence": "draft",
                "standard_metrics": "-",
                "at_bg": "-",
                "backend": "-",
                "shape": None,
                "claim": "draft / not run",
                "warnings": [],
            }
        )
    return rows


def summarize_at_bg_status(at_bg: list[ArtifactSummary]) -> dict[str, Any]:
    """Build compact AT-BG status for overview cards and claim text."""
    completed = [item for item in at_bg if item.status == "complete"]
    selected = [
        {
            "scene_id": item.scene_id,
            "artifact_id": item.artifact_id,
            "trial_id": item.selected_trial_id,
            "method": item.selected_method,
            "parameters": item.selected_parameters,
            "trial_count": item.trial_count,
        }
        for item in completed
        if item.selected_trial_id
    ]
    return {
        "completed_count": len(completed),
        "total_count": len(at_bg),
        "selected": selected,
        "claim_boundary": [
            "synthetic paired diagnostic only",
            "background suppression only",
            "not full AutoTune",
            "not production scoring",
            "not field validation",
            "not AutoTune superiority evidence",
        ],
    }


def _load_method_rank_summary(evidence_root: Path, config: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    for item in config.get("at_bg_artifacts", []):
        if item.get("id") == "AT-BG_multi_scene":
            return _read_json_list(evidence_root / item["path"] / "tables" / "method_rank_summary.json", warnings)
    return []


def _load_trial_rows(evidence_root: Path, config: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in config.get("at_bg_artifacts", []):
        path = evidence_root / item.get("path", "") / "tables" / "trial_table.json"
        for row in _read_json_list(path, warnings):
            row.setdefault("artifact_id", item.get("id", ""))
            rows.append(row)
    return rows


def load_dashboard_state(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load complete read-only research dashboard state."""
    config = load_dashboard_config(config_path)
    evidence_root, root_warnings = resolve_evidence_root(config)
    warnings = list(config.get("warnings", [])) + root_warnings
    if evidence_root is None:
        return {
            "evidence_root": "",
            "warnings": warnings,
            "gprmax_artifacts": [],
            "at_bg_artifacts": [],
            "scene_status": summarize_scene_status([], [], config.get("draft_scenes", [])),
            "at_bg_status": summarize_at_bg_status([]),
            "method_rank_summary": [],
            "trial_rows": [],
        }

    gprmax = [
        load_gprmax_artifact(
            evidence_root / item.get("path", ""),
            display_name=item.get("display_name", ""),
            scene_id=item.get("scene_id", ""),
        )
        for item in config.get("gprmax_artifacts", [])
    ]
    at_bg = [
        load_at_bg_artifact(
            evidence_root / item.get("path", ""),
            display_name=item.get("display_name", ""),
        )
        for item in config.get("at_bg_artifacts", [])
    ]
    for artifact in gprmax + at_bg:
        warnings.extend(artifact.warnings)

    return {
        "evidence_root": str(evidence_root),
        "warnings": warnings,
        "gprmax_artifacts": [item.to_dict() for item in gprmax],
        "at_bg_artifacts": [item.to_dict() for item in at_bg],
        "scene_status": summarize_scene_status(gprmax, at_bg, config.get("draft_scenes", [])),
        "at_bg_status": summarize_at_bg_status(at_bg),
        "method_rank_summary": _load_method_rank_summary(evidence_root, config, warnings),
        "trial_rows": _load_trial_rows(evidence_root, config, warnings),
    }


def read_csv_preview(path: str | Path, limit: int = 20) -> list[dict[str, str]]:
    """Read a small CSV preview for UI tables."""
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for _, row in zip(range(limit), reader)]
