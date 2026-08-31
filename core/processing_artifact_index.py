#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Index processed artifacts produced by the field workbench.

The target-positioning, spatial-results and report pages should resolve saved
processing outputs from the project artifact files rather than from transient UI
state.  This module is intentionally lightweight and read-only except for helper
serialization in tests.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from core.project_catalog import ProjectCatalog
from core.storage_uri import make_h5_uri


@dataclass(frozen=True)
class ProcessingArtifactRecord:
    artifact_id: str
    line_id: str
    data_path: str
    params_path: str = ""
    manifest_path: str = ""
    method_id: str = ""
    method_name: str = ""
    category: str = ""
    status: str = "unknown"
    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()
    sample_count_changed: bool = False
    trace_count_changed: bool = False
    created_at: str = ""
    role: str = "processing_result"
    axis_transform: dict[str, Any] | None = None
    warnings: list[Any] | None = None
    output_data_sha256: str = ""
    params_sha256: str = ""
    manifest_sha256: str = ""
    save_schema: str = ""
    branch_id: str = ""
    parent_artifact_id: str = ""

    @property
    def shape_changed(self) -> bool:
        return bool(self.sample_count_changed or self.trace_count_changed)

    @property
    def is_display_compare_transform(self) -> bool:
        return self.role == "display_compare_transform"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["output_shape"] = list(self.output_shape)
        payload["warnings"] = list(self.warnings or [])
        return payload


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return {}


def _coerce_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError, OverflowError):
                return ()
        return tuple(out)
    return ()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _artifact_id_from_data_path(path: Path) -> str:
    name = path.stem
    # Example: L03_processed_20260609_124000 -> L03_processed_20260609_124000
    return name


def _record_from_paths(root: Path, line_id: str, data_path: Path, params_path: Path | None, manifest_path: Path | None) -> ProcessingArtifactRecord:
    params_payload = _safe_load_json(params_path) if params_path and params_path.exists() else {}
    nested_manifest = params_payload.get("manifest") if isinstance(params_payload.get("manifest"), dict) else {}
    manifest_payload = _safe_load_json(manifest_path) if manifest_path and manifest_path.exists() else {}
    manifest = {**nested_manifest, **manifest_payload}
    method_id = str(manifest.get("method_id") or params_payload.get("method") or "")
    method_name = str(manifest.get("method_name") or params_payload.get("method_name") or method_id)
    role = str(manifest.get("artifact_role") or manifest.get("role") or "processing_result")
    axis_transform = manifest.get("axis_transform") if isinstance(manifest.get("axis_transform"), dict) else None
    return ProcessingArtifactRecord(
        artifact_id=str(manifest.get("artifact_id") or _artifact_id_from_data_path(data_path)),
        line_id=str(manifest.get("line_id") or line_id),
        data_path=_relative_to_root(data_path, root),
        params_path=_relative_to_root(params_path, root) if params_path else "",
        manifest_path=_relative_to_root(manifest_path, root) if manifest_path else "",
        method_id=method_id,
        method_name=method_name,
        category=str(manifest.get("category") or ""),
        status=str(manifest.get("status") or "unknown"),
        input_shape=_coerce_shape(manifest.get("input_shape")),
        output_shape=_coerce_shape(manifest.get("output_shape")),
        sample_count_changed=bool(manifest.get("sample_count_changed", False)),
        trace_count_changed=bool(manifest.get("trace_count_changed", False)),
        created_at=str(manifest.get("created_at") or params_payload.get("updated_at") or manifest.get("saved_at") or ""),
        role=role,
        axis_transform=axis_transform,
        warnings=list(manifest.get("warnings") or []),
        output_data_sha256=str(manifest.get("output_data_sha256") or params_payload.get("output_data_sha256") or ""),
        params_sha256=str(manifest.get("params_sha256") or params_payload.get("params_sha256") or ""),
        manifest_sha256=str(manifest.get("manifest_sha256") or params_payload.get("manifest_sha256") or ""),
        save_schema=str(manifest.get("save_schema") or params_payload.get("schema") or ""),
        branch_id=str(manifest.get("branch_id") or params_payload.get("branch_id") or ""),
        parent_artifact_id=str(manifest.get("parent_artifact_id") or params_payload.get("parent_artifact_id") or ""),
    )




def _index_catalog_artifacts(root: Path, line_id: str | None = None) -> list[ProcessingArtifactRecord]:
    catalog_path = root / "catalog.sqlite"
    if not catalog_path.exists():
        return []
    try:
        rows = ProjectCatalog(catalog_path, read_only=True).list_artifacts(
            line_id=line_id, artifact_kind="processing"
        )
    except (sqlite3.Error, OSError, TypeError, ValueError):
        return []
    records: list[ProcessingArtifactRecord] = []
    for row in rows:
        manifest = row.get("manifest") if isinstance(row.get("manifest"), dict) else {}
        params = row.get("params") if isinstance(row.get("params"), dict) else {}
        h5_path = str(row.get("h5_path") or "")
        dataset_path = str(row.get("dataset_path") or "")
        records.append(ProcessingArtifactRecord(
            artifact_id=str(row.get("artifact_id") or ""),
            line_id=str(row.get("line_id") or ""),
            data_path=make_h5_uri(h5_path, dataset_path),
            params_path=str(manifest.get("params_path") or ""),
            manifest_path=str(manifest.get("manifest_path") or ""),
            method_id=str(manifest.get("method_id") or params.get("method") or ""),
            method_name=str(manifest.get("method_name") or manifest.get("method_id") or ""),
            category=str(manifest.get("category") or ""),
            status=str(row.get("status") or "committed"),
            input_shape=_coerce_shape(manifest.get("input_shape")),
            output_shape=_coerce_shape(row.get("shape") or manifest.get("output_shape")),
            sample_count_changed=bool(manifest.get("sample_count_changed", False)),
            trace_count_changed=bool(manifest.get("trace_count_changed", False)),
            created_at=str(row.get("created_at") or manifest.get("saved_at") or ""),
            role=str(row.get("artifact_role") or manifest.get("artifact_role") or "processing_result"),
            axis_transform=manifest.get("axis_transform") if isinstance(manifest.get("axis_transform"), dict) else None,
            warnings=list(manifest.get("warnings") or []),
            output_data_sha256=str(row.get("sha256") or manifest.get("output_data_sha256") or ""),
            params_sha256=str(manifest.get("params_sha256") or ""),
            manifest_sha256=str(manifest.get("manifest_sha256") or ""),
            save_schema=str(manifest.get("save_schema") or "mygpr.processing_save.v3"),
            branch_id=str(row.get("branch_id") or manifest.get("branch_id") or ""),
            parent_artifact_id=str(row.get("parent_artifact_id") or manifest.get("parent_artifact_id") or ""),
        ))
    return records

def index_processing_artifacts(project_root: str | Path, line_id: str | None = None) -> list[ProcessingArtifactRecord]:
    """Return saved processing artifacts for a field project."""
    root = Path(project_root).resolve()
    catalog_records = _index_catalog_artifacts(root, line_id=line_id)
    if catalog_records:
        return catalog_records
    processed_root = root / "processed"
    if not processed_root.exists():
        return []
    line_dirs: Iterable[Path]
    if line_id:
        line_dirs = [processed_root / line_id]
    else:
        line_dirs = [p for p in processed_root.iterdir() if p.is_dir()]
    records: list[ProcessingArtifactRecord] = []
    for line_dir in line_dirs:
        if not line_dir.exists() or not line_dir.is_dir():
            continue
        current_line_id = line_dir.name
        params_candidates = sorted(line_dir.glob(f"{current_line_id}_params*.json"))
        legacy_latest_params = line_dir / f"{current_line_id}_params.json"
        default_params = legacy_latest_params if legacy_latest_params.exists() else (params_candidates[-1] if params_candidates else None)
        manifests = sorted(line_dir.glob(f"{current_line_id}_processing_manifest*.json"))
        for data_path in sorted(line_dir.glob(f"{current_line_id}_processed_*.npy")):
            # Prefer sidecars sharing the data timestamp, then fall back to latest/legacy files.
            timestamp = data_path.stem.replace(f"{current_line_id}_processed_", "", 1)
            matched_manifest = line_dir / f"{current_line_id}_processing_manifest_{timestamp}.json"
            manifest_path = matched_manifest if matched_manifest.exists() else (manifests[-1] if manifests else None)
            matched_params = line_dir / f"{current_line_id}_params_{timestamp}.json"
            params_path = matched_params if matched_params.exists() else default_params
            records.append(_record_from_paths(root, current_line_id, data_path, params_path, manifest_path))
    records.sort(key=lambda item: (item.line_id, item.artifact_id, item.created_at), reverse=True)
    return records


def latest_processing_artifact(project_root: str | Path, line_id: str) -> ProcessingArtifactRecord | None:
    records = index_processing_artifacts(project_root, line_id=line_id)
    return records[0] if records else None


__all__ = [
    "ProcessingArtifactRecord",
    "index_processing_artifacts",
    "latest_processing_artifact",
]
