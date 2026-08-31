#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized evidence-row builders and checksum utilities for report export."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from core.field_project_models import FieldLineRecord, validate_line_id
from core.storage_primitives import atomic_output_path
from core.tabular_security import safe_tabular_value

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)

def _atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: safe_tabular_value(row.get(key, "")) for key in fieldnames})

def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    except Exception:
        return 0

def _quality_report_to_row(line: FieldLineRecord, report: Any | None) -> dict[str, Any]:
    if report is None:
        return {
            "line_id": line.line_id,
            "line_name": line.name,
            "status": "未质检",
            "sample_count": "",
            "trace_count": "",
            "time_window_ns": "",
            "length_m": f"{float(line.length_m or 0.0):.3f}",
            "finite_ratio": "",
            "nan_ratio": "",
            "trajectory_points": "",
            "orientation": "",
            "orientation_message": "未生成质检报告",
            "issue_count": "",
        }
    return {
        "line_id": line.line_id,
        "line_name": line.name,
        "status": getattr(report, "status_label", "--"),
        "sample_count": int(getattr(report, "sample_count", 0)),
        "trace_count": int(getattr(report, "trace_count", 0)),
        "time_window_ns": f"{float(getattr(report, 'time_window_ns', 0.0)):.3f}",
        "length_m": f"{float(getattr(report, 'length_m', 0.0)):.3f}",
        "finite_ratio": f"{float(getattr(report, 'finite_ratio', 0.0)):.6f}",
        "nan_ratio": f"{float(getattr(report, 'nan_ratio', 0.0)):.6f}",
        "trajectory_points": int(getattr(report, "trajectory_points", 0)),
        "orientation": getattr(report, "orientation", ""),
        "orientation_message": getattr(report, "orientation_message", ""),
        "issue_count": len(getattr(report, "issues", []) or []),
    }

def _line_to_row(line: FieldLineRecord) -> dict[str, Any]:
    return {
        "line_id": line.line_id,
        "name": line.name,
        "length_m": f"{float(line.length_m or 0.0):.3f}",
        "data_quality": line.data_quality,
        "rtk_status": line.rtk_status,
        "sensor_sync_status": str(getattr(line, "sensor_sync_status", "") or ""),
        "processing_status": line.processing_status,
        "target_count": int(line.target_count or 0),
        "raw_rows": int(line.raw_rows or 0),
        "raw_size_mb": f"{float(line.raw_size_mb or 0.0):.3f}",
        "data_format": line.data_format,
        "gpr_dataset_path": line.gpr_dataset_path,
        "trajectory_path": line.trajectory_path,
        "processed_result": line.processed_result,
        "params_path": line.params_path,
        "updated_at": line.updated_at,
    }

def _target_to_row(line_id: str, target: dict[str, Any]) -> dict[str, Any]:
    safe_line_id = validate_line_id(line_id)
    return {
        "target_id": target.get("target_id") or target.get("name", ""),
        "line_id": safe_line_id,
        "distance_m": target.get("distance_m", target.get("mileage", "")),
        "depth_m": target.get("depth_m", target.get("depth", "")),
        "x": target.get("x", ""),
        "y": target.get("y", ""),
        "type": target.get("type", ""),
        "status": target.get("status", ""),
        "confidence": target.get("confidence", ""),
        "source_result_id": target.get("source_result_id", ""),
        "source_mode": target.get("source_mode", ""),
        "source_method_id": target.get("source_method_id", ""),
        "source_manifest_path": target.get("source_manifest_path", ""),
        "note": target.get("note", ""),
    }

def _interface_to_row(line_id: str, summary: dict[str, Any], spatial_path: Path, root: Path) -> dict[str, Any]:
    return {
        "line_id": validate_line_id(line_id),
        "status": summary.get("status", "not_started"),
        "version": summary.get("version", ""),
        "keypoint_count": summary.get("keypoint_count", 0),
        "coverage_ratio": f"{float(summary.get('coverage_ratio', 0.0)):.6f}",
        "judged_ratio": f"{float(summary.get('judged_ratio', 0.0)):.6f}",
        "clear_ratio": f"{float(summary.get('clear_ratio', 0.0)):.6f}",
        "weak_ratio": f"{float(summary.get('weak_ratio', 0.0)):.6f}",
        "ignore_ratio": f"{float(summary.get('ignore_ratio', 0.0)):.6f}",
        "no_interface_ratio": f"{float(summary.get('no_interface_ratio', 0.0)):.6f}",
        "spatial_curve_path": _safe_rel(spatial_path, root) if spatial_path.exists() else "",
        "spatial_row_count": _count_csv_rows(spatial_path),
    }

def _artifact_to_row(record: Any) -> dict[str, Any]:
    return {
        "artifact_id": record.artifact_id,
        "line_id": record.line_id,
        "method_id": record.method_id,
        "method_name": record.method_name,
        "role": record.role,
        "status": record.status,
        "input_shape": "×".join(str(v) for v in record.input_shape),
        "output_shape": "×".join(str(v) for v in record.output_shape),
        "shape_changed": "yes" if record.shape_changed else "no",
        "created_at": record.created_at,
        "data_path": record.data_path,
        "params_path": record.params_path,
        "manifest_path": record.manifest_path,
        "output_data_sha256": record.output_data_sha256,
        "params_sha256": record.params_sha256,
        "manifest_sha256": record.manifest_sha256,
        "save_schema": record.save_schema,
    }

def _sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024, cancel_checker=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            if cancel_checker is not None and cancel_checker():
                from core.job_manager import JobCancelled
                raise JobCancelled("报告生成已取消")
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def _sensor_sync_to_row(line: FieldLineRecord, root: Path) -> dict[str, Any]:
    manifest_path = str(getattr(line, "sensor_sync_manifest_path", "") or "")
    payload: dict[str, Any] = {}
    if manifest_path:
        path = root / manifest_path
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
    diagnostics = payload.get("diagnostics", payload.get("summary", {})) or {}
    rtk = diagnostics.get("rtk", {}) if isinstance(diagnostics, dict) else {}
    imu = diagnostics.get("imu", {}) if isinstance(diagnostics, dict) else {}
    alt = diagnostics.get("altimeter", {}) if isinstance(diagnostics, dict) else {}
    return {
        "line_id": line.line_id,
        "status": getattr(line, "sensor_sync_status", "") or line.rtk_status,
        "manifest_path": manifest_path,
        "trace_metadata_path": str(getattr(line, "trace_metadata_path", "") or ""),
        "rtk_coverage_ratio": rtk.get("coverage_ratio", diagnostics.get("rtk_coverage_ratio", "")),
        "rtk_fixed_ratio": rtk.get("fixed_solution_ratio", diagnostics.get("fixed_solution_ratio", "")),
        "rtk_max_residual_s": rtk.get("max_residual_s", ""),
        "imu_coverage_ratio": imu.get("coverage_ratio", diagnostics.get("imu_coverage_ratio", "")),
        "altimeter_coverage_ratio": alt.get("coverage_ratio", diagnostics.get("altimeter_coverage_ratio", "")),
        "gap_count": diagnostics.get("gap_count", ""),
        "jump_count": diagnostics.get("jump_count", ""),
        "warning_count": len(diagnostics.get("warnings", []) or []),
    }

def _load_gis_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "spatial" / "gis_layers.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = []
    for layer in payload.get("layers", []):
        rows.append({
            "layer_id": layer.get("layer_id", ""),
            "name": layer.get("name", ""),
            "kind": layer.get("kind", ""),
            "role": layer.get("role", ""),
            "crs": layer.get("crs", ""),
            "geometry_type": layer.get("geometry_type", ""),
            "visible": layer.get("visible", True),
            "opacity": layer.get("opacity", 1.0),
            "bounds": json.dumps(layer.get("bounds", []), ensure_ascii=False),
            "source_path": layer.get("source_path", ""),
            "is_dem": bool((layer.get("metadata") or {}).get("is_dem")),
            "imported_at": layer.get("imported_at", ""),
        })
    return rows

__all__ = ['_timestamp', '_safe_rel', '_atomic_write_csv', '_count_csv_rows', '_quality_report_to_row', '_line_to_row', '_target_to_row', '_interface_to_row', '_artifact_to_row', '_sha256_file', '_sensor_sync_to_row', '_load_gis_rows']
