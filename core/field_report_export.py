#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal, auditable engineering report-package orchestration.

Stable data models, evidence-row builders and format renderers live in focused
modules.  This module retains the historical public API and coordinates the
transactional package-generation workflow.
"""
from __future__ import annotations

import csv
import json
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from core.field_project_models import local_now
from core.processing_artifact_index import index_processing_artifacts
from core.project_state_tracker import load_project_state
from core.report_export_models import REPORT_PACKAGE_SCHEMA, ReportPackageResult
from core.report_export_renderers import (
    _write_checksums, _write_html_report, _write_pdf_report,
    _write_report_figures, _write_xlsx_report,
)
from core.report_export_rows import (
    _artifact_to_row, _atomic_write_csv, _count_csv_rows, _interface_to_row, _line_to_row,
    _load_gis_rows, _quality_report_to_row, _safe_rel, _sensor_sync_to_row,
    _sha256_file, _target_to_row, _timestamp,
)
from core.reporting_model import ReportDocument, ReportSection, ReportSealer, ReportSnapshot
from core.source_file_registry import ensure_full_source_hashes, load_source_registry
from core.storage_primitives import atomic_write_text


def _software_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except OSError:
        from importlib.metadata import PackageNotFoundError, version as distribution_version
        try:
            return distribution_version("mygpr")
        except PackageNotFoundError:
            return "unknown"


def _interface_axes_for_line(
    store: Any, line_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    traces: list[float] = []
    samples: list[float] = []
    contract = store.root / "metadata" / "interpretations" / "interfaces" / f"{line_id}.json"
    if contract.exists():
        try:
            data = json.loads(contract.read_text(encoding="utf-8"))
            for point in data.get("points", []):
                traces.append(float(point["trace_index"]))
                samples.append(float(point["sample_index"]))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            traces.clear(); samples.clear()
    if len(traces) < 2:
        try:
            annotation = store.load_basal_interface_annotation(line_id, create=False)
        except (OSError, ValueError, TypeError, KeyError):
            annotation = None
        if annotation is not None:
            traces = [float(point.trace_index) for point in annotation.keypoints]
            samples = [float(point.sample_index) for point in annotation.keypoints]
    if len(traces) < 2:
        return None
    order = np.argsort(np.asarray(traces, dtype=float), kind="stable")
    trace_axis = np.asarray(traces, dtype=float)[order]
    sample_axis = np.asarray(samples, dtype=float)[order]
    trace_axis, unique_index = np.unique(trace_axis, return_index=True)
    sample_axis = sample_axis[unique_index]
    try:
        depth_axis = np.asarray(store.load_gpr_dataset(line_id).depth_axis_m, dtype=float)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if trace_axis.size < 2 or depth_axis.size < 2:
        return None
    return trace_axis, sample_axis, depth_axis


def _borehole_comparison_rows(
    store: Any,
    *,
    line_ids: set[str],
    threshold_m: float,
) -> list[dict[str, Any]]:
    """Build deterministic borehole/interface depth comparisons from project evidence."""
    source = store.root / "metadata" / "interpretations" / "boreholes.json"
    if not source.exists():
        legacy = store.root / "interpretations" / "boreholes.json"
        source = legacy if legacy.exists() else source
    if not source.exists():
        return []
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    rows: list[dict[str, Any]] = []
    interface_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
    for item in payload.get("boreholes", []):
        if not isinstance(item, dict):
            continue
        line_id = str(item.get("line_id") or "")
        if not line_id or line_id not in line_ids:
            continue
        axes = interface_cache.setdefault(
            line_id, _interface_axes_for_line(store, line_id)
        )
        trace_index = float(item.get("trace_index", -1.0))
        borehole_depth = float(item.get("basal_depth_m", float("nan")))
        sample_value = interpreted_depth = error = absolute_error = float("nan")
        status = "interface_unavailable"
        if axes is not None and np.isfinite(trace_index) and np.isfinite(borehole_depth):
            trace_axis, sample_axis, depth_axis = axes
            status = "trace_outside_interface"
            if trace_axis[0] <= trace_index <= trace_axis[-1]:
                sample_value = float(np.interp(trace_index, trace_axis, sample_axis))
                interpreted_depth = float(np.interp(
                    sample_value, np.arange(depth_axis.size, dtype=float), depth_axis
                ))
                error = interpreted_depth - borehole_depth
                absolute_error = abs(error)
                status = "passed" if absolute_error <= threshold_m else "failed"
        elif axes is not None:
            status = "invalid_borehole"
        rows.append({
            "borehole_id": str(item.get("borehole_id") or ""),
            "line_id": line_id,
            "trace_index": trace_index,
            "borehole_depth_m": borehole_depth,
            "interpreted_sample": sample_value,
            "interpreted_depth_m": interpreted_depth,
            "error_m": error,
            "absolute_error_m": absolute_error,
            "threshold_m": float(threshold_m),
            "passed": status == "passed",
            "status": status,
        })
    return rows


def _resolve_spatial_evidence(
    store: Any,
    *,
    requested_id: str,
    enabled: bool,
    cancel_checker=None,
) -> tuple[str, Any, Path | None, str, list[dict[str, Any]]]:
    if not enabled:
        return "", None, None, "", []
    result_id = str(requested_id or "")
    record = None
    manifest_path: Path | None = None
    manifest_sha256 = ""
    try:
        from core.spatial_result_versions import SpatialResultVersionService
        service = SpatialResultVersionService(store)
        result_id = result_id or service.current_result_id()
        if result_id:
            record = service.load_result(result_id)
            manifest_path = store.root / "spatial" / "results" / result_id / "manifest.json"
            if manifest_path.exists():
                manifest_sha256 = _sha256_file(
                    manifest_path, cancel_checker=cancel_checker
                )
    except Exception:
        record = None
        manifest_path = None
        manifest_sha256 = ""
    artifacts: list[dict[str, Any]] = []
    if result_id:
        artifacts.append({
            "role": "spatial_result",
            "result_id": result_id,
            "manifest_path": _safe_rel(manifest_path, store.root) if manifest_path else "",
            "manifest_sha256": manifest_sha256,
            "stale_at_capture": bool(getattr(record, "stale", False)),
        })
    return result_id, record, manifest_path, manifest_sha256, artifacts


def _prepare_package_directories(
    store: Any,
    *,
    package_name: str | None,
    spatial_result_id: str,
) -> dict[str, Path]:
    requested_name = package_name or f"report_{_timestamp()}"
    final_dir = store.root / "reports" / requested_name
    if final_dir.exists() and any(final_dir.iterdir()):
        final_dir = final_dir.parent / f"{final_dir.name}_{uuid.uuid4().hex[:6]}"
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = final_dir.parent / f".{final_dir.name}.{uuid.uuid4().hex}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    paths = {
        "final": final_dir,
        "staging": staging,
        "tables": staging / "tables",
        "json": staging / "json",
        "html": staging / "html",
        "figures": staging / "figures",
        "spatial": staging / "spatial_result",
    }
    for key in ("tables", "json", "html", "figures"):
        paths[key].mkdir(parents=True, exist_ok=True)
    if spatial_result_id:
        source = store.root / "spatial" / "results" / spatial_result_id
        if source.is_dir():
            shutil.copytree(source, paths["spatial"])
    return paths


def _select_report_scope(
    store: Any,
    *,
    selected_line_ids: tuple[str, ...],
    selected_artifact_ids: tuple[str, ...],
) -> tuple[list[Any], list[Any]]:
    all_lines = list(store.list_lines())
    all_line_ids = {line.line_id for line in all_lines}
    unknown_lines = sorted(set(selected_line_ids) - all_line_ids)
    if unknown_lines:
        raise ValueError(f"report profile references unknown lines: {unknown_lines}")
    lines = [
        line for line in all_lines
        if not selected_line_ids or line.line_id in selected_line_ids
    ]
    all_artifacts = list(index_processing_artifacts(store.root))
    all_artifact_ids = {record.artifact_id for record in all_artifacts}
    unknown_artifacts = sorted(set(selected_artifact_ids) - all_artifact_ids)
    if unknown_artifacts:
        raise ValueError(f"report profile references unknown artifacts: {unknown_artifacts}")
    line_scope = {line.line_id for line in lines}
    artifacts = [
        record for record in all_artifacts
        if record.line_id in line_scope
        and (not selected_artifact_ids or record.artifact_id in selected_artifact_ids)
    ]
    return lines, artifacts


def _spatial_row(store: Any, line_id: str, interface_path: Path) -> dict[str, Any]:
    spatial_path = (
        interface_path if interface_path.exists()
        else store.root / "spatial" / f"{line_id}_targets_xy.csv"
    )
    has_xy_count = 0
    empty_xy_count = 0
    if spatial_path.exists():
        try:
            with spatial_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("x") and row.get("y"):
                        has_xy_count += 1
                    else:
                        empty_xy_count += 1
        except Exception:
            has_xy_count = 0
            empty_xy_count = 0
    return {
        "line_id": line_id,
        "spatial_csv_path": _safe_rel(spatial_path, store.root) if spatial_path.exists() else "",
        "row_count": _count_csv_rows(spatial_path),
        "has_xy_count": has_xy_count,
        "empty_xy_count": empty_xy_count,
    }


def _collect_line_evidence(
    store: Any,
    *,
    lines: list[Any],
    include_interpretation: bool,
    include_spatial: bool,
    check_cancel,
    progress_callback=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    quality_rows: list[dict[str, Any]] = []
    interface_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        check_cancel()
        quality_rows.append(_quality_report_to_row(
            line, store.load_quality_report(line.line_id)
        ))
        try:
            interface_summary = store.basal_interface_summary(line.line_id)
        except Exception:
            interface_summary = {"line_id": line.line_id, "status": "not_started"}
        interface_path = store.root / "spatial" / f"{line.line_id}_basal_interface_xy.csv"
        if str(interface_summary.get("status")) != "not_started":
            if not interface_path.exists():
                try:
                    interface_path = store.export_spatial_interface_curve(line.line_id)
                except Exception:
                    pass
            interface_rows.append(_interface_to_row(
                line.line_id, interface_summary, interface_path, store.root
            ))
        target_rows.extend(
            _target_to_row(line.line_id, target)
            for target in store.load_targets(line.line_id)
        )
        spatial_rows.append(_spatial_row(store, line.line_id, interface_path))
        if progress_callback is not None:
            progress_callback(index, max(len(lines), 1), f"汇总测线 {line.line_id}")
    if not include_interpretation:
        interface_rows.clear()
        target_rows.clear()
    if not include_spatial:
        spatial_rows.clear()
    return quality_rows, interface_rows, target_rows, spatial_rows


def generate_project_report_package(
    store: Any,
    *,
    package_name: str | None = None,
    report_profile: dict[str, Any] | None = None,
    cancel_checker=None,
    progress_callback=None,
) -> ReportPackageResult:
    """Generate a formal, auditable engineering delivery package.

    The package contains PDF/HTML/XLSX reports, traceable CSV/JSON tables,
    engineering figures, a file audit, and SHA-256 checksums.  The operation is
    cancellation-aware so the UI can run it through the shared Job Manager.
    """
    from core.job_manager import JobCancelled

    def check_cancel() -> None:
        if cancel_checker is not None and cancel_checker():
            raise JobCancelled("报告生成已取消")

    def report(current: int, total: int, message: str) -> None:
        check_cancel()
        if progress_callback is not None:
            progress_callback(current, total, message)

    generated_at = local_now()
    report_profile = dict(report_profile or {})
    report_id = str(report_profile.get("report_id") or "")
    template_version = str(report_profile.get("template_version") or "field-industry-v2")
    section_flags = {
        key: bool(report_profile.get(key, True))
        for key in (
            "include_project_summary", "include_line_inventory",
            "include_processing_history", "include_bscan_images",
            "include_interpretation", "include_spatial_results",
            "include_borehole_comparison", "include_integrity_manifest",
        )
    }
    # The integrity manifest is a mandatory engineering-delivery control and cannot be disabled.
    section_flags["include_integrity_manifest"] = True
    selected_line_ids = tuple(dict.fromkeys(str(value) for value in report_profile.get("selected_line_ids", ()) if str(value)))
    selected_artifact_ids = tuple(dict.fromkeys(str(value) for value in report_profile.get("selected_artifact_ids", ()) if str(value)))

    # Formal delivery freezes source identities and the current project revision.
    ensure_full_source_hashes(
        store,
        cancel_requested=cancel_checker,
        progress_callback=lambda current, total, message: progress_callback(current, total, message) if progress_callback is not None else None,
    )
    state = load_project_state(store.root)
    software_version = _software_version()
    source_rows_for_snapshot = [record.to_dict() for record in load_source_registry(store)]

    spatial_result_id, spatial_record, spatial_manifest_path, spatial_manifest_sha256, input_artifacts = _resolve_spatial_evidence(
        store,
        requested_id=str(report_profile.get("spatial_result_id") or ""),
        enabled=section_flags["include_spatial_results"],
        cancel_checker=cancel_checker,
    )
    report_snapshot = ReportSnapshot.capture(
        project_id=store.manifest.project_id,
        project_revision=int(state.get("data_revision") or 0),
        software_version=software_version,
        template_version=template_version,
        source_identities=source_rows_for_snapshot,
        input_artifacts=input_artifacts,
    )

    package_paths = _prepare_package_directories(
        store, package_name=package_name, spatial_result_id=spatial_result_id
    )
    final_package_dir = package_paths["final"]
    package_dir = package_paths["staging"]
    tables_dir = package_paths["tables"]
    json_dir = package_paths["json"]
    html_dir = package_paths["html"]
    figures_dir = package_paths["figures"]

    def final_path(path: Path) -> Path:
        return final_package_dir / path.relative_to(package_dir)

    try:
        report(0, 10, "汇总项目、测线与处理记录")
        manifest = store.manifest
        lines, artifacts = _select_report_scope(
            store,
            selected_line_ids=selected_line_ids,
            selected_artifact_ids=selected_artifact_ids,
        )
        line_scope = {line.line_id for line in lines}
        artifact_rows = (
            [_artifact_to_row(record) for record in artifacts]
            if section_flags["include_processing_history"] else []
        )
        line_rows = (
            [_line_to_row(line) for line in lines]
            if section_flags["include_line_inventory"] else []
        )
        sensor_sync_rows = [_sensor_sync_to_row(line, store.root) for line in lines]
        gis_rows = (
            _load_gis_rows(store.root)
            if section_flags["include_spatial_results"] else []
        )
        quality_rows, interface_rows, target_rows, spatial_rows = _collect_line_evidence(
            store,
            lines=lines,
            include_interpretation=section_flags["include_interpretation"],
            include_spatial=section_flags["include_spatial_results"],
            check_cancel=check_cancel,
            progress_callback=progress_callback,
        )

        borehole_threshold_m = float(report_profile.get("borehole_error_threshold_m", 1.0))
        borehole_rows = (
            _borehole_comparison_rows(
                store, line_ids=line_scope, threshold_m=borehole_threshold_m
            )
            if section_flags["include_borehole_comparison"] else []
        )

        project_payload = {
            "project_id": manifest.project_id,
            "project_no": manifest.project_no,
            "name": report_profile.get("title") or manifest.name,
            "location": manifest.location,
            "operator": manifest.operator,
            "compiler": report_profile.get("compiler") or getattr(manifest, "compiler", "") or manifest.operator,
            "reviewer": report_profile.get("reviewer") or getattr(manifest, "reviewer", ""),
            "approver": report_profile.get("approver") or getattr(manifest, "approver", ""),
            "device_model": manifest.device_model,
            "coordinate_system": manifest.coordinate_system,
            "vertical_datum": manifest.vertical_datum,
            "created_at": manifest.created_at,
            "updated_at": manifest.updated_at,
            "project_path": str(store.root),
        }
        metrics = {
            "line_count": len(lines),
            "imported_line_count": sum(1 for line in lines if line.gpr_dataset_path or line.raw_path),
            "qc_passed_count": sum(1 for row in quality_rows if row.get("status") == "通过"),
            "qc_warning_count": sum(1 for row in quality_rows if row.get("status") == "警告"),
            "qc_failed_count": sum(1 for row in quality_rows if row.get("status") == "失败"),
            "processing_artifact_count": len(artifact_rows),
            "interface_line_count": len(interface_rows),
            "confirmed_interface_count": sum(1 for row in interface_rows if row.get("status") == "confirmed"),
            "interface_average_coverage": (sum(float(row.get("coverage_ratio", 0.0)) for row in interface_rows) / len(interface_rows)) if interface_rows else 0.0,
            "target_count": len(interface_rows) if interface_rows else len(target_rows),
            "legacy_point_target_count": len(target_rows),
            "spatial_export_count": sum(1 for row in spatial_rows if row.get("spatial_csv_path")),
            "sensor_sync_line_count": sum(1 for row in sensor_sync_rows if row.get("manifest_path")),
            "gis_layer_count": len(gis_rows),
            "dem_layer_count": sum(1 for row in gis_rows if row.get("is_dem")),
            "borehole_comparison_count": len(borehole_rows),
            "borehole_passed_count": sum(1 for row in borehole_rows if row.get("passed")),
            "borehole_max_absolute_error_m": max(
                (float(row["absolute_error_m"]) for row in borehole_rows if np.isfinite(float(row["absolute_error_m"]))),
                default=float("nan"),
            ),
        }
        summary = {
            "schema": REPORT_PACKAGE_SCHEMA,
            "generated_at": generated_at,
            "report_revision": str(report_profile.get("revision") or "R1"),
            "approval": {
                "compiler": project_payload["compiler"],
                "reviewer": project_payload["reviewer"],
                "approver": project_payload["approver"],
            },
            "project": project_payload,
            "metrics": metrics,
            "files": {},
            "template_version": template_version,
            "report_profile": {
                "sections": section_flags,
                "selected_line_ids": [line.line_id for line in lines],
                "selected_artifact_ids": [record.artifact_id for record in artifacts],
                "note": str(report_profile.get("note") or ""),
            },
            "snapshot": report_snapshot.to_dict(),
            "source_binding": {
                "spatial_result_id": spatial_result_id,
                "spatial_manifest_path": _safe_rel(spatial_manifest_path, store.root) if spatial_manifest_path else "",
                "spatial_manifest_sha256": spatial_manifest_sha256,
                "spatial_stale_at_capture": bool(getattr(spatial_record, "stale", False)),
            },
            "preflight_warnings": list(report_profile.get("preflight_warnings") or []),
            "lifecycle": {"state": "draft", "sealed": False, "approval_status": "工作草稿"},
        }

        line_csv = tables_dir / "line_manifest.csv"
        quality_csv = tables_dir / "quality_summary.csv"
        interfaces_csv = tables_dir / "basal_interface_summary.csv"
        targets_csv = tables_dir / "legacy_targets_summary.csv"
        processing_csv = tables_dir / "processing_artifacts.csv"
        spatial_csv = tables_dir / "spatial_exports.csv"
        sensor_sync_csv = tables_dir / "sensor_sync_summary.csv"
        gis_layers_csv = tables_dir / "gis_layers.csv"
        borehole_csv = tables_dir / "borehole_comparison.csv"
        audit_csv = tables_dir / "file_audit.csv"
        summary_json = json_dir / "project_report_summary.json"
        html_report = html_dir / "project_report.html"
        pdf_report = package_dir / "project_report.pdf"
        xlsx_report = package_dir / "project_report.xlsx"
        checksums_path = package_dir / "checksums.sha256"
        report_manifest = package_dir / "report_manifest.json"

        report(2, 10, "写入标准化 CSV 与 JSON")
        _atomic_write_csv(line_csv, ["line_id", "name", "length_m", "data_quality", "rtk_status", "sensor_sync_status", "processing_status", "target_count", "raw_rows", "raw_size_mb", "data_format", "gpr_dataset_path", "trajectory_path", "processed_result", "params_path", "updated_at"], line_rows)
        _atomic_write_csv(quality_csv, ["line_id", "line_name", "status", "sample_count", "trace_count", "time_window_ns", "length_m", "finite_ratio", "nan_ratio", "trajectory_points", "orientation", "orientation_message", "issue_count"], quality_rows)
        _atomic_write_csv(interfaces_csv, ["line_id", "status", "version", "keypoint_count", "coverage_ratio", "judged_ratio", "clear_ratio", "weak_ratio", "ignore_ratio", "no_interface_ratio", "spatial_curve_path", "spatial_row_count"], interface_rows)
        _atomic_write_csv(targets_csv, ["target_id", "line_id", "distance_m", "depth_m", "x", "y", "type", "status", "confidence", "source_result_id", "source_mode", "source_method_id", "source_manifest_path", "note"], target_rows)
        _atomic_write_csv(processing_csv, ["artifact_id", "line_id", "method_id", "method_name", "role", "status", "input_shape", "output_shape", "shape_changed", "created_at", "data_path", "params_path", "manifest_path", "output_data_sha256", "params_sha256", "manifest_sha256", "save_schema"], artifact_rows)
        _atomic_write_csv(spatial_csv, ["line_id", "spatial_csv_path", "row_count", "has_xy_count", "empty_xy_count"], spatial_rows)
        _atomic_write_csv(sensor_sync_csv, ["line_id", "status", "manifest_path", "trace_metadata_path", "rtk_coverage_ratio", "rtk_fixed_ratio", "rtk_max_residual_s", "imu_coverage_ratio", "altimeter_coverage_ratio", "gap_count", "jump_count", "warning_count"], sensor_sync_rows)
        _atomic_write_csv(gis_layers_csv, ["layer_id", "name", "kind", "role", "crs", "geometry_type", "visible", "opacity", "bounds", "source_path", "is_dem", "imported_at"], gis_rows)
        _atomic_write_csv(borehole_csv, ["borehole_id", "line_id", "trace_index", "borehole_depth_m", "interpreted_sample", "interpreted_depth_m", "error_m", "absolute_error_m", "threshold_m", "passed", "status"], borehole_rows)

        report(3, 10, "生成工程图件")
        figure_paths = []
        if section_flags["include_bscan_images"] or section_flags["include_spatial_results"]:
            figure_paths = _write_report_figures(
                store,
                figures_dir,
                line_ids={line.line_id for line in lines} if section_flags["include_bscan_images"] else set(),
                include_plan_map=section_flags["include_spatial_results"] and not selected_line_ids,
                cancel_checker=cancel_checker,
                progress_callback=lambda current, total, message: progress_callback(current, max(total, 1), message) if progress_callback is not None else None,
            )

        summary["files"] = {
            "line_manifest_csv": _safe_rel(final_path(line_csv), store.root),
            "quality_summary_csv": _safe_rel(final_path(quality_csv), store.root),
            "basal_interface_summary_csv": _safe_rel(final_path(interfaces_csv), store.root),
            "legacy_targets_summary_csv": _safe_rel(final_path(targets_csv), store.root),
            "processing_artifacts_csv": _safe_rel(final_path(processing_csv), store.root),
            "spatial_exports_csv": _safe_rel(final_path(spatial_csv), store.root),
            "sensor_sync_summary_csv": _safe_rel(final_path(sensor_sync_csv), store.root),
            "gis_layers_csv": _safe_rel(final_path(gis_layers_csv), store.root),
            "borehole_comparison_csv": _safe_rel(final_path(borehole_csv), store.root),
            "html_report": _safe_rel(final_path(html_report), store.root),
            "pdf_report": _safe_rel(final_path(pdf_report), store.root),
            "xlsx_report": _safe_rel(final_path(xlsx_report), store.root),
            "checksums": _safe_rel(final_path(checksums_path), store.root),
            "file_audit_csv": _safe_rel(final_path(audit_csv), store.root),
            "figures_dir": _safe_rel(final_path(figures_dir), store.root),
        }
        store.write_json(summary_json, summary)

        report(5, 10, "生成 Excel 工程汇总")
        _write_xlsx_report(
            xlsx_report,
            summary=summary,
            sheets=[
                ("测线清单", line_rows), ("数据质检", quality_rows), ("传感器同步", sensor_sync_rows),
                ("基覆界面", interface_rows), ("处理记录", artifact_rows), ("空间成果", spatial_rows),
                ("GIS图层", gis_rows), ("钻孔对比", borehole_rows), ("历史点目标", target_rows),
            ],
        )

        report(6, 10, "生成 HTML 正式报告")
        _write_html_report(
            html_report, summary=summary, lines=line_rows, quality=quality_rows,
            interfaces=interface_rows, targets=target_rows, artifacts=artifact_rows,
            spatial=spatial_rows, sensor_sync=sensor_sync_rows, gis_layers=gis_rows,
            boreholes=borehole_rows, figure_paths=figure_paths,
        )
        report(7, 10, "生成 PDF 正式报告")
        _write_pdf_report(
            pdf_report, summary=summary, lines=line_rows, quality=quality_rows,
            interfaces=interface_rows, targets=target_rows, artifacts=artifact_rows,
            spatial=spatial_rows, sensor_sync=sensor_sync_rows, gis_layers=gis_rows,
            boreholes=borehole_rows, figure_paths=figure_paths,
        )

        # The manifest is written before hashing so it is itself covered by checksums.
        delivery_dir = store.root / "reports" / "delivery"
        delivery_zip = delivery_dir / f"{final_package_dir.name}_delivery.zip"
        delivery_sha256 = delivery_dir / f"{final_package_dir.name}_delivery.zip.sha256"
        final_seal_path = final_path(package_dir / "report_seal.json")
        existing_files = [p for p in package_dir.rglob("*") if p.is_file()]
        planned_file_count = len(existing_files) + 5  # report_document, manifest, audit, checksums, seal
        result = ReportPackageResult(
            package_dir=_safe_rel(final_package_dir, store.root),
            manifest_path=_safe_rel(final_path(report_manifest), store.root),
            html_path=_safe_rel(final_path(html_report), store.root),
            summary_json_path=_safe_rel(final_path(summary_json), store.root),
            line_manifest_csv_path=_safe_rel(final_path(line_csv), store.root),
            quality_csv_path=_safe_rel(final_path(quality_csv), store.root),
            targets_csv_path=_safe_rel(final_path(targets_csv), store.root),
            processing_csv_path=_safe_rel(final_path(processing_csv), store.root),
            spatial_csv_path=_safe_rel(final_path(spatial_csv), store.root),
            pdf_path=_safe_rel(final_path(pdf_report), store.root),
            interfaces_csv_path=_safe_rel(final_path(interfaces_csv), store.root),
            xlsx_path=_safe_rel(final_path(xlsx_report), store.root),
            sensor_sync_csv_path=_safe_rel(final_path(sensor_sync_csv), store.root),
            gis_layers_csv_path=_safe_rel(final_path(gis_layers_csv), store.root),
            audit_csv_path=_safe_rel(final_path(audit_csv), store.root),
            checksums_path=_safe_rel(final_path(checksums_path), store.root),
            figures_dir=_safe_rel(final_path(figures_dir), store.root),
            seal_path=_safe_rel(final_seal_path, store.root),
            delivery_zip_path=_safe_rel(delivery_zip, store.root),
            delivery_zip_sha256_path=_safe_rel(delivery_sha256, store.root),
            spatial_result_id=spatial_result_id,
            snapshot_id=report_snapshot.snapshot_id,
            file_count=planned_file_count,
            generated_at=generated_at,
        )
        document_sections: list[ReportSection] = []
        if section_flags["include_project_summary"]:
            document_sections.extend((
                ReportSection("project", "项目概况", project_payload),
                ReportSection("metrics", "成果统计", metrics),
                ReportSection("quality", "数据质检", {"rows": quality_rows}),
                ReportSection("sync", "传感器同步", {"rows": sensor_sync_rows}),
            ))
        if section_flags["include_interpretation"]:
            document_sections.append(ReportSection("interfaces", "基覆界面", {"rows": interface_rows}))
        if section_flags["include_spatial_results"]:
            document_sections.append(ReportSection("gis", "空间成果", {"rows": gis_rows, "spatial_result_id": spatial_result_id}))
        if section_flags["include_borehole_comparison"]:
            document_sections.append(ReportSection("boreholes", "钻孔对比", {"rows": borehole_rows, "threshold_m": borehole_threshold_m}))
        report_document = ReportDocument(
            title=str(project_payload["name"]),
            snapshot=report_snapshot,
            approval=dict(summary["approval"]),
            sections=tuple(document_sections),
        )
        store.write_json(json_dir / "report_document.json", report_document.to_dict())

        approval_status = "已批准" if summary["approval"].get("approver") else ("待批准" if summary["approval"].get("reviewer") else "工作草稿")
        seal_path = package_dir / "report_seal.json"
        summary["lifecycle"] = {
            "state": "sealed",
            "sealed": True,
            "approval_status": approval_status,
            "seal_path": _safe_rel(final_path(seal_path), store.root),
            "snapshot_id": report_snapshot.snapshot_id,
            "project_revision": report_snapshot.project_revision,
            "sealed_at": generated_at,
        }
        manifest_payload = {
            "schema": REPORT_PACKAGE_SCHEMA,
            "generated_at": generated_at,
            "project_id": manifest.project_id,
            "report_id": report_id or final_package_dir.name,
            "report_revision": summary["report_revision"],
            "result": result.to_dict(),
            "summary": summary,
            "seal_path": _safe_rel(final_path(seal_path), store.root),
        }
        store.write_json(summary_json, summary)
        store.write_json(report_manifest, manifest_payload)

        report(8, 10, "生成文件审计与 SHA-256 校验")
        audit_rows = []
        excluded = {audit_csv, checksums_path, seal_path}
        for file in sorted(p for p in package_dir.rglob("*") if p.is_file() and p not in excluded):
            check_cancel()
            audit_rows.append({
                "path": file.relative_to(package_dir).as_posix(),
                "size_bytes": file.stat().st_size,
                "sha256": _sha256_file(file, cancel_checker=cancel_checker),
                "generated_at": generated_at,
            })
        _atomic_write_csv(audit_csv, ["path", "size_bytes", "sha256", "generated_at"], audit_rows)
        _write_checksums(package_dir, checksums_path, cancel_checker=cancel_checker)
        ReportSealer().seal(package_dir, snapshot=report_snapshot, approval=dict(summary["approval"]))

        report(9, 10, "封装正式交付 ZIP")
        delivery_dir.mkdir(parents=True, exist_ok=True)
        temp_zip = delivery_dir / f".{delivery_zip.name}.{uuid.uuid4().hex}.tmp"
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(p for p in package_dir.rglob("*") if p.is_file()):
                check_cancel()
                arcname = f"{final_package_dir.name}/{file.relative_to(package_dir).as_posix()}"
                archive.write(file, arcname)
        zip_hash = _sha256_file(temp_zip, cancel_checker=cancel_checker)

        # Commit package and delivery bundle atomically only after every report file,
        # checksum and seal has been completed successfully.
        if final_package_dir.exists():
            raise FileExistsError(final_package_dir)
        package_dir.replace(final_package_dir)
        temp_zip.replace(delivery_zip)
        atomic_write_text(delivery_sha256, f"{zip_hash}  {delivery_zip.name}\n")

        latest_manifest = store.root / "reports" / "latest_report_manifest.json"
        store.write_json(latest_manifest, manifest_payload)
        manifest.reports = {
            "status": "已生成",
            "schema": REPORT_PACKAGE_SCHEMA,
            "generated_at": generated_at,
            "report_id": report_id or final_package_dir.name,
            "snapshot_id": report_snapshot.snapshot_id,
            "project_revision": report_snapshot.project_revision,
            "spatial_result_id": spatial_result_id,
            "approval_status": approval_status,
            "latest_package_dir": result.package_dir,
            "latest_manifest_path": result.manifest_path,
            "latest_html_path": result.html_path,
            "latest_pdf_path": result.pdf_path,
            "latest_xlsx_path": result.xlsx_path,
            "delivery_zip_path": result.delivery_zip_path,
            "delivery_zip_sha256_path": result.delivery_zip_sha256_path,
            "checksums_path": result.checksums_path,
            "seal_path": result.seal_path,
            "file_count": result.file_count,
        }
        store.save_manifest()
        store.storage.catalog.register_export({
            "export_id": report_id or final_package_dir.name,
            "export_kind": "engineering_report",
            "path": result.package_dir,
            "status": "sealed",
            "sha256": zip_hash,
            "created_at": generated_at,
            "metadata": {
                "manifest_path": result.manifest_path,
                "pdf_path": result.pdf_path,
                "html_path": result.html_path,
                "delivery_zip_path": result.delivery_zip_path,
                "file_count": result.file_count,
                "report_profile": summary["report_profile"],
            },
        })
        store.append_log(f"生成正式工程成果报告包：{result.package_dir}")
        report(10, 10, "正式工程报告包生成完成")
        return result
    except Exception:
        if package_dir.exists():
            shutil.rmtree(package_dir, ignore_errors=True)
        delivery_root = store.root / "reports" / "delivery"
        if delivery_root.exists():
            for candidate in delivery_root.glob(f".{final_package_dir.name}_delivery.zip.*.tmp"):
                try:
                    candidate.unlink()
                except OSError:
                    pass
        raise

__all__ = ["REPORT_PACKAGE_SCHEMA", "ReportPackageResult", "generate_project_report_package"]
