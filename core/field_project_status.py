#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project status aggregation for the MyGPR field workbench.

The project-management and home pages should show facts derived from the
project directory instead of hard-coded demo metrics.  This module is read-only
and keeps the UI from becoming the place where project statistics are invented.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.field_project_store import FieldProjectStore
from core.field_data_quality import DataQualityStatus
from core.processing_artifact_index import index_processing_artifacts
from core.project_state_tracker import load_project_state
from core.report_package_versions import ReportPackageVersionService
from core.schema_registry import SchemaError


@dataclass(frozen=True)
class ProjectStatusSnapshot:
    project_name: str = "未打开项目"
    project_no: str = "--"
    created_at: str = "--"
    updated_at: str = "--"
    location: str = "--"
    operator: str = "--"
    device_model: str = "--"
    coordinate_system: str = "--"
    vertical_datum: str = "--"
    project_path: str = "--"
    status: str = "未打开"
    line_count: int = 0
    imported_line_count: int = 0
    processed_line_count: int = 0
    # Compatibility names retained for legacy widgets; in the formal v0.9.26
    # workflow these values describe basal-interface annotations, not point targets.
    target_count: int = 0
    confirmed_target_count: int = 0
    pending_target_count: int = 0
    spatial_point_count: int = 0
    interface_line_count: int = 0
    confirmed_interface_count: int = 0
    draft_interface_count: int = 0
    interface_average_coverage: float = 0.0
    interface_spatial_row_count: int = 0
    legacy_point_target_count: int = 0
    trajectory_file_count: int = 0
    report_file_count: int = 0
    report_version_count: int = 0
    current_report_id: str = ""
    current_report_snapshot_id: str = ""
    current_report_spatial_result_id: str = ""
    report_status: str = "未生成"
    raw_size_mb: float = 0.0
    qc_passed_count: int = 0
    qc_warning_count: int = 0
    qc_failed_count: int = 0
    storage_usage_mb: float = 0.0
    latest_update: str = "--"
    attention_items: list[tuple[str, str, str, str]] = field(default_factory=list)
    task_rows: list[tuple[str, str, str, str, str, str, str]] = field(default_factory=list)
    activity_rows: list[tuple[str, str, str, str]] = field(default_factory=list)
    delivery_rows: list[tuple[str, str, str, str, str, str]] = field(default_factory=list)
    dirty_modules: dict[str, bool] = field(default_factory=dict)
    stale_reasons: dict[str, list[str]] = field(default_factory=dict)
    data_revision: int = 0

    @property
    def processed_percent(self) -> float:
        if self.line_count <= 0:
            return 0.0
        return self.processed_line_count / self.line_count * 100.0

    @property
    def imported_percent(self) -> float:
        if self.line_count <= 0:
            return 0.0
        return self.imported_line_count / self.line_count * 100.0

    @property
    def data_health_label(self) -> str:
        if self.line_count == 0:
            return "待导入"
        if self.imported_line_count < self.line_count:
            return "需补充"
        return "正常"


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    except (OSError, UnicodeError, csv.Error):
        return 0


def _safe_size_mb(root: Path, rel: str) -> float:
    if not rel:
        return 0.0
    path = root / rel
    try:
        return path.stat().st_size / (1024 * 1024) if path.exists() and path.is_file() else 0.0
    except OSError:
        return 0.0


def _file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def _read_log_rows(root: Path, limit: int = 5) -> list[tuple[str, str, str, str]]:
    log_path = root / "logs" / "field_workbench.log"
    if not log_path.exists():
        return []
    try:
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    except (OSError, UnicodeError):
        return []
    rows: list[tuple[str, str, str, str]] = []
    for raw in reversed(lines[-limit:]):
        ts = "--"
        text = raw
        if raw.startswith("[") and "]" in raw:
            ts, text = raw[1:].split("]", 1)
            ts = ts.strip()[-8:] if ts.strip() else "--"
            text = text.strip()
        icon = "✓" if any(k in text for k in ("保存", "创建", "打开", "导入")) else "ℹ"
        rows.append((icon, text[:28] or "项目活动", text, ts))
    return rows


def _delivery_rows(root: Path) -> list[tuple[str, str, str, str, str, str]]:
    reports = root / "reports"
    if not reports.exists():
        return []
    rows: list[tuple[str, str, str, str, str, str]] = []
    for path in sorted((p for p in reports.rglob("*") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)[:8]:
        suffix = path.suffix.upper().lstrip(".") or "文件"
        size_kb = path.stat().st_size / 1024
        size = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"
        updated = path.stat().st_mtime
        try:
            import datetime as _dt
            updated_text = _dt.datetime.fromtimestamp(updated).strftime("%H:%M:%S")
        except (OSError, OverflowError, ValueError):
            updated_text = "--"
        rows.append((path.name, suffix, size, updated_text, "已生成", "↗"))
    return rows


@dataclass(frozen=True)
class _QualityMetrics:
    reports: list[Any]
    passed: int
    warning: int
    failed: int


@dataclass(frozen=True)
class _InterfaceMetrics:
    line_count: int
    confirmed: int
    pending: int
    target_count: int
    average_coverage: float
    spatial_rows: int
    spatial_points: int
    legacy_point_count: int


@dataclass(frozen=True)
class _ReportMetrics:
    file_count: int
    records: list[Any]
    current: Any | None
    status: str


def _load_project_state_snapshot(root: Path) -> tuple[dict[str, Any], dict[str, bool], dict[str, list[str]]]:
    try:
        project_state = load_project_state(root)
    except (OSError, SchemaError, TypeError, ValueError):
        project_state = {}
    dirty_modules = dict(project_state.get("dirty") or {})
    stale_reasons = {str(key): list(values or []) for key, values in dict(project_state.get("stale_reasons") or {}).items()}
    return project_state, dirty_modules, stale_reasons


def _collect_quality_metrics(store: FieldProjectStore, lines: list[Any]) -> _QualityMetrics:
    reports: list[Any] = []
    for line in lines:
        try:
            report = store.load_quality_report(line.line_id)
        except Exception:
            report = None
        if report is not None:
            reports.append(report)
    return _QualityMetrics(
        reports=reports,
        passed=sum(1 for report in reports if report.status == DataQualityStatus.PASSED),
        warning=sum(1 for report in reports if report.status == DataQualityStatus.WARNING),
        failed=sum(1 for report in reports if report.status == DataQualityStatus.FAILED),
    )


def _legacy_target_counts(store: FieldProjectStore, line_id: str) -> tuple[int, int, int]:
    try:
        targets = store.load_targets(line_id)
    except (OSError, ValueError, TypeError, KeyError):
        return 0, 0, 0
    confirmed = 0
    pending = 0
    for item in targets:
        status = str(item.get("status", ""))
        if "确认" in status or "已" in status:
            confirmed += 1
        elif "复核" in status or "待" in status:
            pending += 1
    return len(targets), confirmed, pending


def _collect_interface_metrics(store: FieldProjectStore, lines: list[Any], root: Path) -> _InterfaceMetrics:
    interface_line_count = confirmed = pending = interface_spatial_rows = 0
    legacy_point_count = legacy_confirmed = legacy_pending = 0
    coverage_values: list[float] = []
    for line in lines:
        try:
            summary = store.basal_interface_summary(line.line_id)
        except Exception:
            summary = {"status": "not_started", "coverage_ratio": 0.0, "keypoint_count": 0}
        status = str(summary.get("status"))
        if status != "not_started":
            interface_line_count += 1
            coverage_values.append(float(summary.get("coverage_ratio") or 0.0))
            if status == "confirmed":
                confirmed += 1
            else:
                pending += 1
        interface_spatial_rows += _count_csv_rows(root / "spatial" / f"{line.line_id}_basal_interface_xy.csv")
        count, legacy_ok, legacy_wait = _legacy_target_counts(store, line.line_id)
        legacy_point_count += count
        legacy_confirmed += legacy_ok
        legacy_pending += legacy_wait

    if interface_line_count:
        target_count = interface_line_count
    else:
        target_count = legacy_point_count
        confirmed = legacy_confirmed
        pending = legacy_pending
    spatial_points = interface_spatial_rows or sum(
        _count_csv_rows(root / "spatial" / f"{line.line_id}_targets_xy.csv") for line in lines
    )
    average_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
    return _InterfaceMetrics(
        line_count=interface_line_count,
        confirmed=confirmed,
        pending=pending,
        target_count=target_count,
        average_coverage=average_coverage,
        spatial_rows=interface_spatial_rows,
        spatial_points=spatial_points,
        legacy_point_count=legacy_point_count,
    )


def _collect_report_metrics(
    store: FieldProjectStore,
    report_payload: dict[str, Any],
    report_count: int,
    dirty_modules: dict[str, bool],
) -> _ReportMetrics:
    records: list[Any] = []
    current = None
    try:
        service = ReportPackageVersionService(store)
        records = service.list_reports()
        current_id = service.current_report_id()
        current = service.load_report(current_id) if current_id else (records[0] if records else None)
    except Exception:
        current = None
    status = str(report_payload.get("status") or ("已生成" if report_count else "未生成"))
    if (current is not None and current.stale) or (bool(dirty_modules.get("report")) and report_count > 0):
        status = "需重新生成"
    return _ReportMetrics(file_count=report_count, records=records, current=current, status=status)


def _build_attention_items(
    *,
    line_count: int,
    imported_count: int,
    quality: _QualityMetrics,
    trajectory_count: int,
    interface: _InterfaceMetrics,
    dirty_modules: dict[str, bool],
    stale_reasons: dict[str, list[str]],
    report_status: str,
) -> list[tuple[str, str, str, str]]:
    attention: list[tuple[str, str, str, str]] = []
    if not line_count:
        attention.append(("ℹ", "暂无测线", "请通过项目管理页导入测线数据。", "0 条"))
    missing_raw = line_count - imported_count
    if missing_raw > 0:
        attention.append(("⚠", "测线数据未完整导入", f"还有 {missing_raw} 条测线缺少 GPR 矩阵或原始文件。", f"{missing_raw} 条"))
    unchecked = imported_count - len(quality.reports)
    if unchecked > 0:
        attention.append(("⚠", "导入数据尚未质检", f"还有 {unchecked} 条已导入测线未生成数据质检报告。", f"{unchecked} 条"))
    if quality.failed > 0:
        attention.append(("⚠", "数据质检失败", f"有 {quality.failed} 条测线存在阻断性数据质量问题。", f"{quality.failed} 条"))
    elif quality.warning > 0:
        attention.append(("⚠", "数据质检警告", f"有 {quality.warning} 条测线存在方向、轨迹或矩阵风险。", f"{quality.warning} 条"))
    missing_traj = line_count - trajectory_count
    if line_count and missing_traj > 0:
        attention.append(("⚠", "辅助定位文件不完整", f"还有 {missing_traj} 条测线未附加 RTK/IMU 轨迹。", f"{missing_traj} 条"))
    if interface.pending > 0:
        attention.append(("ℹ", "待复核界面标注", f"共有 {interface.pending} 条测线的基覆界面曲线需要人工复核或确认。", f"{interface.pending} 条"))
    not_annotated = max(imported_count - interface.line_count, 0)
    if imported_count and not_annotated > 0:
        attention.append(("◷", "基覆界面尚未标注", f"还有 {not_annotated} 条已导入测线没有界面曲线或无界面判定。", f"{not_annotated} 条"))
    if bool(dirty_modules.get("spatial")):
        reason = "；".join((stale_reasons.get("spatial") or ["界面标注或轨迹数据已变化"])[-2:])
        attention.append(("◷", "空间成果需刷新", reason, "需刷新"))
    if bool(dirty_modules.get("report")):
        reason = "；".join((stale_reasons.get("report") or ["项目数据已变化"])[-2:])
        attention.append(("◷", "成果报告需重新生成", reason, "需生成"))
    if report_status != "已生成" and not bool(dirty_modules.get("report")):
        attention.append(("◷", "成果报告未生成", "当前项目还没有正式报告文件。", report_status))
    return attention or [("✓", "项目数据检查", "当前项目主要数据项完整。", "正常")]


def _build_task_rows(
    *,
    manifest: Any,
    line_count: int,
    imported_count: int,
    processed_count: int,
    trajectory_count: int,
    quality: _QualityMetrics,
    interface: _InterfaceMetrics,
    report_count: int,
    report_status: str,
) -> list[tuple[str, str, str, str, str, str, str]]:
    missing_raw = line_count - imported_count
    missing_traj = line_count - trajectory_count
    if imported_count and quality.failed == 0 and quality.warning == 0 and len(quality.reports) == imported_count:
        quality_status = "● 通过"
    elif quality.warning:
        quality_status = "⚠ 警告"
    elif quality.failed:
        quality_status = "⚠ 失败"
    else:
        quality_status = "◷ 待质检"
    if interface.confirmed == imported_count and imported_count:
        interface_status = "● 已完成"
    elif interface.line_count:
        interface_status = "⚠ 待复核"
    else:
        interface_status = "◷ 待标注"
    report_task_status = "● 已生成" if report_status == "已生成" else (
        "◷ 需重新生成" if report_status == "需重新生成" else "◷ 待生成"
    )
    return [
        ("测线数据导入", "数据导入", "● 已完成" if not missing_raw else "⚠ 未完成", f"{imported_count}/{line_count or 0}", manifest.created_at, manifest.updated_at, "查看"),
        ("数据质检", "质量检查", quality_status, f"{len(quality.reports)}/{imported_count or 0}", "--", manifest.updated_at, "查看"),
        ("RTK/IMU 文件附加", "辅助定位", "● 已完成" if line_count and not missing_traj else "⚠ 待补充", f"{trajectory_count}/{line_count or 0}", "--", manifest.updated_at, "查看"),
        ("处理结果保存", "处理任务", "● 已完成" if processed_count else "◷ 待处理", f"{processed_count}/{line_count or 0}", "--", manifest.updated_at, "查看"),
        ("基覆界面标注", "人工标注", interface_status, f"{interface.confirmed}/{imported_count or 0}", "--", manifest.updated_at, "查看"),
        ("成果报告", "报告任务", report_task_status, str(report_count), "--", manifest.updated_at, "查看"),
    ]


def build_project_status_snapshot(store: FieldProjectStore | None) -> ProjectStatusSnapshot:
    """Build a read-only project status snapshot from project files."""
    if store is None:
        return ProjectStatusSnapshot(attention_items=[("ℹ", "未打开项目", "请新建或打开 MyGPR 项目。", "待处理")])

    root = store.root
    manifest = store.manifest
    project_state, dirty_modules, stale_reasons = _load_project_state_snapshot(root)
    lines = store.list_lines()
    artifact_line_ids = {
        artifact.line_id for artifact in index_processing_artifacts(root) if artifact.status not in {"failed", "error"}
    }
    imported_lines = [line for line in lines if bool(line.gpr_dataset_path or line.raw_path)]
    processed_lines = [
        line for line in lines if line.line_id in artifact_line_ids or "完成" in str(line.processing_status)
    ]
    trajectory_lines = [line for line in lines if bool(line.trajectory_path)]
    quality = _collect_quality_metrics(store, lines)
    interface = _collect_interface_metrics(store, lines, root)

    raw_size = sum(float(getattr(line, "raw_size_mb", 0.0) or 0.0) for line in lines)
    if raw_size <= 0:
        raw_size = sum(_safe_size_mb(root, line.gpr_dataset_path) + _safe_size_mb(root, line.raw_path) for line in lines)

    report_count = _file_count(root / "reports")
    report_payload = manifest.reports if isinstance(manifest.reports, dict) else {}
    report = _collect_report_metrics(store, report_payload, report_count, dirty_modules)
    attention = _build_attention_items(
        line_count=len(lines),
        imported_count=len(imported_lines),
        quality=quality,
        trajectory_count=len(trajectory_lines),
        interface=interface,
        dirty_modules=dirty_modules,
        stale_reasons=stale_reasons,
        report_status=report.status,
    )
    tasks = _build_task_rows(
        manifest=manifest,
        line_count=len(lines),
        imported_count=len(imported_lines),
        processed_count=len(processed_lines),
        trajectory_count=len(trajectory_lines),
        quality=quality,
        interface=interface,
        report_count=report.file_count,
        report_status=report.status,
    )
    activity = _read_log_rows(root) or [("ℹ", "项目已加载", f"当前项目：{manifest.name}", str(manifest.updated_at)[-8:])]
    delivery = _delivery_rows(root) or [("暂无正式报告文件", "--", "--", "--", report.status, "--")]
    current_report = report.current
    return ProjectStatusSnapshot(
        project_name=manifest.name,
        project_no=manifest.project_no,
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        location=manifest.location,
        operator=manifest.operator,
        device_model=manifest.device_model,
        coordinate_system=manifest.coordinate_system,
        vertical_datum=manifest.vertical_datum,
        project_path=str(root),
        status=manifest.status or "正常",
        line_count=len(lines),
        imported_line_count=len(imported_lines),
        processed_line_count=len(processed_lines),
        target_count=interface.target_count,
        confirmed_target_count=interface.confirmed,
        pending_target_count=interface.pending,
        spatial_point_count=interface.spatial_points,
        interface_line_count=interface.line_count,
        confirmed_interface_count=interface.confirmed,
        draft_interface_count=interface.pending,
        interface_average_coverage=float(interface.average_coverage),
        interface_spatial_row_count=interface.spatial_rows,
        legacy_point_target_count=interface.legacy_point_count,
        trajectory_file_count=len(trajectory_lines),
        report_file_count=report.file_count,
        report_version_count=len(report.records),
        current_report_id=current_report.report_id if current_report is not None else "",
        current_report_snapshot_id=current_report.snapshot_id if current_report is not None else "",
        current_report_spatial_result_id=current_report.spatial_result_id if current_report is not None else "",
        report_status=report.status,
        raw_size_mb=float(raw_size),
        qc_passed_count=quality.passed,
        qc_warning_count=quality.warning,
        qc_failed_count=quality.failed,
        storage_usage_mb=store.storage_usage_mb(),
        latest_update=manifest.updated_at,
        attention_items=attention[:5],
        task_rows=tasks,
        activity_rows=activity[:5],
        delivery_rows=delivery[:8],
        dirty_modules=dirty_modules,
        stale_reasons=stale_reasons,
        data_revision=int(project_state.get("data_revision") or 0),
    )


__all__ = ["ProjectStatusSnapshot", "build_project_status_snapshot"]
