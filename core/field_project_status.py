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
    target_count: int = 0
    confirmed_target_count: int = 0
    pending_target_count: int = 0
    spatial_point_count: int = 0
    trajectory_file_count: int = 0
    report_file_count: int = 0
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
    except Exception:
        return 0


def _safe_size_mb(root: Path, rel: str) -> float:
    if not rel:
        return 0.0
    path = root / rel
    try:
        return path.stat().st_size / (1024 * 1024) if path.exists() and path.is_file() else 0.0
    except Exception:
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
    except Exception:
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
        except Exception:
            updated_text = "--"
        rows.append((path.name, suffix, size, updated_text, "已生成", "↗"))
    return rows


def build_project_status_snapshot(store: FieldProjectStore | None) -> ProjectStatusSnapshot:
    """Build a read-only project status snapshot from project files."""
    if store is None:
        return ProjectStatusSnapshot(attention_items=[("ℹ", "未打开项目", "请新建或打开 MyGPR 项目。", "待处理")])

    root = store.root
    manifest = store.manifest
    try:
        project_state = load_project_state(root)
    except Exception:
        project_state = {}
    dirty_modules = dict(project_state.get("dirty") or {})
    stale_reasons = {str(k): list(v or []) for k, v in dict(project_state.get("stale_reasons") or {}).items()}
    lines = store.list_lines()
    artifacts = index_processing_artifacts(root)
    artifact_line_ids = {a.line_id for a in artifacts if a.status not in {"failed", "error"}}

    imported_lines = [line for line in lines if bool(line.gpr_dataset_path or line.raw_path)]
    processed_lines = [line for line in lines if line.line_id in artifact_line_ids or "完成" in str(line.processing_status)]
    trajectory_lines = [line for line in lines if bool(line.trajectory_path)]
    quality_reports = []
    for line in lines:
        try:
            report = store.load_quality_report(line.line_id)
        except Exception:
            report = None
        if report is not None:
            quality_reports.append(report)
    qc_passed = sum(1 for report in quality_reports if report.status == DataQualityStatus.PASSED)
    qc_warning = sum(1 for report in quality_reports if report.status == DataQualityStatus.WARNING)
    qc_failed = sum(1 for report in quality_reports if report.status == DataQualityStatus.FAILED)

    target_count = 0
    confirmed = 0
    pending = 0
    for line in lines:
        targets = store.load_targets(line.line_id)
        if not targets and line.target_count:
            target_count += int(line.target_count)
            continue
        target_count += len(targets)
        for item in targets:
            status = str(item.get("status", ""))
            if "确认" in status or "已" in status:
                confirmed += 1
            elif "复核" in status or "待" in status:
                pending += 1

    spatial_points = sum(_count_csv_rows(root / "spatial" / f"{line.line_id}_targets_xy.csv") for line in lines)
    raw_size = sum(float(getattr(line, "raw_size_mb", 0.0) or 0.0) for line in lines)
    if raw_size <= 0:
        raw_size = sum(_safe_size_mb(root, line.gpr_dataset_path) + _safe_size_mb(root, line.raw_path) for line in lines)

    report_count = _file_count(root / "reports")
    report_payload = manifest.reports if isinstance(manifest.reports, dict) else {}
    report_status = str(report_payload.get("status") or ("已生成" if report_count else "未生成"))
    if bool(dirty_modules.get("report")) and report_count > 0:
        report_status = "需重新生成"

    attention: list[tuple[str, str, str, str]] = []
    if not lines:
        attention.append(("ℹ", "暂无测线", "请通过项目管理页导入测线数据。", "0 条"))
    missing_raw = len(lines) - len(imported_lines)
    if missing_raw > 0:
        attention.append(("⚠", "测线数据未完整导入", f"还有 {missing_raw} 条测线缺少 GPR 矩阵或原始文件。", f"{missing_raw} 条"))
    unchecked = len(imported_lines) - len(quality_reports)
    if unchecked > 0:
        attention.append(("⚠", "导入数据尚未质检", f"还有 {unchecked} 条已导入测线未生成数据质检报告。", f"{unchecked} 条"))
    if qc_failed > 0:
        attention.append(("⚠", "数据质检失败", f"有 {qc_failed} 条测线存在阻断性数据质量问题。", f"{qc_failed} 条"))
    elif qc_warning > 0:
        attention.append(("⚠", "数据质检警告", f"有 {qc_warning} 条测线存在方向、轨迹或矩阵风险。", f"{qc_warning} 条"))
    missing_traj = len(lines) - len(trajectory_lines)
    if lines and missing_traj > 0:
        attention.append(("⚠", "辅助定位文件不完整", f"还有 {missing_traj} 条测线未附加 RTK/IMU 轨迹。", f"{missing_traj} 条"))
    if pending > 0:
        attention.append(("ℹ", "待复核目标", f"共有 {pending} 个目标需要人工复核或确认。", f"{pending} 个"))
    if bool(dirty_modules.get("spatial")):
        reason = "；".join((stale_reasons.get("spatial") or ["目标或轨迹数据已变化"])[-2:])
        attention.append(("◷", "空间成果需刷新", reason, "需刷新"))
    if bool(dirty_modules.get("report")):
        reason = "；".join((stale_reasons.get("report") or ["项目数据已变化"])[-2:])
        attention.append(("◷", "成果报告需重新生成", reason, "需生成"))
    if report_status != "已生成" and not bool(dirty_modules.get("report")):
        attention.append(("◷", "成果报告未生成", "当前项目还没有正式报告文件。", report_status))
    if not attention:
        attention.append(("✓", "项目数据检查", "当前项目主要数据项完整。", "正常"))

    tasks: list[tuple[str, str, str, str, str, str, str]] = [
        ("测线数据导入", "数据导入", "● 已完成" if not missing_raw else "⚠ 未完成", f"{len(imported_lines)}/{len(lines) or 0}", manifest.created_at, manifest.updated_at, "查看"),
        ("数据质检", "质量检查", "● 通过" if imported_lines and qc_failed == 0 and qc_warning == 0 and len(quality_reports) == len(imported_lines) else ("⚠ 警告" if qc_warning else ("⚠ 失败" if qc_failed else "◷ 待质检")), f"{len(quality_reports)}/{len(imported_lines) or 0}", "--", manifest.updated_at, "查看"),
        ("RTK/IMU 文件附加", "辅助定位", "● 已完成" if lines and not missing_traj else "⚠ 待补充", f"{len(trajectory_lines)}/{len(lines) or 0}", "--", manifest.updated_at, "查看"),
        ("处理结果保存", "处理任务", "● 已完成" if processed_lines else "◷ 待处理", f"{len(processed_lines)}/{len(lines) or 0}", "--", manifest.updated_at, "查看"),
        ("目标标注", "标注任务", "● 已完成" if target_count else "◷ 待标注", str(target_count), "--", manifest.updated_at, "查看"),
        ("成果报告", "报告任务", "● 已生成" if report_status == "已生成" else ("◷ 需重新生成" if report_status == "需重新生成" else "◷ 待生成"), str(report_count), "--", manifest.updated_at, "查看"),
    ]

    activity = _read_log_rows(root)
    if not activity:
        activity = [("ℹ", "项目已加载", f"当前项目：{manifest.name}", str(manifest.updated_at)[-8:])]

    delivery = _delivery_rows(root)
    if not delivery:
        delivery = [("暂无正式报告文件", "--", "--", "--", report_status, "--")]

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
        target_count=target_count,
        confirmed_target_count=confirmed,
        pending_target_count=pending,
        spatial_point_count=spatial_points,
        trajectory_file_count=len(trajectory_lines),
        report_file_count=report_count,
        report_status=report_status,
        raw_size_mb=float(raw_size),
        qc_passed_count=qc_passed,
        qc_warning_count=qc_warning,
        qc_failed_count=qc_failed,
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
