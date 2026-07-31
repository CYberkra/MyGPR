#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency rules for MyGPR project events.

These rules deliberately stay small and explicit.  They answer: when an input
or derived artifact changes, which modules are stale and which status labels
should be shown to users.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.project_events import ProjectEvent, ProjectEventType


MODULE_LABELS: dict[str, str] = {
    "project": "项目管理",
    "processing": "测线处理",
    "targets": "界面标注",
    "spatial": "空间成果",
    "report": "成果报告",
    "source_files": "源文件",
}


@dataclass(frozen=True)
class DependencyImpact:
    dirty_modules: set[str] = field(default_factory=set)
    clear_modules: set[str] = field(default_factory=set)
    affected_modules: set[str] = field(default_factory=set)
    report_stale: bool = False
    report_reason: str = ""
    spatial_stale: bool = False
    spatial_reason: str = ""


def _line_prefix(event: ProjectEvent) -> str:
    return f"{event.line_id} " if event.line_id else ""


def resolve_event_impact(event: ProjectEvent) -> DependencyImpact:
    """Return dependency impact for a project event."""

    et = event.event_type
    reason = event.reason.strip()
    line = _line_prefix(event)

    if et == ProjectEventType.LINE_SELECTED:
        return DependencyImpact(affected_modules={"project", "processing", "targets", "spatial", "report"})

    if et in {ProjectEventType.PROJECT_OPENED, ProjectEventType.PROJECT_CLOSED, ProjectEventType.PROJECT_DELETED}:
        return DependencyImpact(affected_modules={"project", "processing", "targets", "spatial", "report", "source_files"})

    if et == ProjectEventType.LINE_IMPORTED:
        msg = reason or f"{line}测线数据已导入"
        return DependencyImpact(
            dirty_modules={"processing", "targets", "spatial", "report"},
            affected_modules={"project", "processing", "targets", "spatial", "report", "source_files"},
            spatial_stale=True,
            spatial_reason=msg,
            report_stale=True,
            report_reason=msg,
        )

    if et == ProjectEventType.TRAJECTORY_IMPORTED:
        msg = reason or f"{line}RTK/IMU 轨迹已更新"
        return DependencyImpact(
            dirty_modules={"spatial", "report"},
            affected_modules={"project", "spatial", "report"},
            spatial_stale=True,
            spatial_reason=msg,
            report_stale=True,
            report_reason=msg,
        )

    if et == ProjectEventType.QC_UPDATED:
        msg = reason or f"{line}质检结果已更新"
        return DependencyImpact(
            dirty_modules={"report"},
            affected_modules={"project", "report"},
            report_stale=True,
            report_reason=msg,
        )

    if et == ProjectEventType.BSCAN_ORIENTATION_FIXED:
        msg = reason or f"{line}B-scan 方向已修正，相关处理/目标需复核"
        return DependencyImpact(
            dirty_modules={"processing", "targets", "spatial", "report"},
            affected_modules={"project", "processing", "targets", "spatial", "report"},
            spatial_stale=True,
            spatial_reason=msg,
            report_stale=True,
            report_reason=msg,
        )

    if et == ProjectEventType.PROCESSING_RESULT_SAVED:
        msg = reason or f"{line}处理结果已保存"
        return DependencyImpact(
            clear_modules={"processing"},
            dirty_modules={"targets", "report"},
            affected_modules={"project", "processing", "targets", "report"},
            report_stale=True,
            report_reason=msg,
        )

    if et in {
        ProjectEventType.TARGETS_CHANGED,
        ProjectEventType.TARGET_SELECTED,
        ProjectEventType.ANNOTATION_SAVED,
        ProjectEventType.ANNOTATION_CONFIRMED,
    }:
        if et == ProjectEventType.TARGET_SELECTED:
            return DependencyImpact(affected_modules={"targets", "spatial"})
        msg = reason or f"{line}界面标注已变化"
        return DependencyImpact(
            clear_modules={"targets"},
            dirty_modules={"spatial", "report"},
            affected_modules={"project", "targets", "spatial", "report"},
            spatial_stale=True,
            spatial_reason=msg,
            report_stale=True,
            report_reason=msg,
        )

    if et == ProjectEventType.SPATIAL_RESULT_SELECTED:
        return DependencyImpact(affected_modules={"spatial", "report"})

    if et == ProjectEventType.REPORT_SELECTED:
        return DependencyImpact(affected_modules={"report"})

    if et == ProjectEventType.SPATIAL_MARKED_STALE:
        msg = reason or "空间成果需要刷新"
        return DependencyImpact(
            dirty_modules={"spatial", "report"},
            affected_modules={"spatial", "report"},
            spatial_stale=True,
            spatial_reason=msg,
            report_stale=True,
            report_reason=msg,
        )

    if et == ProjectEventType.SPATIAL_RESULTS_REFRESHED:
        msg = reason or "空间成果已刷新"
        return DependencyImpact(
            clear_modules={"spatial"},
            dirty_modules={"report"},
            affected_modules={"project", "spatial", "report"},
            report_stale=True,
            report_reason=msg,
        )

    if et == ProjectEventType.SPATIAL_EXPORT_GENERATED:
        msg = reason or "空间坐标成果已导出"
        return DependencyImpact(
            dirty_modules={"report"},
            affected_modules={"spatial", "report"},
            report_stale=True,
            report_reason=msg,
        )

    if et == ProjectEventType.REPORT_GENERATED:
        return DependencyImpact(
            clear_modules={"report"},
            affected_modules={"project", "report"},
        )

    if et == ProjectEventType.REPORT_MARKED_STALE:
        msg = reason or "项目数据已变化"
        return DependencyImpact(
            dirty_modules={"report"},
            affected_modules={"report"},
            report_stale=True,
            report_reason=msg,
        )

    if et in {ProjectEventType.LINE_SOURCE_RELINKED, ProjectEventType.LINE_SOURCE_STATUS_CHECKED}:
        msg = reason or f"{line}源文件状态已更新"
        return DependencyImpact(
            dirty_modules={"report"},
            affected_modules={"project", "source_files", "report"},
            report_stale=True,
            report_reason=msg,
        )

    if et == ProjectEventType.LINE_DELETED:
        msg = reason or f"{line}测线已删除"
        return DependencyImpact(
            dirty_modules={"spatial", "report"},
            affected_modules={"project", "processing", "targets", "spatial", "report", "source_files"},
            spatial_stale=True,
            spatial_reason=msg,
            report_stale=True,
            report_reason=msg,
        )

    return DependencyImpact(affected_modules=set(event.affected_modules or []))


__all__ = ["DependencyImpact", "MODULE_LABELS", "resolve_event_impact"]
