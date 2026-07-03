#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data quality checks for imported MyGPR field lines.

This module is deliberately independent from Qt.  It evaluates the normalized
GPR dataset and optional trajectory after import, writes a durable JSON report,
and returns a compact status for project tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from core.gpr_data_model import GPRDataSet
from core.trajectory_model import TrajectoryModel
from core.field_project_models import local_now

DATA_QUALITY_SCHEMA = "mygpr.line_data_quality.v1"


class DataQualityStatus(str, Enum):
    """Overall data quality report status."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class DataOrientation(str, Enum):
    """B-scan matrix orientation classification."""

    INVALID = "invalid"
    TRANSPOSE_RISK = "transpose_risk"
    FEW_TRACES = "few_traces"
    FEW_SAMPLES = "few_samples"
    INVALID_LENGTH = "invalid_length"
    INVALID_TIME_WINDOW = "invalid_time_window"
    SAMPLES_BY_TRACES = "samples_by_traces"


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    code: str
    message: str
    suggestion: str = ""


@dataclass(frozen=True)
class LineDataQualityReport:
    schema: str
    line_id: str
    status: str
    status_label: str
    checked_at: str
    sample_count: int
    trace_count: int
    time_window_ns: float
    length_m: float
    amplitude_min: float
    amplitude_max: float
    amplitude_p995: float
    nan_ratio: float
    finite_ratio: float
    trajectory_points: int
    orientation: str
    orientation_message: str
    suggested_action: str
    issues: list[QualityIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LineDataQualityReport":
        issues = [QualityIssue(**issue) for issue in payload.get("issues", []) if isinstance(issue, dict)]
        data = {k: v for k, v in payload.items() if k != "issues"}
        return cls(issues=issues, **data)


def _safe_float(value: float) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return out


def _status_from_issues(issues: list[QualityIssue]) -> tuple[str, str]:
    severities = {issue.severity for issue in issues}
    if "error" in severities:
        return DataQualityStatus.FAILED, "失败"
    if "warning" in severities:
        return DataQualityStatus.WARNING, "警告"
    return DataQualityStatus.PASSED, "通过"


def _orientation_check(dataset: GPRDataSet) -> tuple[str, str, str, QualityIssue | None]:
    rows = int(dataset.sample_count)
    cols = int(dataset.trace_count)
    length_m = float(dataset.length_m)
    time_window = float(dataset.time_window_ns)
    if rows <= 0 or cols <= 0:
        return DataOrientation.INVALID, "B-scan 矩阵为空。", "重新导入测线数据。", QualityIssue("error", "empty_matrix", "B-scan 矩阵为空。", "重新导入测线数据。")
    # MyGPR field sidecar convention is rows=samples/time/depth, cols=traces/distance.
    # Typical YingShan lines are around 501 x 2000+; matrix CSVs may be smaller, but
    # a matrix with many more rows than columns is a strong transposition risk.
    if rows > max(cols * 2, 128):
        issue = QualityIssue(
            "warning",
            "transpose_risk",
            f"矩阵形态为 {rows}×{cols}，采样点数显著大于道数，存在转置导入风险。",
            "确认原始 CSV 中行/列含义；必要时使用方向修正入口转置后再处理。",
        )
        return DataOrientation.TRANSPOSE_RISK, issue.message, issue.suggestion, issue
    if cols < 8:
        issue = QualityIssue("warning", "few_traces", f"道数较少：{cols}。", "确认该文件是否为完整 B-scan。")
        return DataOrientation.FEW_TRACES, issue.message, issue.suggestion, issue
    if rows < 16:
        issue = QualityIssue("warning", "few_samples", f"采样点较少：{rows}。", "确认采样点数是否被截断。")
        return DataOrientation.FEW_SAMPLES, issue.message, issue.suggestion, issue
    if length_m <= 0:
        issue = QualityIssue("warning", "invalid_length", "测线长度为 0 或无效。", "检查 Trace interval 或 distance_axis。")
        return DataOrientation.INVALID_LENGTH, issue.message, issue.suggestion, issue
    if time_window <= 0:
        issue = QualityIssue("warning", "invalid_time_window", "时间窗为 0 或无效。", "检查 Time windows (ns) 头信息。")
        return DataOrientation.INVALID_TIME_WINDOW, issue.message, issue.suggestion, issue
    return DataOrientation.SAMPLES_BY_TRACES, f"方向正常：samples×traces = {rows}×{cols}。", "无需修正。", None


def evaluate_line_data_quality(dataset: GPRDataSet, trajectory: TrajectoryModel | None = None) -> LineDataQualityReport:
    matrix = np.asarray(dataset.matrix, dtype=np.float32)
    issues: list[QualityIssue] = []
    if matrix.ndim != 2:
        issues.append(QualityIssue("error", "not_2d", f"矩阵维度不是二维：{matrix.shape!r}。", "重新导入标准 B-scan 数据。"))
        finite = np.asarray([], dtype=np.float32)
    else:
        finite_mask = np.isfinite(matrix)
        finite = matrix[finite_mask]
        finite_ratio = float(finite_mask.sum() / matrix.size) if matrix.size else 0.0
        nan_ratio = 1.0 - finite_ratio
        if matrix.size == 0:
            issues.append(QualityIssue("error", "empty_matrix", "B-scan 矩阵为空。", "重新导入测线数据。"))
        elif finite_ratio < 0.995:
            severity = "error" if finite_ratio < 0.95 else "warning"
            issues.append(QualityIssue(severity, "nan_inf", f"矩阵存在 NaN/Inf，占比 {nan_ratio:.2%}。", "检查原始 CSV 是否包含坏值或空值。"))
        if finite.size and float(np.nanmax(finite) - np.nanmin(finite)) <= 1e-9:
            issues.append(QualityIssue("error", "flat_amplitude", "振幅范围接近 0，数据可能为空或导出错误。", "检查 amplitude 列和采集文件。"))
    finite_ratio = float(np.isfinite(matrix).sum() / matrix.size) if matrix.size else 0.0
    nan_ratio = 1.0 - finite_ratio
    if finite.size:
        amp_min = _safe_float(np.nanmin(finite))
        amp_max = _safe_float(np.nanmax(finite))
        amp_p995 = _safe_float(np.nanpercentile(np.abs(finite), 99.5))
    else:
        amp_min = amp_max = amp_p995 = 0.0
    orientation, orientation_message, suggested_action, orientation_issue = _orientation_check(dataset)
    if orientation_issue is not None:
        issues.append(orientation_issue)
    traj_points = len(trajectory.points) if trajectory is not None else 0
    if trajectory is None or traj_points <= 0:
        issues.append(QualityIssue("warning", "missing_trajectory", "未找到轨迹数据。", "导入 RTK/IMU 或使用含经纬度的 MyGPR CSV。"))
    elif dataset.trace_count and abs(traj_points - dataset.trace_count) > max(3, int(dataset.trace_count * 0.02)):
        issues.append(QualityIssue("warning", "trajectory_trace_mismatch", f"轨迹点数 {traj_points} 与道数 {dataset.trace_count} 不一致。", "检查轨迹与雷达道是否一一对应。"))
    status, label = _status_from_issues(issues)
    return LineDataQualityReport(
        schema=DATA_QUALITY_SCHEMA,
        line_id=dataset.line_id,
        status=status,
        status_label=label,
        checked_at=local_now(),
        sample_count=int(dataset.sample_count),
        trace_count=int(dataset.trace_count),
        time_window_ns=float(dataset.time_window_ns),
        length_m=float(dataset.length_m),
        amplitude_min=amp_min,
        amplitude_max=amp_max,
        amplitude_p995=amp_p995,
        nan_ratio=float(nan_ratio),
        finite_ratio=float(finite_ratio),
        trajectory_points=int(traj_points),
        orientation=orientation,
        orientation_message=orientation_message,
        suggested_action=suggested_action,
        issues=issues,
    )


__all__ = ["DATA_QUALITY_SCHEMA", "DataQualityStatus", "DataOrientation", "QualityIssue", "LineDataQualityReport", "evaluate_line_data_quality"]
