#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import preflight helpers for user-facing MyGPR project operations.

This module performs lightweight checks before a file is copied into a project.
It is intentionally dialog-independent so both GUI code and tests can use the
same validation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.gpr_data_model import detect_mygpr_airborne_sidecar_csv, load_gpr_dataset
from core.gpr_format_registry import get_format_spec

DIRECT_IMPORT_EXTENSIONS = {".csv", ".txt", ".npy", ".npz", ".h5", ".hdf5"}


@dataclass(frozen=True)
class ImportPreflightResult:
    """Result shown to users before importing one measurement file."""

    path: str
    exists: bool
    is_file: bool
    extension: str
    format_name: str
    support: str
    can_import: bool
    message: str
    suggestions: tuple[str, ...] = ()
    sample_count: int = 0
    trace_count: int = 0
    length_m: float = 0.0
    time_window_ns: float = 0.0
    dielectric_constant: float = 0.0
    data_min: float = 0.0
    data_max: float = 0.0
    source_kind: str = ""
    has_trajectory: bool = False
    column_summary: str = ""

    @property
    def shape_text(self) -> str:
        return f"{self.sample_count} × {self.trace_count}" if self.sample_count and self.trace_count else "--"

    def to_log_lines(self) -> list[str]:
        lines = [
            f"文件：{Path(self.path).name}",
            f"格式：{self.format_name or self.extension or '--'}",
            f"支持状态：{self.support}",
            f"矩阵尺寸：{self.shape_text}",
        ]
        if self.source_kind:
            lines.append(f"数据类型：{self.source_kind}")
        if self.length_m:
            lines.append(f"测线长度：{self.length_m:.2f} m")
        if self.time_window_ns:
            lines.append(f"时间窗：{self.time_window_ns:.2f} ns")
        if self.column_summary:
            lines.append(f"列识别：{self.column_summary}")
        lines.append(f"定位信息：{'已识别' if self.has_trajectory else '未识别'}")
        lines.append(f"结论：{self.message}")
        if self.suggestions:
            lines.append("建议：" + "；".join(self.suggestions))
        return lines


def build_import_preflight(source: str | Path, *, line_id: str = "L01", dielectric_constant: float = 9.0) -> ImportPreflightResult:
    src = Path(source).expanduser().resolve()
    suffix = src.suffix.lower()
    spec = get_format_spec(src)
    display_name = spec.display_name if spec else (suffix or "未知格式")

    if not src.exists():
        return ImportPreflightResult(
            path=str(src),
            exists=False,
            is_file=False,
            extension=suffix,
            format_name=display_name,
            support="missing",
            can_import=False,
            message="数据文件不存在。",
            suggestions=("检查文件路径是否正确",),
        )
    if not src.is_file():
        return ImportPreflightResult(
            path=str(src),
            exists=True,
            is_file=False,
            extension=suffix,
            format_name=display_name,
            support="directory",
            can_import=False,
            message="当前导入入口只支持单个文件，不支持目录。",
            suggestions=("请选择一个 CSV / NPY / NPZ / H5 文件",),
        )

    if suffix in DIRECT_IMPORT_EXTENSIONS:
        try:
            dataset = load_gpr_dataset(src, line_id=line_id, dielectric_constant=dielectric_constant)
            matrix = np.asarray(dataset.matrix, dtype=np.float32)
            is_sidecar = dataset.format_name == "mygpr-airborne-sidecar-csv"
            columns = dataset.metadata.get("columns", []) if isinstance(dataset.metadata, dict) else []
            return ImportPreflightResult(
                path=str(src),
                exists=True,
                is_file=True,
                extension=suffix,
                format_name=dataset.format_name or display_name,
                support="direct",
                can_import=True,
                message="可直接导入，已识别为旧 MyGPR 航空 GPR 主数据 CSV。" if is_sidecar else "可直接导入，已识别为二维 B-scan 矩阵。",
                suggestions=("将自动生成 B-scan 矩阵、轨迹 CSV 和导入 manifest",) if is_sidecar else ("确认测线编号和名称后即可导入",),
                sample_count=int(dataset.sample_count),
                trace_count=int(dataset.trace_count),
                length_m=float(dataset.length_m),
                time_window_ns=float(dataset.time_window_ns),
                dielectric_constant=float(dataset.dielectric_constant),
                data_min=float(np.nanmin(matrix)) if np.isfinite(matrix).any() else 0.0,
                data_max=float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 0.0,
                source_kind="MyGPR 航空 GPR 主数据 CSV" if is_sidecar else "二维矩阵数据",
                has_trajectory=bool(dataset.metadata.get("trajectory_rows")) if isinstance(dataset.metadata, dict) else False,
                column_summary=", ".join(columns) if columns else "",
            )
        except Exception as exc:
            return ImportPreflightResult(
                path=str(src),
                exists=True,
                is_file=True,
                extension=suffix,
                format_name=display_name,
                support="direct-failed",
                can_import=False,
                message=f"文件格式已支持，但内容未通过 B-scan 矩阵校验：{exc}",
                suggestions=("确认文件中包含二维数值矩阵", "必要时先导出为标准 CSV / NPY / NPZ"),
            )

    if spec is not None:
        return ImportPreflightResult(
            path=str(src),
            exists=True,
            is_file=True,
            extension=suffix,
            format_name=spec.display_name,
            support=spec.support,
            can_import=False,
            message=f"已识别 {spec.display_name}，但当前项目向导尚未直接解码该格式。",
            suggestions=(spec.notes or "建议先转换为 CSV / NPY / NPZ / H5 后导入", "后续可作为厂商格式专项适配"),
        )

    return ImportPreflightResult(
        path=str(src),
        exists=True,
        is_file=True,
        extension=suffix,
        format_name=display_name,
        support="unsupported",
        can_import=False,
        message=f"暂不支持该数据格式：{suffix or '无扩展名'}。",
        suggestions=("当前正式导入入口支持 CSV / TXT / NPY / NPZ / H5 / HDF5",),
    )


__all__ = ["DIRECT_IMPORT_EXTENSIONS", "ImportPreflightResult", "build_import_preflight"]
