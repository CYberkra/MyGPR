#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve target-interpretation source bindings into displayable B-scan views.

The target page must render the same data artifact that new targets are bound to.
This module keeps that source-resolution logic outside the Qt page so spatial
results and reports can reuse it later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.gpr_data_model import GPRDataSet
from core.target_source_binding import TargetSourceBinding


@dataclass(frozen=True)
class TargetSourceDataView:
    """A display-ready B-scan view for target interpretation."""

    source: TargetSourceBinding
    dataset: GPRDataSet
    matrix: np.ndarray
    distance_axis_m: np.ndarray
    depth_axis_m: np.ndarray
    vertical_axis: np.ndarray
    vertical_axis_label: str
    source_note: str = ""

    @property
    def uses_depth_axis(self) -> bool:
        return "深度" in self.vertical_axis_label

    @property
    def shape_text(self) -> str:
        rows, cols = self.matrix.shape
        return f"{rows}×{cols}"


def _safe_load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _project_path(project_root: str | Path | None, relative_or_absolute: str) -> Path | None:
    if not relative_or_absolute:
        return None
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    if project_root is None:
        return path
    return (Path(project_root) / path).resolve()


def _manifest_for_source(project_root: str | Path | None, source: TargetSourceBinding) -> dict[str, Any]:
    manifest_path = _project_path(project_root, source.manifest_path)
    return _safe_load_json(manifest_path)


def _processed_dataset_from_source(
    project_root: str | Path | None,
    source: TargetSourceBinding,
    raw_dataset: GPRDataSet | None,
) -> GPRDataSet:
    data_path = _project_path(project_root, source.data_path)
    if data_path is None or not data_path.exists():
        raise FileNotFoundError(f"Target source data file is missing: {source.data_path}")
    matrix = np.load(data_path)
    manifest = _manifest_for_source(project_root, source)

    length_m = raw_dataset.length_m if raw_dataset is not None else None
    time_window_ns = raw_dataset.time_window_ns if raw_dataset is not None else 250.0
    dielectric = raw_dataset.dielectric_constant if raw_dataset is not None else 9.0

    # Prefer explicit source metadata where available, but keep the raw line axes
    # as the safest fallback because processed .npy artifacts store only matrix
    # values in the current project contract.
    input_dataset = manifest.get("input_dataset") if isinstance(manifest.get("input_dataset"), dict) else {}
    if length_m is None and input_dataset:
        try:
            length_m = float(input_dataset.get("length_m"))
        except Exception:
            length_m = None
    try:
        time_window_ns = float(input_dataset.get("time_window_ns", time_window_ns))
    except Exception:
        pass
    try:
        dielectric = float(input_dataset.get("dielectric_constant", dielectric))
    except Exception:
        pass

    return GPRDataSet.from_matrix(
        source.line_id,
        np.asarray(matrix, dtype=np.float32),
        length_m=length_m,
        time_window_ns=time_window_ns,
        dielectric_constant=dielectric,
        source_path=str(data_path),
        format_name=f"target-source:{source.source_mode}:{source.method_id or 'artifact'}",
        metadata={
            "target_source": source.to_dict(),
            "processing_manifest": manifest,
        },
    )


def resolve_target_source_view(
    *,
    project_root: str | Path | None,
    source: TargetSourceBinding,
    raw_dataset: GPRDataSet | None,
    fallback_dataset: GPRDataSet | None = None,
) -> TargetSourceDataView:
    """Resolve a target source to a concrete matrix and vertical-axis contract."""
    if source.source_mode == "raw":
        if raw_dataset is None:
            raise ValueError("Raw target source requested but no raw dataset is available")
        dataset = raw_dataset
        note = "当前显示原始 B-scan 数据。"
    else:
        try:
            dataset = _processed_dataset_from_source(project_root, source, raw_dataset)
            note = f"当前显示已保存来源：{source.label}。"
        except Exception:
            if fallback_dataset is None and raw_dataset is None:
                raise
            dataset = fallback_dataset or raw_dataset  # type: ignore[assignment]
            note = f"来源文件不可读取，已回退到当前可用数据：{source.label}。"

    matrix = np.asarray(dataset.matrix, dtype=np.float32)
    distance_axis = np.asarray(dataset.distance_axis_m, dtype=np.float32)
    depth_axis = np.asarray(dataset.depth_axis_m, dtype=np.float32)
    if source.is_axis_transform:
        vertical_axis = depth_axis
        vertical_label = "深度 (m)"
        note += " 该来源包含显示与对比 / 坐标轴转换。"
    else:
        vertical_axis = np.asarray(dataset.time_axis_ns, dtype=np.float32)
        vertical_label = "时间 (ns)"
    return TargetSourceDataView(
        source=source,
        dataset=dataset,
        matrix=matrix,
        distance_axis_m=distance_axis,
        depth_axis_m=depth_axis,
        vertical_axis=vertical_axis,
        vertical_axis_label=vertical_label,
        source_note=note,
    )


__all__ = ["TargetSourceDataView", "resolve_target_source_view"]
