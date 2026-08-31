#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target annotation source binding helpers for the field workbench.

Targets must remain traceable to the data view used during interpretation.  The
source can be the raw B-scan dataset, the latest saved processing artifact, or a
specific saved processing artifact such as a time-to-depth display/compare
result.  This module keeps that metadata outside UI callbacks so the spatial and
report pages can rely on persisted CSV fields later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

from core.gpr_data_model import GPRDataSet
from core.processing_artifact_index import ProcessingArtifactRecord


@dataclass(frozen=True)
class TargetSourceBinding:
    source_result_id: str
    source_mode: str
    label: str
    line_id: str
    data_path: str = ""
    manifest_path: str = ""
    method_id: str = ""
    method_name: str = ""
    artifact_role: str = "raw_data"
    axis_transform: dict[str, Any] | None = None
    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()

    @property
    def is_axis_transform(self) -> bool:
        return bool(self.axis_transform) or self.artifact_role == "display_compare_transform"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["output_shape"] = list(self.output_shape)
        payload["axis_transform"] = self.axis_transform or {}
        return payload

    def to_target_fields(self) -> dict[str, str]:
        """Return the persisted CSV fields for a target bound to this source."""
        return {
            "source_result_id": self.source_result_id,
            "source_mode": self.source_mode,
            "source_data_path": self.data_path,
            "source_manifest_path": self.manifest_path,
            "source_method_id": self.method_id,
            "source_method_name": self.method_name,
            "source_artifact_role": self.artifact_role,
            "source_axis_transform": json.dumps(self.axis_transform or {}, ensure_ascii=False, sort_keys=True),
            "source_input_shape": json.dumps(list(self.input_shape), ensure_ascii=False),
            "source_output_shape": json.dumps(list(self.output_shape), ensure_ascii=False),
        }


def raw_target_source(dataset: GPRDataSet | None, *, line_id: str) -> TargetSourceBinding:
    shape = tuple(dataset.matrix.shape) if dataset is not None else ()
    return TargetSourceBinding(
        source_result_id=f"{line_id}_raw",
        source_mode="raw",
        label=f"原始数据（{line_id}）",
        line_id=line_id,
        data_path=str(dataset.source_path or "") if dataset is not None else "",
        method_id="raw",
        method_name="原始 B-scan",
        artifact_role="raw_data",
        input_shape=shape,
        output_shape=shape,
    )


def artifact_target_source(record: ProcessingArtifactRecord, *, label_prefix: str = "处理结果") -> TargetSourceBinding:
    prefix = "显示与对比" if record.is_display_compare_transform else label_prefix
    method_name = record.method_name or record.method_id or "处理结果"
    label = f"{prefix}：{method_name}（{record.artifact_id}）"
    mode = "display_compare" if record.is_display_compare_transform else "processed"
    return TargetSourceBinding(
        source_result_id=record.artifact_id,
        source_mode=mode,
        label=label,
        line_id=record.line_id,
        data_path=record.data_path,
        manifest_path=record.manifest_path,
        method_id=record.method_id,
        method_name=method_name,
        artifact_role=record.role,
        axis_transform=record.axis_transform,
        input_shape=record.input_shape,
        output_shape=record.output_shape,
    )


def bind_target_to_source(target: dict[str, Any], source: TargetSourceBinding) -> dict[str, Any]:
    """Return a copy of ``target`` with source-traceability fields attached."""
    bound = dict(target)
    bound.update(source.to_target_fields())
    return bound


def source_label_from_target(target: dict[str, Any]) -> str:
    mode = str(target.get("source_mode") or "")
    method = str(target.get("source_method_name") or "")
    sid = str(target.get("source_result_id") or "")
    role = str(target.get("source_artifact_role") or "")
    if mode == "raw" or sid.endswith("_raw"):
        return "原始数据"
    if role == "display_compare_transform" or mode == "display_compare":
        return f"显示与对比：{method or sid}"
    return f"处理结果：{method or sid}" if (method or sid) else "--"


__all__ = [
    "TargetSourceBinding",
    "raw_target_source",
    "artifact_target_source",
    "bind_target_to_source",
    "source_label_from_target",
]
