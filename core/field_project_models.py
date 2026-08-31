#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared data models and atomic file helpers for the field project stores."""

from __future__ import annotations

import csv
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.storage_primitives import atomic_write_json as _durable_write_json
from core.storage_primitives import atomic_write_text as _durable_write_text
from core.storage_primitives import utc_now
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass

FIELD_PROJECT_SCHEMA = "mygpr.field_project.v3"
TARGET_FIELDS = [
    "target_id",
    "line_id",
    "distance_m",
    "depth_m",
    "x",
    "y",
    "type",
    "confidence",
    "status",
    "note",
    "created_at",
    "updated_at",
    "source_result_id",
    "source_mode",
    "source_data_path",
    "source_manifest_path",
    "source_method_id",
    "source_method_name",
    "source_artifact_role",
    "source_axis_transform",
    "source_input_shape",
    "source_output_shape",
]

LINE_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*$"


def validate_line_id(line_id: str) -> str:
    """Return a safe project line id or raise ``ValueError``.

    Line ids are used to create paths below ``raw/<line_id>/``.  Restrict them
    to a conservative ASCII identifier form so user input cannot create nested
    paths, Windows device names, or traversal sequences.
    """
    import re

    value = str(line_id or "").strip()
    if not value:
        raise ValueError("测线编号不能为空。")
    if not re.fullmatch(LINE_ID_PATTERN, value):
        raise ValueError(f"非法测线编号：{line_id!r}；仅允许字母、数字和下划线，且必须以字母开头。")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if value.upper() in reserved:
        raise ValueError(f"非法测线编号：{line_id!r} 是 Windows 保留设备名。")
    return value


def local_now() -> str:
    """Return the canonical UTC timestamp used in persisted project documents."""
    return utc_now()


class LineStatus(StrEnum):
    PENDING = "--"
    DRAFT = "草稿"
    IMPORTED = "已导入"
    PROCESSED = "已处理"
    COMPLETED = "已完成"


def atomic_write_text(path: Path, text: str) -> None:
    _durable_write_text(path, text)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _durable_write_json(path, payload)


def count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            return max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    except OSError:
        return 0


@dataclass
class FieldLineRecord:
    line_id: str
    name: str
    length_m: float = 0.0
    data_quality: str | LineStatus = LineStatus.PENDING
    rtk_status: str = "未定位"
    processing_status: str = "未处理"
    updated_at: str = ""
    raw_path: str = ""
    raw_rows: int = 0
    trace_count: int = 0
    raw_size_mb: float = 0.0
    target_count: int = 0
    processed_result: str = ""
    params_path: str = ""
    gpr_dataset_path: str = ""
    trajectory_path: str = ""
    sensor_sync_manifest_path: str = ""
    trace_metadata_path: str = ""
    sensor_sync_status: str = "未同步"
    data_format: str = ""
    interface_status: str = "未开始"
    interface_coverage: float = 0.0
    interface_keypoint_count: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FieldLineRecord":
        quality_raw = payload.get("data_quality", LineStatus.PENDING)
        quality: str | LineStatus
        if isinstance(quality_raw, LineStatus):
            quality = quality_raw
        else:
            quality = str(quality_raw or LineStatus.PENDING.value)
        safe_line_id = validate_line_id(str(payload.get("line_id") or ""))
        return cls(
            line_id=safe_line_id,
            name=str(payload.get("name") or payload.get("line_id") or ""),
            length_m=float(payload.get("length_m") or 0.0),
            data_quality=quality,
            rtk_status=str(payload.get("rtk_status") or "未定位"),
            processing_status=str(payload.get("processing_status") or "未处理"),
            updated_at=str(payload.get("updated_at") or ""),
            raw_path=str(payload.get("raw_path") or ""),
            raw_rows=int(payload.get("raw_rows") or 0),
            trace_count=int(payload.get("trace_count") or 0),
            raw_size_mb=float(payload.get("raw_size_mb") or 0.0),
            target_count=int(payload.get("target_count") or 0),
            processed_result=str(payload.get("processed_result") or ""),
            params_path=str(payload.get("params_path") or ""),
            gpr_dataset_path=str(payload.get("gpr_dataset_path") or ""),
            trajectory_path=str(payload.get("trajectory_path") or ""),
            sensor_sync_manifest_path=str(payload.get("sensor_sync_manifest_path") or ""),
            trace_metadata_path=str(payload.get("trace_metadata_path") or ""),
            sensor_sync_status=str(payload.get("sensor_sync_status") or "未同步"),
            data_format=str(payload.get("data_format") or ""),
            interface_status=str(payload.get("interface_status") or "未开始"),
            interface_coverage=float(payload.get("interface_coverage") or 0.0),
            interface_keypoint_count=int(payload.get("interface_keypoint_count") or 0),
        )

    def to_ui_dict(self) -> dict[str, Any]:
        return {
            "id": self.line_id,
            "name": self.name,
            "length": float(self.length_m),
            "quality": self.data_quality.value if hasattr(self.data_quality, "value") else self.data_quality,
            "rtk": f"● {self.rtk_status}",
            "status": f"● {self.processing_status}",
            "updated": self.updated_at or "--",
            "targets": int(self.target_count),
            "interface_status": self.interface_status,
            "interface_coverage": float(self.interface_coverage),
            "interface_keypoints": int(self.interface_keypoint_count),
            "source": self.raw_path,
            "rows": int(self.raw_rows),
            "traces": int(self.trace_count),
            "raw_size_mb": float(self.raw_size_mb),
            "sensor_sync_status": self.sensor_sync_status,
        }


@dataclass
class FieldProjectManifest:
    schema: str = FIELD_PROJECT_SCHEMA
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_no: str = "PROJ-2025-0518-001"
    name: str = "基覆界面识别项目"
    location: str = "未设置"
    device_model: str = "无人机/半航空 GPR + RTK/IMU"
    coordinate_system: str = "CGCS2000 / 3-degree GK"
    vertical_datum: str = "1985 国家高程基准"
    coordinate_crs_wkt: str = ""
    vertical_crs_wkt: str = ""
    revision: int = 0
    storage_policy: dict[str, Any] = field(default_factory=lambda: {
        "single_writer": True,
        "atomic_commit": True,
        "bounded_memory": True,
        "immutable_source_files": True,
        "immutable_raw": False,
        "normalized_raw_write_policy": "controlled_replace_with_backup",
    })
    storage_backend: str = "hybrid_hdf5_sqlite_v1"
    catalog_path: str = "catalog.sqlite"
    line_container_pattern: str = "data/lines/{line_id}.h5"
    legacy_layout: bool = False
    created_at: str = field(default_factory=local_now)
    updated_at: str = field(default_factory=local_now)
    operator: str = "操作员"
    status: str = "正常"
    lines: list[dict[str, Any]] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=lambda: {"status": "未生成", "file_count": 0})

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FieldProjectManifest":
        schema = str(payload.get("schema") or "")
        if schema != FIELD_PROJECT_SCHEMA:
            prefix = "mygpr.field_project.v"
            try:
                version = int(schema[len(prefix):]) if schema.startswith(prefix) else -1
            except ValueError:
                version = -1
            # Newer project documents may be inspected in read-only mode.  The
            # registry owns the downgrade decision; the model only accepts the
            # known field subset without rewriting the source schema.
            if version < 2:
                raise ValueError(f"Unsupported field project schema: {schema!r}")
        allowed = {name for name in cls.__dataclass_fields__}
        data = {key: value for key, value in payload.items() if key in allowed}
        lines = data.get("lines", [])
        if not isinstance(lines, list):
            raise ValueError("Project manifest lines must be a list")
        data["lines"] = [asdict(FieldLineRecord.from_dict(item)) for item in lines]
        return cls(**data)

    def line_records(self) -> list[FieldLineRecord]:
        return [FieldLineRecord.from_dict(item) for item in self.lines]

    def set_lines(self, lines: Iterable[FieldLineRecord]) -> None:
        self.lines = [asdict(line) for line in lines]
        self.updated_at = local_now()


__all__ = [
    "FIELD_PROJECT_SCHEMA",
    "TARGET_FIELDS",
    "FieldLineRecord",
    "FieldProjectManifest",
    "LINE_ID_PATTERN",
    "LineStatus",
    "atomic_write_json",
    "atomic_write_text",
    "count_csv_rows",
    "validate_line_id",
    "local_now",
]
