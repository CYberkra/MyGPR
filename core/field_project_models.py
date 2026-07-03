#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared data models and atomic file helpers for the field project stores."""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass

FIELD_PROJECT_SCHEMA = "mygpr.field_project.v1"
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
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class LineStatus(StrEnum):
    PENDING = "--"
    DRAFT = "草稿"
    IMPORTED = "已导入"
    PROCESSED = "已处理"
    COMPLETED = "已完成"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


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
    data_quality: LineStatus = LineStatus.PENDING
    rtk_status: str = "未定位"
    processing_status: str = "未处理"
    updated_at: str = ""
    raw_path: str = ""
    raw_rows: int = 0
    raw_size_mb: float = 0.0
    target_count: int = 0
    processed_result: str = ""
    params_path: str = ""
    gpr_dataset_path: str = ""
    trajectory_path: str = ""
    data_format: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FieldLineRecord":
        return cls(**{k: payload.get(k, getattr(cls, k, None)) for k in cls.__dataclass_fields__})

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
            "source": self.raw_path,
            "rows": int(self.raw_rows),
        }


@dataclass
class FieldProjectManifest:
    schema: str = FIELD_PROJECT_SCHEMA
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_no: str = "PROJ-2025-0518-001"
    name: str = "城市道路地下目标定位项目"
    location: str = "江苏省南京市建邺区"
    device_model: str = "IDS Stream DP Pro + CX-RTK2"
    coordinate_system: str = "CGCS2000 / 3-degree GK"
    vertical_datum: str = "1985 国家高程基准"
    created_at: str = field(default_factory=local_now)
    updated_at: str = field(default_factory=local_now)
    operator: str = "操作员"
    status: str = "正常"
    lines: list[dict[str, Any]] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=lambda: {"status": "未生成", "file_count": 0})

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FieldProjectManifest":
        if payload.get("schema") != FIELD_PROJECT_SCHEMA:
            raise ValueError(f"Unsupported field project schema: {payload.get('schema')!r}")
        allowed = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in payload.items() if key in allowed})

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
