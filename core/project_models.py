#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned, JSON-friendly project records for the MyGPR workbench."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RawFileRef:
    path: str
    role: str = "primary"
    size_bytes: int = 0
    sha256: str | None = None
    integrity_status: str = "pending"
    source_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RawFileRef":
        return cls(**payload)


@dataclass
class LineRecordV1:
    line_id: str
    name: str
    raw_files: list[RawFileRef] = field(default_factory=list)
    sidecars: dict[str, str] = field(default_factory=dict)
    status: str = "imported"
    source_format: str = "unknown"
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LineRecordV1":
        data = dict(payload)
        data["raw_files"] = [RawFileRef.from_dict(item) for item in data.get("raw_files", [])]
        return cls(**data)


@dataclass
class ProjectManifestV1:
    project_id: str
    name: str
    temporary: bool = False
    schema: str = "mygpr.project.v1"
    created_at: str = ""
    updated_at: str = ""
    coordinate_reference: str | None = None
    task_conditions: dict[str, Any] = field(default_factory=dict)
    line_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectManifestV1":
        return cls(**payload)


@dataclass
class QcItem:
    code: str
    severity: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledgement_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QcReportV1:
    line_id: str
    items: list[QcItem]
    created_at: str
    schema: str = "mygpr.qc_report.v1"

    @property
    def can_process(self) -> bool:
        return not any(item.severity == "error" for item in self.items)

    @property
    def requires_review(self) -> bool:
        return any(
            item.severity == "warning" and not item.acknowledged for item in self.items
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["can_process"] = self.can_process
        payload["requires_review"] = self.requires_review
        return payload


@dataclass
class ProcessingResultV1:
    result_id: str
    line_id: str
    name: str
    data_path: str
    processing_chain: list[dict[str, Any]]
    trace_metadata_path: str | None = None
    header_info: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    schema: str = "mygpr.processing_result.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProcessingResultV1":
        return cls(**payload)


@dataclass
class InterpretationFeatureV1:
    feature_id: str
    line_id: str
    feature_type: str
    geometry: dict[str, Any]
    confidence: float
    result_id: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    schema: str = "mygpr.interpretation_feature.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterpretationFeatureV1":
        return cls(**payload)


__all__ = [
    "InterpretationFeatureV1",
    "LineRecordV1",
    "ProcessingResultV1",
    "ProjectManifestV1",
    "QcItem",
    "QcReportV1",
    "RawFileRef",
]
