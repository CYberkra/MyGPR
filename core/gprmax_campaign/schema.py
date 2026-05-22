#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schema dataclasses for gprMax campaign dry-run validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VALIDATION_READY = "ready"
VALIDATION_WARNING = "warning"
VALIDATION_INVALID = "invalid"


@dataclass(frozen=True)
class CampaignScene:
    """One simulation scene entry from campaign YAML."""

    scene_id: str
    raw_model: Path
    background_model: Path
    materials: Path
    target_roi: Path
    expected_outputs: Any
    tags: list[str]
    description: str = ""


@dataclass(frozen=True)
class Campaign:
    """Loaded campaign object."""

    campaign_id: str
    output_root: Path
    gprmax_executable: str
    scenes: list[CampaignScene]
    source_path: Path


@dataclass(frozen=True)
class ValidationIssue:
    """One validation issue (warning/error) at campaign or scene level."""

    level: str
    code: str
    message: str
    scene_id: str | None = None
    field: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "scene_id": self.scene_id,
            "field": self.field,
            "path": self.path,
        }


@dataclass
class SceneValidationResult:
    """Validation result for a single scene."""

    scene_id: str
    status: str
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class CampaignValidationResult:
    """Validation result for the full campaign."""

    campaign_id: str
    status: str
    total_scenes: int
    ready_count: int
    warning_count: int
    invalid_count: int
    scenes: list[SceneValidationResult] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "status": self.status,
            "total_scenes": self.total_scenes,
            "ready_count": self.ready_count,
            "warning_count": self.warning_count,
            "invalid_count": self.invalid_count,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "issues": [issue.to_dict() for issue in self.issues],
        }
