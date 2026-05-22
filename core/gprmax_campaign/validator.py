#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dry-run validator for gprMax campaign scenes."""

from __future__ import annotations

import os
from pathlib import Path

from core.gprmax_campaign.schema import (
    Campaign,
    CampaignValidationResult,
    SceneValidationResult,
    ValidationIssue,
    VALIDATION_INVALID,
    VALIDATION_READY,
    VALIDATION_WARNING,
)


_REQUIRED_EXPECTED_OUTPUTS = {
    "raw_with_target",
    "background_only",
    "target_response",
}


def validate_campaign(campaign: Campaign) -> CampaignValidationResult:
    """Validate campaign and scene readiness without running gprMax."""
    campaign_issues: list[ValidationIssue] = []
    scene_results: list[SceneValidationResult] = []

    campaign_id = str(campaign.campaign_id or "").strip()
    if not campaign_id:
        campaign_issues.append(
            ValidationIssue(
                level="error",
                code="campaign_id_missing",
                message="campaign_id must be non-empty",
                field="campaign_id",
            )
        )

    _validate_output_root(campaign.output_root, campaign_issues)

    if not str(campaign.gprmax_executable or "").strip():
        campaign_issues.append(
            ValidationIssue(
                level="error",
                code="gprmax_executable_missing",
                message="gprmax_executable must be non-empty",
                field="gprmax_executable",
            )
        )

    scene_id_counts: dict[str, int] = {}
    for scene in campaign.scenes:
        scene_key = str(scene.scene_id or "").strip()
        if scene_key:
            scene_id_counts[scene_key] = scene_id_counts.get(scene_key, 0) + 1

    for index, scene in enumerate(campaign.scenes):
        scene_results.append(
            _validate_scene(scene=scene, index=index, scene_id_counts=scene_id_counts)
        )

    ready_count = sum(1 for item in scene_results if item.status == VALIDATION_READY)
    warning_count = sum(
        1 for item in scene_results if item.status == VALIDATION_WARNING
    )
    invalid_count = sum(
        1 for item in scene_results if item.status == VALIDATION_INVALID
    )

    campaign_has_error = any(issue.level == "error" for issue in campaign_issues)
    campaign_has_warning = any(issue.level == "warning" for issue in campaign_issues)
    if campaign_has_error or invalid_count > 0:
        status = VALIDATION_INVALID
    elif campaign_has_warning or warning_count > 0:
        status = VALIDATION_WARNING
    else:
        status = VALIDATION_READY

    return CampaignValidationResult(
        campaign_id=campaign.campaign_id,
        status=status,
        total_scenes=len(scene_results),
        ready_count=ready_count,
        warning_count=warning_count,
        invalid_count=invalid_count,
        scenes=scene_results,
        issues=campaign_issues,
    )


def _validate_output_root(output_root: Path, issues: list[ValidationIssue]) -> None:
    root = Path(output_root).expanduser().resolve()
    if str(root).strip() == "":
        issues.append(
            ValidationIssue(
                level="error",
                code="output_root_missing",
                message="output_root must be provided",
                field="output_root",
            )
        )
        return

    if root.exists() and not root.is_dir():
        issues.append(
            ValidationIssue(
                level="error",
                code="output_root_not_directory",
                message="output_root exists but is not a directory",
                field="output_root",
                path=str(root),
            )
        )
        return

    writable_target = root if root.exists() else _nearest_existing_parent(root)
    if writable_target is None:
        issues.append(
            ValidationIssue(
                level="error",
                code="output_root_parent_missing",
                message="output_root parent does not exist and cannot be resolved",
                field="output_root",
                path=str(root),
            )
        )
        return

    if not os.access(str(writable_target), os.W_OK):
        issues.append(
            ValidationIssue(
                level="error",
                code="output_root_not_writable",
                message="output_root is not writable and parent is not writable",
                field="output_root",
                path=str(writable_target),
            )
        )


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while True:
        if current.exists():
            return current if current.is_dir() else current.parent
        if current.parent == current:
            return None
        current = current.parent


def _validate_scene(
    scene,
    *,
    index: int,
    scene_id_counts: dict[str, int],
) -> SceneValidationResult:
    issues: list[ValidationIssue] = []
    scene_id = str(scene.scene_id or "").strip() or f"<scene_{index}>"

    if not str(scene.scene_id or "").strip():
        issues.append(
            ValidationIssue(
                level="error",
                code="scene_id_missing",
                message="scene_id must be non-empty",
                scene_id=scene_id,
                field="scene_id",
            )
        )

    if scene_id_counts.get(scene_id, 0) > 1:
        issues.append(
            ValidationIssue(
                level="error",
                code="scene_id_duplicate",
                message=f"scene_id is duplicated: {scene_id}",
                scene_id=scene_id,
                field="scene_id",
            )
        )

    _validate_file_path(
        scene_id=scene_id,
        field="raw_model",
        path=scene.raw_model,
        required_suffixes={".in"},
        issues=issues,
    )
    _validate_file_path(
        scene_id=scene_id,
        field="background_model",
        path=scene.background_model,
        required_suffixes={".in"},
        issues=issues,
    )
    _validate_file_path(
        scene_id=scene_id,
        field="materials",
        path=scene.materials,
        required_suffixes={".txt", ".json"},
        issues=issues,
    )
    _validate_file_path(
        scene_id=scene_id,
        field="target_roi",
        path=scene.target_roi,
        required_suffixes={".json", ".yaml", ".yml"},
        issues=issues,
    )

    if Path(scene.raw_model).resolve() == Path(scene.background_model).resolve():
        issues.append(
            ValidationIssue(
                level="error",
                code="scene_pair_paths_equal",
                message="raw_model and background_model must be different paths",
                scene_id=scene_id,
                field="background_model",
            )
        )

    missing_outputs = _missing_expected_outputs(scene.expected_outputs)
    if missing_outputs:
        issues.append(
            ValidationIssue(
                level="error",
                code="expected_outputs_missing",
                message="expected_outputs missing required keys: "
                + ", ".join(sorted(missing_outputs)),
                scene_id=scene_id,
                field="expected_outputs",
            )
        )

    has_error = any(item.level == "error" for item in issues)
    has_warning = any(item.level == "warning" for item in issues)
    if has_error:
        status = VALIDATION_INVALID
    elif has_warning:
        status = VALIDATION_WARNING
    else:
        status = VALIDATION_READY
    return SceneValidationResult(scene_id=scene_id, status=status, issues=issues)


def _validate_file_path(
    *,
    scene_id: str,
    field: str,
    path: Path,
    required_suffixes: set[str],
    issues: list[ValidationIssue],
) -> None:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        issues.append(
            ValidationIssue(
                level="error",
                code=f"{field}_missing",
                message=f"{field} does not exist",
                scene_id=scene_id,
                field=field,
                path=str(file_path),
            )
        )
        return
    suffix = file_path.suffix.lower()
    if suffix not in required_suffixes:
        issues.append(
            ValidationIssue(
                level="warning",
                code=f"{field}_suffix_unexpected",
                message=(
                    f"{field} suffix '{suffix or '<none>'}' is unusual; expected "
                    + "/".join(sorted(required_suffixes))
                ),
                scene_id=scene_id,
                field=field,
                path=str(file_path),
            )
        )


def _missing_expected_outputs(expected_outputs) -> set[str]:
    if isinstance(expected_outputs, dict):
        keys = {str(key) for key in expected_outputs.keys()}
        return _REQUIRED_EXPECTED_OUTPUTS - keys
    if isinstance(expected_outputs, list):
        keys = {str(item) for item in expected_outputs}
        return _REQUIRED_EXPECTED_OUTPUTS - keys
    return set(_REQUIRED_EXPECTED_OUTPUTS)
