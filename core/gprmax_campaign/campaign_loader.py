#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load gprMax campaign YAML into typed campaign dataclasses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.gprmax_campaign.schema import Campaign, CampaignScene


def load_campaign_yaml(path: str | Path) -> Campaign:
    """Load and parse one campaign YAML file."""
    yaml = _require_yaml_module()
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Campaign YAML not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Campaign YAML path must be a file: {source_path}")

    try:
        with source_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - defensive YAML parser errors
        raise ValueError(f"Failed to parse campaign YAML: {source_path}; {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Campaign YAML must contain a mapping: {source_path}")

    campaign_id = _require_non_empty_str(payload, "campaign_id")
    output_root = _resolve_path_value(
        _require_non_empty_str(payload, "output_root"),
        source_path.parent,
    )
    gprmax_executable = _require_non_empty_str(payload, "gprmax_executable")

    scenes_payload = payload.get("scenes")
    if not isinstance(scenes_payload, list) or not scenes_payload:
        raise ValueError("Campaign YAML 'scenes' must be a non-empty list")

    scenes: list[CampaignScene] = []
    for idx, scene_payload in enumerate(scenes_payload):
        if not isinstance(scene_payload, dict):
            raise ValueError(f"Scene item at index {idx} must be a mapping")
        scene_id = _require_non_empty_str(scene_payload, "scene_id")
        raw_model = _resolve_path_value(
            _require_non_empty_str(scene_payload, "raw_model"),
            source_path.parent,
        )
        background_model = _resolve_path_value(
            _require_non_empty_str(scene_payload, "background_model"),
            source_path.parent,
        )
        materials = _resolve_path_value(
            _require_non_empty_str(scene_payload, "materials"),
            source_path.parent,
        )
        target_roi = _resolve_path_value(
            _require_non_empty_str(scene_payload, "target_roi"),
            source_path.parent,
        )
        if "expected_outputs" not in scene_payload:
            raise ValueError(
                f"Scene '{scene_id}' missing required field: expected_outputs"
            )
        expected_outputs = scene_payload.get("expected_outputs")
        tags_raw = scene_payload.get("tags")
        if tags_raw is None:
            tags: list[str] = []
        elif isinstance(tags_raw, list):
            tags = [str(item) for item in tags_raw]
        else:
            raise ValueError(f"Scene '{scene_id}' field 'tags' must be a list")
        description = str(scene_payload.get("description") or "")
        scenes.append(
            CampaignScene(
                scene_id=scene_id,
                raw_model=raw_model,
                background_model=background_model,
                materials=materials,
                target_roi=target_roi,
                expected_outputs=expected_outputs,
                tags=tags,
                description=description,
            )
        )

    return Campaign(
        campaign_id=campaign_id,
        output_root=output_root,
        gprmax_executable=gprmax_executable,
        scenes=scenes,
        source_path=source_path,
    )


def _require_yaml_module():
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for gprMax campaign YAML loading. "
            "Please install it in the MyGPR environment."
        ) from exc
    return yaml


def _require_non_empty_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Campaign field '{key}' must be a non-empty string")
    return value.strip()


def _resolve_path_value(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()
