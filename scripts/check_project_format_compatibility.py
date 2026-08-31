#!/usr/bin/env python3
"""Verify the declared project-format compatibility contract against code."""
from __future__ import annotations

import json
from pathlib import Path

from core.field_project_models import FIELD_PROJECT_SCHEMA
from core.schema_registry import DEFAULT_SCHEMA_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "project_format_compatibility.json"


def validate_contract(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != "mygpr.project_format_compatibility.v1":
        errors.append("invalid compatibility schema")
    family = str(payload.get("family") or "")
    try:
        definition = DEFAULT_SCHEMA_REGISTRY.definition(family)
    except Exception as exc:  # converted into a deterministic gate error
        return [f"unregistered project schema family: {family}: {exc}"]
    expected_current = definition.current_schema
    if payload.get("current_schema") != expected_current:
        errors.append(f"current_schema mismatch: {payload.get('current_schema')} != {expected_current}")
    if FIELD_PROJECT_SCHEMA != expected_current:
        errors.append(f"FIELD_PROJECT_SCHEMA mismatch: {FIELD_PROJECT_SCHEMA} != {expected_current}")
    minimum = int(payload.get("minimum_migratable_version", 0))
    if minimum < 1 or minimum > definition.current_version:
        errors.append("invalid minimum_migratable_version")
    missing = [version for version in range(minimum, definition.current_version) if version not in definition.migrations]
    if missing:
        errors.append(f"missing migration steps: {missing}")
    if payload.get("newer_schema_policy") != "read_only":
        errors.append("newer schema policy must remain read_only")
    if payload.get("snapshot_before_migration") is not True:
        errors.append("migration snapshots must remain enabled")
    required = payload.get("required_fields_after_migration")
    if not isinstance(required, list) or not all(isinstance(item, str) and item for item in required):
        errors.append("required_fields_after_migration must be a non-empty string list")
    return errors


def main() -> int:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    errors = validate_contract(payload)
    if errors:
        print("\n".join(errors))
        return 1
    print("project format compatibility: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
