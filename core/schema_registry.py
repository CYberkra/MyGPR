#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central schema registration, validation, migration and quarantine support."""
from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from core.storage_primitives import atomic_write_json, utc_now

_SCHEMA_RE = re.compile(r"^(?P<family>mygpr\.[a-zA-Z0-9_.-]+)\.v(?P<version>\d+)$")


class SchemaError(RuntimeError):
    pass


class DocumentCorruptionError(SchemaError):
    def __init__(self, path: Path, message: str, *, quarantine_path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.quarantine_path = quarantine_path


class UnsupportedSchemaError(SchemaError):
    pass


class NewerSchemaReadOnly(SchemaError):
    pass


Migration = Callable[[dict[str, Any]], dict[str, Any]]
Validator = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class SchemaDefinition:
    family: str
    current_version: int
    validator: Validator | None = None
    migrations: Mapping[int, Migration] = field(default_factory=dict)
    owner: str = "core"

    @property
    def current_schema(self) -> str:
        return f"{self.family}.v{self.current_version}"


@dataclass(frozen=True)
class LoadedDocument:
    payload: dict[str, Any]
    source_schema: str
    current_schema: str
    migrated: bool = False
    read_only: bool = False


class SchemaRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, SchemaDefinition] = {}

    def register(self, definition: SchemaDefinition) -> None:
        existing = self._definitions.get(definition.family)
        if existing is not None and existing != definition:
            raise SchemaError(f"Schema family already registered: {definition.family}")
        self._definitions[definition.family] = definition

    def definition(self, family: str) -> SchemaDefinition:
        try:
            return self._definitions[family]
        except KeyError as exc:
            raise UnsupportedSchemaError(f"Unregistered schema family: {family}") from exc

    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    @staticmethod
    def parse(schema: str) -> tuple[str, int]:
        match = _SCHEMA_RE.match(str(schema or ""))
        if not match:
            raise UnsupportedSchemaError(f"Invalid schema identifier: {schema!r}")
        return match.group("family"), int(match.group("version"))

    def migrate(self, payload: Mapping[str, Any], *, family: str) -> LoadedDocument:
        definition = self.definition(family)
        source_schema = str(payload.get("schema") or "")
        source_family, source_version = self.parse(source_schema)
        if source_family != family:
            raise UnsupportedSchemaError(f"Expected {family}, got {source_family}")
        if source_version > definition.current_version:
            return LoadedDocument(dict(payload), source_schema, definition.current_schema, read_only=True)
        current = dict(payload)
        version = source_version
        while version < definition.current_version:
            migration = definition.migrations.get(version)
            if migration is None:
                raise UnsupportedSchemaError(
                    f"Missing migration {family}.v{version} -> v{version + 1}"
                )
            current = dict(migration(current))
            version += 1
            current["schema"] = f"{family}.v{version}"
        if definition.validator is not None:
            definition.validator(current)
        return LoadedDocument(
            current,
            source_schema=source_schema,
            current_schema=definition.current_schema,
            migrated=source_version != definition.current_version,
        )

    def load_path(
        self,
        path: str | Path,
        *,
        family: str,
        write_migrated: bool = True,
        quarantine_root: str | Path | None = None,
    ) -> LoadedDocument:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("JSON root must be an object")
            loaded = self.migrate(payload, family=family)
        except (OSError, ValueError, TypeError, SchemaError) as exc:
            quarantine = self.quarantine(source, quarantine_root=quarantine_root, reason=str(exc))
            if isinstance(exc, UnsupportedSchemaError):
                raise
            raise DocumentCorruptionError(
                source,
                f"文档损坏或无法验证：{source}: {exc}",
                quarantine_path=quarantine,
            ) from exc
        if loaded.migrated and write_migrated and not loaded.read_only:
            self.snapshot_before_migration(source, payload)
            atomic_write_json(source, loaded.payload)
        return loaded

    @staticmethod
    def snapshot_before_migration(path: Path, payload: Mapping[str, Any]) -> Path:
        root = path.parent / ".schema_snapshots"
        root.mkdir(parents=True, exist_ok=True)
        snapshot = root / f"{path.name}.{utc_now().replace(':', '').replace('+', '_')}.{uuid.uuid4().hex[:8]}.json"
        atomic_write_json(snapshot, dict(payload))
        return snapshot

    @staticmethod
    def quarantine(path: Path, *, quarantine_root: str | Path | None = None, reason: str = "") -> Path | None:
        if not path.exists():
            return None
        root = Path(quarantine_root) if quarantine_root is not None else path.parent / "quarantine"
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{path.name}.{utc_now().replace(':', '').replace('+', '_')}.{uuid.uuid4().hex[:8]}.corrupt"
        shutil.copy2(path, target)
        atomic_write_json(target.with_suffix(target.suffix + ".reason.json"), {
            "schema": "mygpr.quarantine_record.v1",
            "source": str(path),
            "quarantine": str(target),
            "reason": str(reason),
            "created_at": utc_now(),
        })
        return target


DEFAULT_SCHEMA_REGISTRY = SchemaRegistry()


def _require_keys(*keys: str) -> Validator:
    def validate(payload: Mapping[str, Any]) -> None:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise SchemaError(f"Missing required fields: {', '.join(missing)}")
    return validate


def _migrate_field_project_v1(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated.setdefault("coordinate_crs_wkt", "")
    migrated.setdefault("vertical_crs_wkt", "")
    migrated.setdefault("revision", 0)
    migrated.setdefault("storage_policy", {"single_writer": True, "atomic_commit": True})
    return migrated




def _migrate_field_project_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """Upgrade metadata without silently moving any measurement arrays."""
    migrated = dict(payload)
    migrated.setdefault("storage_backend", "legacy_files_v2")
    migrated.setdefault("catalog_path", "")
    migrated.setdefault("line_container_pattern", "")
    migrated.setdefault("legacy_layout", True)
    policy = dict(migrated.get("storage_policy") or {})
    policy.setdefault("single_writer", True)
    policy.setdefault("atomic_commit", True)
    policy.setdefault("bounded_memory", True)
    policy.setdefault("immutable_source_files", True)
    policy.setdefault("immutable_raw", False)
    policy.setdefault("normalized_raw_write_policy", "controlled_replace_with_backup")
    migrated["storage_policy"] = policy
    return migrated

def _migrate_sensor_sync_v1(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    config = dict(migrated.get("config") or {})
    for name in ("radar", "rtk", "imu", "altimeter"):
        config.setdefault(f"{name}_clock", {"name": name, "epoch": "relative_start", "unit": "s", "time_scale": "device"})
    migrated["config"] = config
    return migrated


def _migrate_gis_layers_v1(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    layers = []
    for row in migrated.get("layers", []):
        item = dict(row)
        item.setdefault("source_sha256", "")
        item.setdefault("style_version", 1)
        item.setdefault("z_order", 0)
        item.setdefault("vertical_crs", "")
        item.setdefault("lineage", {})
        layers.append(item)
    migrated["layers"] = layers
    return migrated


def _migrate_source_files_v1(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    rows = []
    for row in migrated.get("sources", []):
        item = dict(row)
        item.setdefault("quick_sha256", "")
        item.setdefault("chunk_hashes", [])
        item.setdefault("merkle_root", "")
        item.setdefault("full_hash_status", "complete" if item.get("sha256") else "pending")
        rows.append(item)
    migrated["sources"] = rows
    return migrated


def _migrate_job_journal_v1(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    jobs = []
    for row in migrated.get("jobs", []):
        item = dict(row)
        item.setdefault("diagnostic_id", "")
        item.setdefault("resources", [])
        item.setdefault("dependencies", [])
        item.setdefault("worker_kind", "io_thread")
        item.setdefault("checkpoint", {})
        item.setdefault("retry_limit", 0)
        jobs.append(item)
    migrated["jobs"] = jobs
    return migrated


for _definition in (
    SchemaDefinition("mygpr.field_project", 3, _require_keys("project_id", "name", "lines", "revision", "storage_backend"), {1: _migrate_field_project_v1, 2: _migrate_field_project_v2}, owner="project"),
    SchemaDefinition("mygpr.project", 1, _require_keys("project_id", "name"), owner="compatibility"),
    SchemaDefinition("mygpr.project_state", 1, _require_keys("data_revision", "dirty"), owner="project"),
    SchemaDefinition("mygpr.workspace_context", 1, _require_keys("active_workspace", "selected_line_id"), owner="project"),
    SchemaDefinition("mygpr.project_integrity_report", 1, _require_keys("project_root", "summary", "issues"), owner="project"),
    SchemaDefinition("mygpr.project_runtime_session", 1, _require_keys("session_id", "state", "opened_at"), owner="project"),
    SchemaDefinition("mygpr.release_hardening_evidence", 1, _require_keys("checks", "summary"), owner="release"),
    SchemaDefinition("mygpr.module_linkage_evidence", 1, _require_keys("stages", "screenshots"), owner="ui"),
    SchemaDefinition("mygpr.sensor_sync", 2, migrations={1: _migrate_sensor_sync_v1}, owner="sync"),
    SchemaDefinition("mygpr.gis_layers", 2, _require_keys("layers"), {1: _migrate_gis_layers_v1}, owner="gis"),
    SchemaDefinition("mygpr.source_files", 2, _require_keys("sources"), {1: _migrate_source_files_v1}, owner="storage"),
    SchemaDefinition("mygpr.job_journal", 2, _require_keys("jobs"), {1: _migrate_job_journal_v1}, owner="jobs"),
    SchemaDefinition("mygpr.report_package", 3, owner="reporting"),
    SchemaDefinition("mygpr.report_manifest", 4, owner="reporting"),
):
    DEFAULT_SCHEMA_REGISTRY.register(_definition)



__all__ = [
    "DEFAULT_SCHEMA_REGISTRY",
    "DocumentCorruptionError",
    "LoadedDocument",
    "NewerSchemaReadOnly",
    "SchemaDefinition",
    "SchemaError",
    "SchemaRegistry",
    "UnsupportedSchemaError",
]
