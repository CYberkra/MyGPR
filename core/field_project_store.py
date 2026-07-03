#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Field workbench project manifest store for the MyGPR product UI.

The project store is intentionally a small coordinator.  Line data, processed
artifacts, target CSV files, spatial exports and demo bootstrap helpers live in
separate mixins so new product features do not keep expanding this file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.app_errors import error_info_from_exception
from core.field_artifact_store import FieldArtifactStoreMixin
from core.field_demo_store import FieldDemoStoreMixin
from core.field_line_store import FieldLineStoreMixin
from core.field_project_models import (
    FIELD_PROJECT_SCHEMA,
    TARGET_FIELDS,
    FieldLineRecord,
    FieldProjectManifest,
    atomic_write_json,
    atomic_write_text,
    local_now,
)
from core.field_spatial_store import FieldSpatialStoreMixin
from core.field_target_store import FieldTargetStoreMixin

logger = logging.getLogger(__name__)


class FieldProjectStore(
    FieldLineStoreMixin,
    FieldTargetStoreMixin,
    FieldSpatialStoreMixin,
    FieldArtifactStoreMixin,
    FieldDemoStoreMixin,
):
    """Durable project manifest store for the field workbench UI.

    Standard project layout::

        project.json
        raw/
        processed/
        targets/
        spatial/
        reports/
        logs/
    """

    MANIFEST_NAME = "project.json"
    DIRECTORIES = ("raw", "processed", "targets", "spatial", "reports", "logs", "metadata")

    def __init__(self, root: str | Path, manifest: FieldProjectManifest) -> None:
        self.root = Path(root).resolve()
        self.manifest = manifest

    @staticmethod
    def now() -> str:
        return local_now()

    @staticmethod
    def write_text(path: Path, text: str) -> None:
        atomic_write_text(path, text)

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        atomic_write_json(path, payload)

    @classmethod
    def open(cls, root: str | Path) -> "FieldProjectStore":
        root_path = Path(root).resolve()
        try:
            payload = json.loads((root_path / cls.MANIFEST_NAME).read_text(encoding="utf-8"))
            store = cls(root_path, FieldProjectManifest.from_dict(payload))
            store.ensure_structure()
            return store
        except Exception as exc:
            error_info = error_info_from_exception(exc, category="io")
            logger.error("Failed to open project: %s [%s]", error_info.user_message, error_info.error_code)
            raise

    @classmethod
    def create_empty(
        cls,
        root: str | Path,
        *,
        name: str = "新建 MyGPR 项目",
        location: str = "",
        operator: str = "操作员",
        project_no: str = "",
        device_model: str = "",
        coordinate_system: str = "",
        vertical_datum: str = "",
    ) -> "FieldProjectStore":
        """Create a formal empty project without demo lines or synthetic data."""
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        manifest = FieldProjectManifest(
            name=name,
            location=location or "未填写",
            operator=operator or "操作员",
            lines=[],
        )
        if project_no.strip():
            manifest.project_no = project_no.strip()
        if device_model.strip():
            manifest.device_model = device_model.strip()
        if coordinate_system.strip():
            manifest.coordinate_system = coordinate_system.strip()
        if vertical_datum.strip():
            manifest.vertical_datum = vertical_datum.strip()
        store = cls(root_path, manifest)
        store.ensure_structure()
        store.save_manifest()
        store.append_log(f"创建项目：{manifest.name}")
        return store

    @classmethod
    def create(cls, root: str | Path, *, sample_csv: str | Path | None = None) -> "FieldProjectStore":
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        manifest = FieldProjectManifest()
        store = cls(root_path, manifest)
        store.ensure_structure()
        store.manifest.set_lines(store._default_lines())
        store.save_manifest()
        if sample_csv is not None and Path(sample_csv).exists():
            store.import_line_file("L03", Path(sample_csv), name="过路口测线", copy_into_project=True)
        store.ensure_demo_gpr_artifacts("L03")
        if not store.targets_path("L03").exists():
            store.save_targets("L03", store.default_targets("L03"))
        store.export_spatial_targets_xy("L03")
        return store

    @classmethod
    def create_or_open_demo(cls, repo_root: str | Path, *, sample_csv: str | Path | None = None) -> "FieldProjectStore":
        root = Path(repo_root).resolve() / "runtime_projects" / "field_demo_project"
        if (root / cls.MANIFEST_NAME).exists():
            store = cls.open(root)
            store.ensure_demo_artifacts(sample_csv=sample_csv)
            return store
        return cls.create(root, sample_csv=sample_csv)

    def ensure_structure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in self.DIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def save_manifest(self) -> None:
        try:
            self.manifest.updated_at = local_now()
            atomic_write_json(self.root / self.MANIFEST_NAME, asdict(self.manifest))
        except Exception as exc:
            error_info = error_info_from_exception(exc, category="io")
            logger.error("Failed to save project manifest: %s [%s]", error_info.user_message, error_info.error_code)
            raise


__all__ = [
    "FIELD_PROJECT_SCHEMA",
    "FieldLineRecord",
    "FieldProjectManifest",
    "FieldProjectStore",
    "TARGET_FIELDS",
    "local_now",
]
