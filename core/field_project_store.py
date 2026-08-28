#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Field workbench project manifest store for the MyGPR product UI.

The project store is intentionally a small coordinator.  Line data, processed
artifacts, target CSV files and spatial exports live in separate mixins so new
product features do not keep expanding this file.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mygpr.domain.common.errors import error_info_from_exception
from core.field_artifact_store import FieldArtifactStoreMixin
from core.field_interface_store import FieldInterfaceStoreMixin
from core.field_line_store import FieldLineStoreMixin
from core.field_processing_draft_store import FieldProcessingDraftStoreMixin
from core.field_project_models import (
    FIELD_PROJECT_SCHEMA, TARGET_FIELDS, FieldLineRecord, FieldProjectManifest,
    atomic_write_json, atomic_write_text, local_now,
)
from core.field_project_protocol import FieldProjectStoreProtocol
from core.field_project_runtime_store import FieldProjectRuntimeStoreMixin
from core.field_spatial_store import FieldSpatialStoreMixin
from core.project_repository import ProjectAccessMode, ProjectRepository, ProjectSession
from core.project_root_guard import ensure_project_root_marker
from core.project_storage_backend import create_storage_backend, HYBRID_STORAGE_BACKEND
from core.schema_registry import DEFAULT_SCHEMA_REGISTRY
from core.field_target_store import FieldTargetStoreMixin

logger = logging.getLogger(__name__)


class FieldProjectStore(
    FieldLineStoreMixin, FieldInterfaceStoreMixin, FieldProcessingDraftStoreMixin,
    FieldTargetStoreMixin, FieldSpatialStoreMixin, FieldArtifactStoreMixin,
    FieldProjectRuntimeStoreMixin,
):
    """Durable project-manifest coordinator (see ``FieldProjectStoreProtocol``)."""

    MANIFEST_NAME = "project.json"
    DIRECTORIES = (
        "raw", "processed", "targets", "spatial", "reports",  # compatibility/export surfaces
        "data/lines", "attachments", "exports", "cache/previews", "cache/staging",
        "logs", "metadata", "backups",
    )

    def __init__(self, root: str | Path, manifest: FieldProjectManifest, *, session: ProjectSession | None = None) -> None:
        self.root = Path(root).resolve()
        self.manifest = manifest
        self.session = session or ProjectRepository.open_session(self.root, mode=ProjectAccessMode.AUTO)
        self.storage = create_storage_backend(self.root, manifest, read_only=self.session.read_only)

    @property
    def read_only(self) -> bool:
        return bool(self.session.read_only)

    def assert_writable(self) -> None:
        self.session.assert_writable()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "FieldProjectStore": return self
    def __exit__(self, exc_type, exc, tb) -> None: self.close()

    @staticmethod
    def now() -> str:
        return local_now()

    def write_text(self, path: Path, text: str) -> None:
        self.assert_writable()
        atomic_write_text(path, text)

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self.assert_writable()
        atomic_write_json(path, payload)

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        access_mode: str | ProjectAccessMode = ProjectAccessMode.AUTO,
        recover_stale_lock: bool = False,
    ) -> "FieldProjectStore":
        root_path = Path(root).resolve()
        session = ProjectRepository.open_session(root_path, mode=access_mode, recover_stale=recover_stale_lock)
        try:
            loaded = DEFAULT_SCHEMA_REGISTRY.load_path(
                root_path / cls.MANIFEST_NAME,
                family="mygpr.field_project",
                write_migrated=not session.read_only,
                quarantine_root=root_path / "metadata" / "quarantine",
            )
            if loaded.read_only and not session.read_only:
                session.close()
                session = ProjectRepository.open_session(root_path, mode=ProjectAccessMode.READ_ONLY)
            store = cls(root_path, FieldProjectManifest.from_dict(loaded.payload), session=session)
            if not store.read_only:
                store.ensure_structure(recover_transactions=not session.lock.reentrant)
            return store
        except Exception as exc:
            session.close()
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
            storage_backend=HYBRID_STORAGE_BACKEND,
            catalog_path="catalog.sqlite",
            line_container_pattern="data/lines/{line_id}.h5",
            legacy_layout=False,
        )
        if project_no.strip():
            manifest.project_no = project_no.strip()
        if device_model.strip():
            manifest.device_model = device_model.strip()
        if coordinate_system.strip():
            manifest.coordinate_system = coordinate_system.strip()
        if vertical_datum.strip():
            manifest.vertical_datum = vertical_datum.strip()
        session = ProjectRepository.open_session(root_path, mode=ProjectAccessMode.WRITE, recover_stale=False)
        store = cls(root_path, manifest, session=session)
        store.ensure_structure()
        store.save_manifest()
        store.append_log(f"创建项目：{manifest.name}")
        return store

    def ensure_structure(self, *, recover_transactions: bool = True) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in self.DIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.storage.ensure_structure(recover_transactions=recover_transactions)
        ensure_project_root_marker(self.root, self.manifest.project_id)

    def save_manifest(self) -> None:
        try:
            self.assert_writable()
            self.manifest.updated_at = local_now()
            self.manifest.revision = int(getattr(self.manifest, "revision", 0)) + 1
            with self.session.transaction("save-project-manifest") as transaction:
                target = transaction.track(self.root / self.MANIFEST_NAME)
                atomic_write_json(target, asdict(self.manifest))
            if getattr(self.storage, "is_hybrid", False):
                self.storage.catalog.set_meta("project_name", self.manifest.name)
                self.storage.catalog.set_meta("manifest_revision", str(self.manifest.revision))
        except Exception as exc:
            error_info = error_info_from_exception(exc, category="io")
            logger.error("Failed to save project manifest: %s [%s]", error_info.user_message, error_info.error_code)
            raise

__all__ = ["FIELD_PROJECT_SCHEMA", "FieldLineRecord", "FieldProjectManifest", "FieldProjectStore", "FieldProjectStoreProtocol", "TARGET_FIELDS", "local_now"]
