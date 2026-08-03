#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit protocol interface for :class:`FieldProjectStore`.

Defining this protocol makes the 7-mixin coordinator's public API surface
type-checkable and documentable, rather than relying on implicit mixin
composition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.field_project_models import FieldProjectManifest, FieldLineRecord
from core.project_storage_backend import ProjectStorageBackend


class FieldProjectStoreProtocol(Protocol):
    """Explicit interface for the 7-mixin ``FieldProjectStore`` coordinator."""

    # ------------------------------------------------------------------
    # Lifecycle / session
    # ------------------------------------------------------------------
    def __enter__(self) -> FieldProjectStoreProtocol: ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
    def close(self) -> None: ...

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        access_mode: str | Any = ...,
        recover_stale_lock: bool = ...,
    ) -> FieldProjectStoreProtocol: ...

    @classmethod
    def create_empty(
        cls,
        root: str | Path,
        *,
        name: str = ...,
        location: str = ...,
        operator: str = ...,
        project_no: str = ...,
        device_model: str = ...,
        coordinate_system: str = ...,
        vertical_datum: str = ...,
    ) -> FieldProjectStoreProtocol: ...

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------
    root: Path
    manifest: FieldProjectManifest
    read_only: bool
    storage: ProjectStorageBackend

    # ------------------------------------------------------------------
    # Manifest / structure
    # ------------------------------------------------------------------
    def ensure_structure(self, *, recover_transactions: bool = ...) -> None: ...
    def save_manifest(self) -> None: ...
    def assert_writable(self) -> None: ...

    # ------------------------------------------------------------------
    # Line store (FieldLineStoreMixin)
    # ------------------------------------------------------------------
    def list_lines(self) -> list[FieldLineRecord]: ...
    def get_line(self, line_id: str) -> FieldLineRecord: ...
    def upsert_line(self, line: FieldLineRecord) -> None: ...
    def delete_line(self, line_id: str, *, reason: str = ...) -> None: ...
    def import_line_file(
        self,
        source: str,
        line_id: str,
        *,
        name: str = ...,
        dielectric_constant: float = ...,
        channel: str = ...,
    ) -> FieldLineRecord: ...
    def load_gpr_dataset(self, line_id: str) -> Any: ...
    def save_gpr_dataset(self, line_id: str, dataset: Any) -> Path: ...
    def read_window(
        self,
        line_id: str,
        *,
        max_samples: int = ...,
        max_traces: int = ...,
    ) -> tuple[Any, list[int], list[int]]: ...
    def read_artifact_window(
        self,
        line_id: str,
        artifact_id: str,
        *,
        max_samples: int = ...,
        max_traces: int = ...,
    ) -> tuple[Any, list[int], list[int]]: ...
    def read_dataset(self, line_id: str) -> Any: ...
    def save_trajectory(self, line_id: str, trajectory: Any) -> Path: ...
    def load_trajectory(self, line_id: str) -> Any: ...
    def run_line_quality_check(self, line_id: str) -> Any: ...

    # ------------------------------------------------------------------
    # Processing draft store (FieldProcessingDraftStoreMixin)
    # ------------------------------------------------------------------
    def save_processing_draft(self, line_id: str, payload: dict[str, Any]) -> Path: ...
    def load_processing_draft(self, line_id: str) -> dict[str, Any] | None: ...
    def clear_processing_draft(self, line_id: str) -> bool: ...
    def list_processing_drafts(self) -> list[Path]: ...

    # ------------------------------------------------------------------
    # Interface store (FieldInterfaceStoreMixin)
    # ------------------------------------------------------------------
    def load_basal_interface_annotation(
        self, line_id: str, *, create: bool = ...
    ) -> Any | None: ...
    def load_basal_interface_labels(self, line_id: str) -> dict[str, Any]: ...
    def basal_interface_summary(self, line_id: str) -> dict[str, Any]: ...

    # ------------------------------------------------------------------
    # Target store (FieldTargetStoreMixin)
    # ------------------------------------------------------------------
    def save_targets(self, line_id: str, targets: list[dict[str, Any]]) -> Path: ...
    def load_targets(self, line_id: str) -> list[dict[str, Any]]: ...

    # ------------------------------------------------------------------
    # Spatial store (FieldSpatialStoreMixin)
    # ------------------------------------------------------------------
    def export_spatial_targets_xy(self, line_id: str) -> Path: ...
    def export_project_spatial_coordinates(
        self,
        *,
        filename: str | None = ...,
        cancel_requested: Any = ...,
        progress_callback: Any = ...,
    ) -> Path: ...

    # ------------------------------------------------------------------
    # Artifact store (FieldArtifactStoreMixin)
    # ------------------------------------------------------------------
    def save_processed_line(
        self,
        line_id: str,
        data: Any,
        params: dict[str, Any],
        *,
        cancel_requested: Any = ...,
        progress_callback: Any = ...,
    ) -> tuple[Path, Path]: ...

    # ------------------------------------------------------------------
    # Runtime store (FieldProjectRuntimeStoreMixin)
    # ------------------------------------------------------------------
    def append_log(self, message: str) -> None: ...
    def total_raw_size_mb(self) -> float: ...
    def storage_usage_mb(self) -> float: ...
    def list_processing_branches(self, line_id: str) -> list[dict[str, Any]]: ...
    def list_project_exports(self, *, export_kind: str | None = ...) -> list[dict[str, Any]]: ...
    def run_project_quality_check(
        self,
        *,
        cancel_requested: Any = ...,
        progress_callback: Any = ...,
    ) -> Any: ...


__all__ = ["FieldProjectStoreProtocol"]
