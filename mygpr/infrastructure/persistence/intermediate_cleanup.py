#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intermediate-artifact cleanup persistence (task F candidate 4 / B7)."""
from __future__ import annotations

from typing import Any

import numpy as np  # noqa: F401 — kept for symmetric mixin typing

from core.processing_artifact_index import index_processing_artifacts
from core.field_project_models import validate_line_id


class IntermediateCleanupMixin:
    """Delete disposable intermediates while retaining explicit saved dependencies."""

    def clear_intermediate_artifacts(self, line_id: str | None = None) -> list[dict[str, Any]]:
        from core.hdf5_line_container import delete_processing_artifact

        safe = validate_line_id(line_id) if line_id else None
        cleaned: list[dict[str, Any]] = []
        with self._lock:
            records = index_processing_artifacts(self._store.root, safe)
            required = {
                str(record.params.get("step_input_artifact_id") or "")
                if record.params.get("lineage_version") == 2 else record.parent_artifact_id
                for record in records
            }
            required.update(str(branch.get('head_artifact_id') or '') for branch in
                            self._store.storage.catalog.list_branches(line_id=safe))
            for record in records:
                if record.artifact_kind != "intermediate":
                    continue
                # Saved direct inputs, legacy run inputs and branch heads remain
                # readable. Cleanup must not silently reparent their provenance.
                # Explicit deletion uses the dependency closure and project trash.
                if record.artifact_id in required:
                    continue
                container = self._store.storage.line_container_path(record.line_id)
                removed = delete_processing_artifact(container, record.artifact_id)
                self._store.storage.catalog.delete_intermediate_artifact(record.artifact_id)
                line_dir = self._store.root / "processed" / record.line_id
                for suffix in (".npy", ".artifact"):
                    (line_dir / f"{record.artifact_id}{suffix}").unlink(missing_ok=True)
                for prefix in (f"{record.line_id}_params", f"{record.line_id}_processing_manifest"):
                    for candidate in line_dir.glob(f"{prefix}_*{record.artifact_id[-4:]}.json"):
                        candidate.unlink(missing_ok=True)
                cleaned.append({
                    "artifact_id": record.artifact_id,
                    "line_id": record.line_id,
                    "name": record.method_name or record.method_id,
                    "run_group_id": record.run_group_id,
                    "output_sha256": record.output_data_sha256,
                    "h5_removed": removed,
                })
        if cleaned:
            import json as _json
            import uuid as _uuid

            log_dir = self._store.root / "metadata" / "intermediates"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"cleanup_log_{_uuid.uuid4().hex[:8]}.json"
            log_path.write_text(
                _json.dumps(
                    {"schema": "mygpr.intermediate_cleanup.v1", "cleaned": cleaned},
                    ensure_ascii=False, indent=2,
                ) + chr(10),
                encoding="utf-8",
            )
        return cleaned
