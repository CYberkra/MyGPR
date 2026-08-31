#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable recovery store for field-workbench processing drafts.

A processing draft is intentionally separate from a formal processing result.
It records the operator's editable chain and selected step so a project can be
closed or the application can terminate unexpectedly without losing work.
Formal results remain immutable artifacts in ``processed/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.field_project_models import local_now, validate_line_id
from core.storage_primitives import atomic_write_json

PROCESSING_DRAFT_SCHEMA = "mygpr.processing_draft.v1"


class FieldProcessingDraftStoreMixin:
    """JSON persistence helpers mixed into :class:`FieldProjectStore`."""

    def processing_drafts_dir(self) -> Path:
        return self.root / "metadata" / "processing_drafts"

    def processing_draft_path(self, line_id: str) -> Path:
        safe = validate_line_id(line_id)
        return self.processing_drafts_dir() / f"{safe}_processing_draft.json"

    def save_processing_draft(self, line_id: str, payload: dict[str, Any]) -> Path:
        self.assert_writable()
        safe = validate_line_id(line_id)
        target = self.processing_draft_path(safe)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = dict(payload or {})
        document.update(
            {
                "schema": PROCESSING_DRAFT_SCHEMA,
                "line_id": safe,
                "saved_at": local_now(),
            }
        )
        atomic_write_json(target, document)
        return target

    def load_processing_draft(self, line_id: str) -> dict[str, Any] | None:
        target = self.processing_draft_path(line_id)
        if not target.exists():
            return None
        payload = json.loads(target.read_text(encoding="utf-8"))
        if str(payload.get("schema") or "") != PROCESSING_DRAFT_SCHEMA:
            raise ValueError(f"不支持的处理草稿格式：{payload.get('schema')!r}")
        if str(payload.get("line_id") or "") != validate_line_id(line_id):
            raise ValueError("处理草稿测线标识与目标测线不一致")
        return payload

    def clear_processing_draft(self, line_id: str) -> bool:
        self.assert_writable()
        target = self.processing_draft_path(line_id)
        existed = target.exists()
        target.unlink(missing_ok=True)
        return existed

    def list_processing_drafts(self) -> list[Path]:
        root = self.processing_drafts_dir()
        if not root.exists():
            return []
        return sorted(root.glob("*_processing_draft.json"))


__all__ = ["FieldProcessingDraftStoreMixin", "PROCESSING_DRAFT_SCHEMA"]
