"""Project-owned marker records; source acquisitions are never written."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from typing import Any
from collections.abc import Mapping

from core.field_project_models import validate_line_id
from core.security_paths import resolve_managed_path
from mygpr.domain.markers.models import MarkerDocument
from mygpr.domain.markers.operations import import_markers, validate_transition


def metadata_value(metadata, key, default=None):
    current = metadata
    while isinstance(current, Mapping):
        if key in current:
            return current[key]
        current = current.get('metadata')
    return default


class MarkerPersistenceMixin:
    _lock: Any
    _store: Any
    summary: Any
    get_dataset_info: Any

    def export_markers(self, views, path, format_name):
        from mygpr.infrastructure.reporting.marker_export import export_views
        return export_views(views, path, format_name)

    def select_marker_ranges(self, view, preview, *, context):
        from mygpr.infrastructure.persistence.marker_selection import commit_selection
        with self._lock:
            self._store.assert_writable()
            current = self.read_markers(view.document.line_id)
            if current.source_id != view.document.source_id or current.revision != view.document.revision:
                raise ValueError('源数据或标记已变化，请重新预览')
            return commit_selection(self, view, preview, context)

    def marker_source_id(self, line_id: str) -> str:
        info = self.get_dataset_info(line_id)
        generation = metadata_value(info.metadata, 'marker_source_version')
        if generation:
            return f'raw:{generation}'
        # Legacy files remain read-only until the first explicit project edit.
        path = resolve_managed_path(self._store.root, f'raw/{validate_line_id(line_id)}/import_manifest.json')
        imported_at = ''
        if path.is_file():
            imported_at = json.loads(path.read_text(encoding='utf-8')).get('imported_at', '')
        identity = (self.summary.project_id, line_id, info.source_path, imported_at,
                    metadata_value(info.metadata, 'data_sha256', ''), info.shape)
        return 'raw:' + str(uuid5(NAMESPACE_URL, repr(identity)))

    def _marker_path(self, line_id: str) -> Path:
        return resolve_managed_path(self._store.root, f'metadata/markers/{validate_line_id(line_id)}.json')

    def read_markers(self, line_id: str) -> MarkerDocument:
        with self._lock:
            safe = validate_line_id(line_id)
            info = self.get_dataset_info(safe)
            source_id = self.marker_source_id(safe)
            imported = import_markers(self.summary.project_id, safe, source_id,
                                      metadata_value(info.metadata, 'acquisition_markers', ()))
            if metadata_value(info.metadata, 'marker_source_orientation_valid') is False:
                imported = replace(imported, markers=tuple(replace(item, anchor=None) for item in imported.markers))
            path = self._marker_path(safe)
            if not path.exists():
                return imported
            document = MarkerDocument.from_dict(json.loads(path.read_text(encoding='utf-8')))
            if document.project_id != imported.project_id or document.line_id != safe:
                raise ValueError('标记文档与项目或测线不匹配')
            ids = {item.uid for item in document.markers}
            additions = tuple(item for item in imported.markers if item.uid not in ids)
            return replace(document, source_id=source_id, markers=document.markers + additions)

    def save_markers(self, document: MarkerDocument, *, expected_revision: int) -> MarkerDocument:
        with self._lock:
            self._store.assert_writable()
            current = self.read_markers(document.line_id)
            if (document.project_id != current.project_id or document.source_id != current.source_id):
                raise ValueError('标记来源已变化，请重新加载当前测线')
            if current.revision != expected_revision or document.revision != expected_revision:
                raise ValueError('标记已被修改，请重新加载后再保存')
            validate_transition(current, document)
            previous = {item.uid: item for item in current.markers}
            proposed = {item.uid: item for item in document.markers}
            for uid, original in previous.items():
                if original.origin != 'acquisition':
                    continue
                edited = proposed.get(uid)
                if (edited is None or edited.origin != original.origin
                        or edited.original != original.original or edited.anchor != original.anchor
                        or edited.created_at != original.created_at):
                    raise ValueError('采集原记录不可修改或删除')
            saved = replace(document, revision=current.revision + 1)
            self._store.write_json(self._marker_path(document.line_id), saved.to_dict())
            return saved
