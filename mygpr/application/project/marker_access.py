"""Marker session access kept separate from the general project facade."""
from typing import Any
from mygpr.domain.markers.models import MarkerDocument


class ProjectMarkerAccessMixin:
    _session: Any

    def marker_source_id(self, project_id: str, line_id: str) -> str:
        return self._session(project_id).marker_source_id(line_id)

    def select_marker_ranges(self, view, preview, *, context):
        return self._session(view.document.project_id).select_marker_ranges(view, preview, context=context)

    def export_markers(self, views, path, format_name):
        if not views:
            raise ValueError('没有可导出的测线')
        return self._session(views[0].document.project_id).export_markers(views, path, format_name)

    def read_markers(self, project_id: str, line_id: str) -> MarkerDocument:
        return self._session(project_id).read_markers(line_id)

    def save_markers(self, project_id: str, document: MarkerDocument, *, expected_revision: int) -> MarkerDocument:
        if document.project_id != project_id:
            raise ValueError('标记不属于当前项目')
        return self._session(project_id).save_markers(document, expected_revision=expected_revision)
