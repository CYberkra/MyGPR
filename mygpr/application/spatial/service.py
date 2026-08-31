"""Project-scoped spatial use cases."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from mygpr.application.project.service import ProjectService
from mygpr.domain.spatial.models import SpatialResult, SpatialTrack


class SpatialService:
    def __init__(self, projects: ProjectService) -> None:
        self._projects = projects

    def load_tracks(self, project_id: str) -> tuple[SpatialTrack, ...]:
        return self._projects.load_spatial_tracks(project_id)

    def list_results(self, project_id: str) -> tuple[SpatialResult, ...]:
        return self._projects.list_spatial_results(project_id)

    def preflight(self, project_id: str, *, line_ids: Sequence[str] | None = None, generate_surface: bool = True) -> Mapping[str, Any]:
        return self._projects.spatial_preflight(project_id, line_ids=line_ids, generate_surface=generate_surface)

    def create_result(self, project_id: str, *, name: str, line_ids: Sequence[str] | None = None, velocity_m_per_ns: float | None = None, generate_surface: bool = True) -> SpatialResult:
        return self._projects.create_spatial_result(project_id, name=name, line_ids=line_ids, velocity_m_per_ns=velocity_m_per_ns, generate_surface=generate_surface)

    def set_current(self, project_id: str, result_id: str) -> None:
        self._projects.set_current_spatial_result(project_id, result_id)

    def build_georeference_3d(
        self,
        project_id: str,
        line_id: str,
        *,
        preview_lod: str = "auto",
        max_preview_traces: int = 240,
        max_preview_samples: int = 160,
    ) -> Mapping[str, Any]:
        return self._projects.build_georeference_3d(
            project_id,
            line_id,
            preview_lod=preview_lod,
            max_preview_traces=max_preview_traces,
            max_preview_samples=max_preview_samples,
        )
