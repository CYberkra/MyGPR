"""Backend job entry points for versioned amplitude slices."""
from __future__ import annotations

from typing import Any, Sequence

from mygpr.application.spatial.amplitude import AmplitudeSource, LineObservations, build_amplitude_grid
from mygpr.application.spatial.amplitude_projection import validate_metric_crs
from mygpr.application.spatial.amplitude_saved import load_saved_slice


class BackendAmplitudeMixin:
    __slots__ = ()

    def submit_amplitude_slice(
        self, project_id: str, sources: Sequence[AmplitudeSource],
        positions: Sequence[LineObservations], start_ns: float, end_ns: float, *,
        cell_size_m: float, attribute: str = 'rms', title: str | None = None,
    ) -> str:
        """Run an observation-only time-window amplitude grid as a leased job."""
        selected, projected = tuple(sources), tuple(positions)
        crs = {item.coordinate_system for item in projected}
        if len(crs) != 1:
            raise ValueError('所有测线必须使用同一坐标系')
        validate_metric_crs(next(iter(crs)))
        return self._submit_project_job(
            project_id, title or '振幅切片',
            lambda context: build_amplitude_grid(
                self.projects, project_id, selected, projected, start_ns, end_ns,
                cell_size_m=cell_size_m, attribute=attribute, context=context, include_curtains=True),
        )

    def submit_save_amplitude_slice(
        self, project_id: str, result: Any, *, name: str = '振幅切片', title: str | None = None,
    ) -> str:
        """Save a previously computed slice as a versioned spatial result."""
        return self._submit_project_job(
            project_id, title or '保存振幅切片',
            lambda context: self.projects.save_amplitude_result(project_id, result, name=name, context=context),
        )

    def submit_open_amplitude_slice(self, project_id: str, result_id: str, *, title: str | None = None) -> str:
        """Load the saved grid and, when sources still match, its 3-D previews."""
        return self._submit_project_job(
            project_id, title or '打开振幅切片',
            lambda context: load_saved_slice(self.projects, project_id, result_id, context=context),
        )
