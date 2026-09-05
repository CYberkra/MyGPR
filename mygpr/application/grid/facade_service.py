#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid application service: grouping + gridding + GIS layer export."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from mygpr.application.grid.service import (
    DEFAULT_GROUP_TOLERANCE_M,
    gridded_interface_depth,
    group_project_tracks,
    interface_depth_preview,
    persist_grouping,
    write_grid_geojson,
)
from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.project.service import ProjectService

_GRID_LAYER_NAME_MAX = 60


class GridService:
    """测线分组与属性网格化入口（供 backend facade 调用）。"""

    def __init__(self, projects: ProjectService) -> None:
        self._projects = projects

    def group_lines(
        self, project_id: str, *, tolerance_m: float = DEFAULT_GROUP_TOLERANCE_M,
    ) -> dict[str, Any]:
        """轨迹聚类分组并持久化，返回分组 payload。"""
        grouping = group_project_tracks(self._projects, project_id, tolerance_m=tolerance_m)
        crs_name = self._resolve_crs(project_id)
        payload = persist_grouping(
            self._projects, project_id, grouping, crs_name=crs_name)
        return payload

    def export_interface_depth_grid(
        self, project_id: str, line_ids: list[str], *, cell_size_m: float = 1.0,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """界面深度 → 规则网格 → GeoJSON → GIS 图层入库。返回图层摘要。"""
        if context is not None:
            context.raise_if_cancelled()
        grid = gridded_interface_depth(
            self._projects, project_id, line_ids, cell_size_m=cell_size_m)
        if context is not None:
            context.raise_if_cancelled()
        crs_name = self._resolve_crs(project_id)
        staging_dir = self._staging_dir(project_id)
        staging_dir.mkdir(parents=True, exist_ok=True)
        attribute = grid.attribute_name
        geojson_path = staging_dir / f"grid_{attribute}_{grid.ncols}x{grid.nrows}.geojson"
        write_grid_geojson(grid, geojson_path, crs_name=crs_name)
        try:
            record = self._projects.import_grid_layer(
                project_id, geojson_path,
                name=self._layer_name(attribute, line_ids), role="analysis")
        finally:
            geojson_path.unlink(missing_ok=True)
        return {
            "layer": record,
            "grid": {
                "attribute": attribute,
                "cell_size_m": grid.cell_size_m,
                "ncols": grid.ncols,
                "nrows": grid.nrows,
                "valid_count": grid.valid_count,
                "x_origin_m": grid.x_origin_m,
                "y_origin_m": grid.y_origin_m,
            },
            "crs": crs_name,
        }

    def interface_depth_preview(
        self, project_id: str, line_ids: Sequence[str], *,
        cell_size_m: float = 1.0,
    ) -> dict[str, Any]:
        """界面深度网格预览 payload（同步、不落盘；落盘走 export）。"""
        return interface_depth_preview(
            self._projects, project_id, line_ids, cell_size_m=cell_size_m)

    def _resolve_crs(self, project_id: str) -> str:
        metadata = self._projects.get_metadata(project_id)
        return str(metadata.coordinate_system or "")

    def _staging_dir(self, project_id: str) -> Path:
        root = Path(self._projects.get_summary(project_id).root_path)
        return root / "grid" / "staging"

    @staticmethod
    def _layer_name(attribute: str, line_ids: list[str]) -> str:
        base = f"网格-{attribute}-{len(line_ids)}线"
        return base[:_GRID_LAYER_NAME_MAX]


__all__ = [
    "DEFAULT_GROUP_TOLERANCE_M",
    "GridService",
]
