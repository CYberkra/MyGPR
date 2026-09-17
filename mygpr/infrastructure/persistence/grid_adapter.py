#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Line-group and grid-GIS-layer persistence mixin.

core 依赖仅两处：坐标投影解析（resolve_projection_spec）与 GIS 入库
（GISLayerStore.import_layer）；分组/网格纯计算已在 domain/application
完成，这里只做投影坐标解析与落盘。
"""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.coordinate_projection import ProjectionError, resolve_projection_spec
from core.gis_layers import GISLayerStore

_LINE_GROUPS_PATH = "grid/line_groups.json"


class GridPersistenceMixin:
    """Project-session line-group + grid-layer operations."""

    def load_line_groups(self) -> Mapping[str, Any] | None:
        path = self._store.root / _LINE_GROUPS_PATH
        with self._lock:
            if not path.exists():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def save_line_groups(self, payload: Mapping[str, Any]) -> None:
        data = dict(payload)
        self._store.write_json(self._store.root / _LINE_GROUPS_PATH, data)
        self._store.append_log(
            f"保存测线分组: groups={len(data.get('groups', []))}, "
            f"tolerance={data.get('tolerance_m', '?')}m")

    def list_gis_layers(self) -> Sequence[Mapping[str, Any]]:
        store = GISLayerStore(self._store.root)
        return [asdict(record) for record in store.list_layers()]

    def import_grid_layer(
        self, geojson_path: Path, *, name: str, role: str
    ) -> Mapping[str, Any]:
        store = GISLayerStore(self._store.root)
        crs_name = self._grid_crs_name()
        record = store.import_layer(
            geojson_path, name=name, role=role, project_crs=crs_name)
        self._store.append_log(f"导入网格 GIS 图层: {name} ({record.layer_id})")
        return asdict(record)

    def _grid_crs_name(self) -> str:
        coordinate_system = str(self._store.manifest.coordinate_system or "")
        try:
            spec = resolve_projection_spec(coordinate_system, mean_longitude=None)
        except ProjectionError:
            return ""
        if spec.epsg:
            return f"EPSG:{spec.epsg}"
        return coordinate_system


__all__ = ["GridPersistenceMixin"]
