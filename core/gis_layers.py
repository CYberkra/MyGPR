#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline GIS layer registry and georeferenced readers for field projects."""
from __future__ import annotations

import csv
import json
import shutil
import hashlib
import uuid
from defusedxml import ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from core.field_project_models import atomic_write_json, local_now
from core.crs_text import canonical_crs_text
from core.schema_registry import DEFAULT_SCHEMA_REGISTRY
from core.security_paths import ensure_direct_child, resolve_managed_path

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
except Exception:  # pragma: no cover
    rasterio = None
    Resampling = None
    WarpedVRT = None

try:
    import fiona
except Exception:  # pragma: no cover
    fiona = None

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None

GIS_LAYER_SCHEMA = "mygpr.gis_layers.v2"
MAX_KML_BYTES = 256 * 1024 * 1024


@dataclass
class GISLayerRecord:
    layer_id: str
    name: str
    kind: str
    source_path: str
    crs: str = ""
    bounds: list[float] = field(default_factory=list)
    geometry_type: str = ""
    role: str = "reference"
    visible: bool = True
    opacity: float = 1.0
    imported_at: str = field(default_factory=local_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_sha256: str = ""
    style_version: int = 1
    z_order: int = 0
    vertical_crs: str = ""
    lineage: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GISLayerRecord":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: payload[key] for key in allowed if key in payload})


@dataclass
class GISVectorFeature:
    geometry_type: str
    coordinates: list[np.ndarray]
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GISRasterPreview:
    array: np.ndarray
    extent: tuple[float, float, float, float]
    crs: str
    nodata: float | None = None
    is_dem: bool = False


def _canonical_crs(value: Any) -> str:
    return canonical_crs_text(value)



def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_crs(value: Any, *, required: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError("工程坐标系必须使用 EPSG 编号或完整 WKT。")
        return ""
    if CRS is None:
        return text
    try:
        crs = CRS.from_user_input(text)
    except Exception as exc:
        raise ValueError(f"无法解析坐标系：{text}") from exc
    return crs.to_wkt()

def _flatten_coordinate_pairs(value: Any) -> Iterable[tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
        yield float(value[0]), float(value[1])
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_coordinate_pairs(child)


def _bounds_from_pairs(pairs: Iterable[tuple[float, float]]) -> list[float]:
    values = list(pairs)
    if not values:
        return []
    x: np.ndarray = np.asarray([p[0] for p in values], dtype=float)
    y: np.ndarray = np.asarray([p[1] for p in values], dtype=float)
    return [float(np.nanmin(x)), float(np.nanmin(y)), float(np.nanmax(x)), float(np.nanmax(y))]


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        from core.job_manager import JobCancelled
        raise JobCancelled("GIS 图层导入已取消")


def _copy_file_cancelable(
    src: Path,
    dst: Path,
    *,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    completed_before: int = 0,
    total_bytes: int | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> int:
    total = int(total_bytes if total_bytes is not None else src.stat().st_size)
    copied = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as source, dst.open("wb") as target:
        while True:
            _raise_if_cancelled(cancel_requested)
            block = source.read(max(int(chunk_size), 1024 * 1024))
            if not block:
                break
            target.write(block)
            copied += len(block)
            if progress_callback is not None:
                progress_callback(
                    completed_before + copied,
                    max(total, 1),
                    f"复制 GIS 文件：{src.name}",
                )
        target.flush()
    shutil.copystat(src, dst)
    return copied


def _copy_shapefile_family(
    src: Path,
    destination_dir: Path,
    *,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    copied_main = destination_dir / src.name
    candidates = [
        candidate
        for candidate in src.parent.glob(f"{stem}.*")
        if candidate.suffix.lower() in {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".fbn", ".fbx", ".ain", ".aih", ".ixs", ".mxs", ".atx", ".xml"}
    ]
    total = sum(candidate.stat().st_size for candidate in candidates)
    completed = 0
    for candidate in candidates:
        completed += _copy_file_cancelable(
            candidate,
            destination_dir / candidate.name,
            cancel_requested=cancel_requested,
            progress_callback=progress_callback,
            completed_before=completed,
            total_bytes=total,
        )
    return copied_main


class GISLayerStore:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.spatial_root = self.project_root / "spatial"
        self.layers_root = self.spatial_root / "gis_layers"
        self.registry_path = self.spatial_root / "gis_layers.json"
        self.layers_root.mkdir(parents=True, exist_ok=True)

    def list_layers(self) -> list[GISLayerRecord]:
        if not self.registry_path.exists():
            return []
        loaded = DEFAULT_SCHEMA_REGISTRY.load_path(
            self.registry_path,
            family="mygpr.gis_layers",
            quarantine_root=self.spatial_root / "quarantine",
        )
        if loaded.read_only:
            raise PermissionError("GIS 图层清单来自更高版本，只能只读打开。")
        return [GISLayerRecord.from_dict(item) for item in loaded.payload.get("layers", [])]

    def save_layers(self, layers: list[GISLayerRecord]) -> None:
        atomic_write_json(
            self.registry_path,
            {"schema": GIS_LAYER_SCHEMA, "updated_at": local_now(), "layers": [asdict(layer) for layer in layers]},
        )

    def get(self, layer_id: str) -> GISLayerRecord:
        for layer in self.list_layers():
            if layer.layer_id == layer_id:
                return layer
        raise KeyError(layer_id)

    def _resolve_layer_source(self, layer: GISLayerRecord, *, require_file: bool = False) -> Path:
        """Resolve a registry path only inside its own managed layer directory."""
        layer_id = str(layer.layer_id or "")
        if not layer_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in layer_id):
            raise ValueError(f"GIS 图层编号无效：{layer_id!r}")
        path = resolve_managed_path(
            self.project_root,
            layer.source_path,
            require_file=require_file,
        )
        layer_dir = ensure_direct_child(self.layers_root, path.parent)
        if layer_dir.name != layer_id:
            raise ValueError(f"GIS 图层来源目录与图层编号不一致：{layer.source_path}")
        return path

    def update(self, layer_id: str, **changes: Any) -> GISLayerRecord:
        layers = self.list_layers()
        for layer in layers:
            if layer.layer_id == layer_id:
                for key, value in changes.items():
                    if hasattr(layer, key):
                        setattr(layer, key, value)
                self.save_layers(layers)
                from core.gis_cache import GISCacheManager
                GISCacheManager(self.project_root).invalidate(source_sha256=layer.source_sha256 or None)
                return layer
        raise KeyError(layer_id)

    def remove(self, layer_id: str) -> None:
        layers = self.list_layers()
        target = next((layer for layer in layers if layer.layer_id == layer_id), None)
        if target is None:
            return
        path = self._resolve_layer_source(target)
        if path.exists():
            if path.suffix.lower() == ".shp":
                for candidate in path.parent.glob(f"{path.stem}.*"):
                    candidate.unlink(missing_ok=True)
            else:
                path.unlink(missing_ok=True)
        self.save_layers([layer for layer in layers if layer.layer_id != layer_id])
        from core.gis_cache import GISCacheManager
        GISCacheManager(self.project_root).invalidate(source_sha256=target.source_sha256 or None)

    def import_layer(
        self,
        source: str | Path,
        *,
        name: str | None = None,
        role: str = "reference",
        project_crs: str = "",
        cancel_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> GISLayerRecord:
        src = Path(source).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        suffix = src.suffix.lower()
        layer_id = uuid.uuid4().hex[:12]
        destination_dir = self.layers_root / layer_id
        destination_dir.mkdir(parents=True, exist_ok=False)
        try:
            _raise_if_cancelled(cancel_requested)
            if suffix == ".shp":
                copied = _copy_shapefile_family(
                    src,
                    destination_dir,
                    cancel_requested=cancel_requested,
                    progress_callback=progress_callback,
                )
            else:
                copied = destination_dir / src.name
                _copy_file_cancelable(
                    src,
                    copied,
                    cancel_requested=cancel_requested,
                    progress_callback=progress_callback,
                    total_bytes=src.stat().st_size,
                )
            _raise_if_cancelled(cancel_requested)
            if progress_callback is not None:
                progress_callback(src.stat().st_size, max(src.stat().st_size, 1), "读取图层坐标系与空间范围")
            kind, crs, bounds, geometry_type, metadata = self._inspect(copied, project_crs=project_crs)
            _raise_if_cancelled(cancel_requested)
            record = GISLayerRecord(
                layer_id=layer_id,
                name=name or src.stem,
                kind=kind,
                source_path=copied.relative_to(self.project_root).as_posix(),
                crs=crs,
                bounds=bounds,
                geometry_type=geometry_type,
                role=role,
                opacity=0.75 if kind == "raster" else 1.0,
                metadata=metadata,
                source_sha256=_sha256_file(copied),
                style_version=1,
                z_order=len(self.list_layers()),
                vertical_crs=str(metadata.get("vertical_crs") or ""),
                lineage={"source_name": src.name, "imported_from": str(src)},
            )
            layers = self.list_layers()
            layers.append(record)
            self.save_layers(layers)
            return record
        except Exception:
            shutil.rmtree(destination_dir, ignore_errors=True)
            raise

    def _inspect(self, path: Path, *, project_crs: str = "") -> tuple[str, str, list[float], str, dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in {".tif", ".tiff", ".mbtiles"}:
            if rasterio is None:
                raise RuntimeError("读取 GeoTIFF/MBTiles 需要 rasterio。")
            with rasterio.open(path) as ds:
                crs = _canonical_crs(ds.crs)
                bounds = [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)]
                metadata = {
                    "width": ds.width,
                    "height": ds.height,
                    "band_count": ds.count,
                    "dtype": str(ds.dtypes[0]),
                    "nodata": ds.nodata,
                    "is_dem": bool(ds.count == 1 and suffix != ".mbtiles"),
                    "container": "MBTiles" if suffix == ".mbtiles" else "GeoTIFF",
                }
            return "raster", crs, bounds, "Raster", metadata
        if suffix in {".geojson", ".json"}:
            payload = json.loads(path.read_text(encoding="utf-8"))
            features = payload.get("features", []) if payload.get("type") == "FeatureCollection" else [payload]
            pairs: list[tuple[float, float]] = []
            types: set[str] = set()
            for feature in features:
                geometry = feature.get("geometry", feature)
                types.add(str(geometry.get("type", "Unknown")))
                pairs.extend(_flatten_coordinate_pairs(geometry.get("coordinates", [])))
            crs = "EPSG:4326"
            if isinstance(payload.get("crs"), dict):
                crs = payload["crs"].get("properties", {}).get("name", crs)
            return "vector", _canonical_crs(crs), _bounds_from_pairs(pairs), "/".join(sorted(types)), {"feature_count": len(features)}
        if suffix == ".kml":
            features = self._read_kml(path)
            pairs = [(float(x), float(y)) for feature in features for coords in feature.coordinates for x, y in coords]
            return "vector", "EPSG:4326", _bounds_from_pairs(pairs), "KML", {"feature_count": len(features)}
        if suffix == ".csv":
            features, crs = self._read_csv_features(path, project_crs=project_crs)
            pairs = [(float(x), float(y)) for feature in features for coords in feature.coordinates for x, y in coords]
            return "vector", crs, _bounds_from_pairs(pairs), "Point", {"feature_count": len(features)}
        if suffix in {".shp", ".gpkg"}:
            if fiona is None:
                raise RuntimeError("读取 Shapefile/GeoPackage 需要 fiona。")
            with fiona.open(path) as collection:
                crs = _canonical_crs(collection.crs_wkt or collection.crs)
                bounds = [float(v) for v in collection.bounds]
                geometry_type = str(collection.schema.get("geometry") or "Unknown")
                count = len(collection)
            return "vector", crs, bounds, geometry_type, {"feature_count": count}
        raise ValueError(f"暂不支持的 GIS 文件：{path.suffix}")

    def load_vector(self, layer: GISLayerRecord, *, target_crs: str = "") -> list[GISVectorFeature]:
        path = self._resolve_layer_source(layer, require_file=True)
        suffix = path.suffix.lower()
        if suffix in {".geojson", ".json"}:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_features = payload.get("features", []) if payload.get("type") == "FeatureCollection" else [payload]
            features = []
            for raw in raw_features:
                geometry = raw.get("geometry", raw)
                features.extend(self._geometry_to_features(geometry, raw.get("properties", {})))
        elif suffix == ".kml":
            features = self._read_kml(path)
        elif suffix == ".csv":
            features, _ = self._read_csv_features(path, project_crs=layer.crs)
        elif suffix in {".shp", ".gpkg"}:
            if fiona is None:
                raise RuntimeError("读取 Shapefile/GeoPackage 需要 fiona。")
            features = []
            with fiona.open(path) as collection:
                for item in collection:
                    geometry = item.get("geometry")
                    if geometry:
                        features.extend(self._geometry_to_features(geometry, dict(item.get("properties") or {})))
        else:
            return []
        if target_crs and layer.crs and _canonical_crs(target_crs) != _canonical_crs(layer.crs):
            features = self._transform_features(features, layer.crs, target_crs)
        return features

    def load_raster_preview(
        self,
        layer: GISLayerRecord,
        *,
        max_size: int = 1600,
        target_crs: str = "",
    ) -> GISRasterPreview:
        """Read a display preview and optionally warp it into the project CRS."""
        if rasterio is None:
            raise RuntimeError("读取 GeoTIFF/MBTiles 需要 rasterio。")
        path = self._resolve_layer_source(layer, require_file=True)
        with rasterio.open(path) as source:
            source_crs = _canonical_crs(source.crs)
            wanted = _canonical_crs(target_crs) if target_crs else source_crs
            needs_warp = bool(wanted and source_crs and wanted != source_crs)
            dataset = WarpedVRT(source, crs=wanted, resampling=Resampling.bilinear) if needs_warp else source
            try:
                scale = max(dataset.width / max_size, dataset.height / max_size, 1.0)
                out_width = max(1, int(round(dataset.width / scale)))
                out_height = max(1, int(round(dataset.height / scale)))
                if dataset.count >= 3:
                    array = dataset.read([1, 2, 3], out_shape=(3, out_height, out_width), resampling=Resampling.bilinear)
                    array = np.moveaxis(array, 0, -1)
                    out = np.zeros_like(array, dtype=np.float32)
                    for band in range(3):
                        values = array[..., band].astype(np.float32)
                        valid = np.isfinite(values)
                        if dataset.nodata is not None:
                            valid &= values != dataset.nodata
                        if valid.any():
                            low, high = np.percentile(values[valid], [2, 98])
                            out[..., band] = np.clip((values - low) / max(high - low, 1e-9), 0, 1)
                    array = out
                    is_dem = False
                else:
                    array = dataset.read([1], out_shape=(1, out_height, out_width), resampling=Resampling.bilinear)[0].astype(np.float32)
                    if dataset.nodata is not None:
                        array[array == dataset.nodata] = np.nan
                    is_dem = True
                extent = (float(dataset.bounds.left), float(dataset.bounds.right), float(dataset.bounds.bottom), float(dataset.bounds.top))
                return GISRasterPreview(array=array, extent=extent, crs=_canonical_crs(dataset.crs), nodata=dataset.nodata, is_dem=is_dem)
            finally:
                if dataset is not source:
                    dataset.close()

    def load_raster_preview_cached(
        self, layer: GISLayerRecord, *, max_size: int = 1600, target_crs: str = ""
    ) -> GISRasterPreview:
        from core.gis_cache import GISCacheKey, GISCacheManager
        key = GISCacheKey(
            source_sha256=layer.source_sha256 or _sha256_file(self._resolve_layer_source(layer, require_file=True)),
            target_crs=_canonical_crs(target_crs) if target_crs else layer.crs,
            style_version=int(layer.style_version),
            max_size=int(max_size),
        )
        return GISCacheManager(self.project_root).get_or_create_raster(
            key, lambda: self.load_raster_preview(layer, max_size=max_size, target_crs=target_crs)
        )

    @staticmethod
    def _geometry_to_features(geometry: dict[str, Any], properties: dict[str, Any]) -> list[GISVectorFeature]:
        kind = str(geometry.get("type", ""))
        coords = geometry.get("coordinates", [])
        if kind == "Point":
            return [GISVectorFeature(kind, [np.asarray([coords[:2]], dtype=float)], properties)]
        if kind == "MultiPoint":
            return [GISVectorFeature("Point", [np.asarray([point[:2]], dtype=float)], properties) for point in coords]
        if kind == "LineString":
            return [GISVectorFeature(kind, [np.asarray(coords, dtype=float)[:, :2]], properties)]
        if kind == "MultiLineString":
            return [GISVectorFeature("LineString", [np.asarray(line, dtype=float)[:, :2]], properties) for line in coords]
        if kind == "Polygon":
            return [GISVectorFeature(kind, [np.asarray(ring, dtype=float)[:, :2] for ring in coords], properties)]
        if kind == "MultiPolygon":
            return [GISVectorFeature("Polygon", [np.asarray(ring, dtype=float)[:, :2] for ring in polygon], properties) for polygon in coords]
        return []

    @staticmethod
    def _read_kml(path: Path) -> list[GISVectorFeature]:
        if path.stat().st_size > MAX_KML_BYTES:
            raise ValueError(f"KML 文件过大，拒绝直接解析：{path.stat().st_size} bytes")
        root = ET.parse(path).getroot()
        features: list[GISVectorFeature] = []
        for element in root.iter():
            tag = element.tag.split("}")[-1]
            if tag != "coordinates" or not (element.text or "").strip():
                continue
            coords = []
            for token in (element.text or "").replace("\n", " ").split():
                parts = token.split(",")
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))
            if not coords:
                continue
            geometry = "Point" if len(coords) == 1 else "LineString"
            features.append(GISVectorFeature(geometry, [np.asarray(coords, dtype=float)], {}))
        return features

    @staticmethod
    def _read_csv_features(path: Path, *, project_crs: str = "") -> tuple[list[GISVectorFeature], str]:
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError("CSV GIS 图层缺少表头。")
            names = set(reader.fieldnames)
            x_key = next((k for k in ("x", "X", "easting", "east_m", "local_x_m", "东坐标") if k in names), None)
            y_key = next((k for k in ("y", "Y", "northing", "north_m", "local_y_m", "北坐标") if k in names), None)
            lon_key = next((k for k in ("longitude", "lon", "lng", "经度") if k in names), None)
            lat_key = next((k for k in ("latitude", "lat", "纬度") if k in names), None)
            if x_key and y_key:
                crs = _canonical_crs(project_crs)
                pairs = [(float(row[x_key]), float(row[y_key]), row) for row in reader if row.get(x_key) and row.get(y_key)]
            elif lon_key and lat_key:
                crs = "EPSG:4326"
                pairs = [(float(row[lon_key]), float(row[lat_key]), row) for row in reader if row.get(lon_key) and row.get(lat_key)]
            else:
                raise ValueError("CSV 图层需包含 x/y 或 longitude/latitude 列。")
        features = [GISVectorFeature("Point", [np.asarray([[x, y]], dtype=float)], dict(row)) for x, y, row in pairs]
        return features, crs

    @staticmethod
    def _transform_features(features: list[GISVectorFeature], source_crs: str, target_crs: str) -> list[GISVectorFeature]:
        if Transformer is None:
            raise RuntimeError("坐标转换需要 pyproj。")
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        transformed = []
        for feature in features:
            coords_out = []
            for coords in feature.coordinates:
                x, y = transformer.transform(coords[:, 0], coords[:, 1])
                coords_out.append(np.column_stack([x, y]).astype(float))
            transformed.append(GISVectorFeature(feature.geometry_type, coords_out, dict(feature.properties)))
        return transformed


    def query_vector_features(
        self,
        layer_id: str,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        target_crs: str = "",
        max_features: int = 50000,
    ) -> list[GISVectorFeature]:
        """Read only features intersecting ``bbox`` using Fiona's spatial index path."""
        layer = self.get(layer_id)
        if layer.kind != "vector":
            raise ValueError("Only vector layers support bbox queries")
        if fiona is None:
            raise RuntimeError("Fiona is required for vector queries")
        source = self._resolve_layer_source(layer, require_file=True)
        features: list[GISVectorFeature] = []
        with fiona.open(source) as collection:
            iterator = collection.filter(bbox=bbox) if bbox is not None else iter(collection)
            for index, item in enumerate(iterator):
                if index >= max(int(max_features), 1):
                    break
                geometry = item.get("geometry") or {}
                pairs = list(_flatten_coordinate_pairs(geometry.get("coordinates") or []))
                if not pairs:
                    continue
                coordinates: list[np.ndarray] = [np.asarray(pairs, dtype=np.float64)]
                features.append(GISVectorFeature(
                    geometry_type=str(geometry.get("type") or layer.geometry_type),
                    coordinates=coordinates,
                    properties=dict(item.get("properties") or {}),
                ))
        return features

    def build_raster_overviews(self, layer_id: str, *, levels: tuple[int, ...] = (2, 4, 8, 16)) -> Path:
        """Build durable internal overviews for large raster/DEM layers."""
        layer = self.get(layer_id)
        if layer.kind not in {"raster", "dem"}:
            raise ValueError("Only raster layers support overviews")
        if rasterio is None or Resampling is None:
            raise RuntimeError("Rasterio is required for raster overviews")
        source = self._resolve_layer_source(layer, require_file=True)
        with rasterio.open(source, "r+") as dataset:
            dataset.build_overviews(list(levels), Resampling.average)
            dataset.update_tags(ns="rio_overview", resampling="average")
        return source


__all__ = [
    "GIS_LAYER_SCHEMA",
    "GISLayerRecord",
    "GISLayerStore",
    "GISRasterPreview",
    "GISVectorFeature",
]
