#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content-addressed GIS preview cache with explicit invalidation contracts."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.array_storage import atomic_save_npz_compressed
from core.storage_primitives import atomic_write_json, utc_now


@dataclass(frozen=True)
class GISCacheKey:
    source_sha256: str
    target_crs: str
    style_version: int
    max_size: int
    bbox: tuple[float, float, float, float] | None = None

    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GISCacheManager:
    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve() / "spatial" / ".cache"
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_create_raster(self, key: GISCacheKey, loader: Callable[[], Any]) -> Any:
        digest = key.digest()
        data_path = self.root / f"{digest}.npz"
        meta_path = self.root / f"{digest}.json"
        if data_path.exists() and meta_path.exists():
            with np.load(data_path, allow_pickle=False) as archive:
                from core.gis_layers import GISRasterPreview
                extent_values = tuple(float(value) for value in archive["extent"])
                if len(extent_values) != 4:
                    raise ValueError("Invalid cached raster extent")
                extent = (extent_values[0], extent_values[1], extent_values[2], extent_values[3])
                return GISRasterPreview(
                    array=archive["array"],
                    extent=extent,
                    crs=str(archive["crs"].item()),
                    nodata=float(archive["nodata"].item()) if bool(archive["has_nodata"].item()) else None,
                    is_dem=bool(archive["is_dem"].item()),
                )
        preview = loader()
        atomic_save_npz_compressed(data_path, {
            "array": np.asarray(preview.array),
            "extent": np.asarray(preview.extent, dtype=np.float64),
            "crs": np.asarray(preview.crs),
            "nodata": np.asarray(0.0 if preview.nodata is None else preview.nodata),
            "has_nodata": np.asarray(preview.nodata is not None),
            "is_dem": np.asarray(preview.is_dem),
        })
        atomic_write_json(meta_path, {
            "schema": "mygpr.gis_cache_entry.v1",
            "created_at": utc_now(),
            "key": key.__dict__,
            "data_file": data_path.name,
        })
        return preview

    def invalidate(self, *, source_sha256: str | None = None) -> int:
        removed = 0
        for meta in self.root.glob("*.json"):
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {}
            if source_sha256 and payload.get("key", {}).get("source_sha256") != source_sha256:
                continue
            meta.unlink(missing_ok=True)
            meta.with_suffix(".npz").unlink(missing_ok=True)
            removed += 1
        return removed

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)


__all__ = ["GISCacheKey", "GISCacheManager"]
