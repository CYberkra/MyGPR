#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Online basemap settings and bounded viewport tile loading.

The online layer is deliberately a *view-only* aid.  It requests only the tiles
needed for the current map viewport, keeps a bounded HTTP cache, and never
builds offline archives from providers whose terms may prohibit prefetching.
Offline field use continues to rely on user-supplied GeoTIFF/MBTiles layers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, UnidentifiedImageError

from core.app_paths import get_app_data_dir, get_settings_dir
from core.crs_text import canonical_crs_text
from core.storage_primitives import atomic_write_bytes, atomic_write_json

try:
    from pyproj import CRS, Transformer
    from pyproj.exceptions import CRSError, ProjError
except ImportError:  # pragma: no cover - pyproj is a required runtime dependency
    CRS = None
    Transformer = None
    CRSError = ValueError
    ProjError = RuntimeError


WEB_MERCATOR_CRS = "EPSG:3857"
WEB_MERCATOR_HALF_WORLD = 20037508.342789244
TILE_SIZE = 256
DEFAULT_MAX_TILES = 24
MAX_TILE_RESPONSE_BYTES = 8 * 1024 * 1024
BASEMAP_CACHE_MAX_BYTES = 512 * 1024 * 1024
BASEMAP_SETTINGS_SCHEMA = "mygpr.online_basemap_settings.v1"


@dataclass(frozen=True)
class BasemapProvider:
    provider_id: str
    name: str
    url_template: str
    attribution: str
    min_zoom: int = 1
    max_zoom: int = 18
    requires_token: bool = False
    token_env: str = ""
    cache_ttl_days: int = 7


PROVIDERS: dict[str, BasemapProvider] = {
    "none": BasemapProvider("none", "不使用在线底图", "", "", 1, 18),
    "tianditu_imagery": BasemapProvider(
        provider_id="tianditu_imagery",
        name="天地图影像（需 Key）",
        url_template=(
            "https://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
            "&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles"
            "&TILECOL={x}&TILEROW={y}&TILEMATRIX={z}&tk={token}"
        ),
        attribution="© 国家地理信息公共服务平台 天地图",
        min_zoom=1,
        max_zoom=18,
        requires_token=True,
        token_env="MYGPR_TIANDITU_TOKEN",
        cache_ttl_days=7,
    ),
    "esri_world_imagery": BasemapProvider(
        provider_id="esri_world_imagery",
        name="Esri 世界影像（无需 Key）",
        url_template=(
            "https://services.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attribution=(
            "Powered by Esri · Sources: Esri, Maxar, Earthstar Geographics, "
            "and the GIS User Community"
        ),
        min_zoom=1,
        max_zoom=19,
        cache_ttl_days=7,
    ),
    "custom_xyz": BasemapProvider(
        provider_id="custom_xyz",
        name="自定义 XYZ / WMTS 模板",
        url_template="",
        attribution="",
        min_zoom=1,
        max_zoom=22,
        cache_ttl_days=7,
    ),
}


@dataclass
class OnlineBasemapSettings:
    schema: str = BASEMAP_SETTINGS_SCHEMA
    enabled: bool = False
    provider_id: str = "none"
    token: str = ""
    custom_url: str = ""
    custom_attribution: str = ""
    max_tiles: int = DEFAULT_MAX_TILES

    @classmethod
    def from_dict(cls, payload: dict) -> "OnlineBasemapSettings":
        allowed = cls.__dataclass_fields__.keys()
        values = {key: payload[key] for key in allowed if key in payload}
        settings = cls(**values)
        settings.max_tiles = int(np.clip(int(settings.max_tiles), 1, 64))
        if settings.provider_id not in PROVIDERS:
            settings.provider_id = "none"
            settings.enabled = False
        return settings


@dataclass
class BasemapPreview:
    array: np.ndarray
    extent: tuple[float, float, float, float]
    crs: str
    attribution: str
    provider_id: str
    zoom: int
    tile_count: int
    cached_tile_count: int


def settings_path() -> Path:
    return Path(get_settings_dir()) / "online_basemap.json"


def load_settings(path: str | Path | None = None) -> OnlineBasemapSettings:
    source = Path(path) if path is not None else settings_path()
    if not source.exists():
        return OnlineBasemapSettings()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return OnlineBasemapSettings()
    return OnlineBasemapSettings.from_dict(payload) if isinstance(payload, dict) else OnlineBasemapSettings()


def save_settings(settings: OnlineBasemapSettings, path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, asdict(settings))
    return target


def resolved_provider(settings: OnlineBasemapSettings) -> BasemapProvider:
    provider = PROVIDERS.get(settings.provider_id, PROVIDERS["none"])
    if provider.provider_id != "custom_xyz":
        return provider
    return BasemapProvider(
        provider_id="custom_xyz",
        name="自定义 XYZ / WMTS 模板",
        url_template=settings.custom_url.strip(),
        attribution=settings.custom_attribution.strip(),
        min_zoom=1,
        max_zoom=22,
        cache_ttl_days=7,
    )


def validate_settings(settings: OnlineBasemapSettings) -> None:
    provider = resolved_provider(settings)
    if not settings.enabled or provider.provider_id == "none":
        raise ValueError("请先启用并选择在线底图。")
    template = provider.url_template.strip()
    if not template:
        raise ValueError("在线底图 URL 模板不能为空。")
    for placeholder in ("{z}", "{x}", "{y}"):
        if placeholder not in template:
            raise ValueError(f"在线底图 URL 模板缺少 {placeholder}。")
    parsed = urlparse(template.replace("{z}", "1").replace("{x}", "0").replace("{y}", "0").replace("{token}", "token"))
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("在线底图 URL 必须使用 HTTP 或 HTTPS。")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("非本机在线底图必须使用 HTTPS。")
    token = settings.token.strip() or (os.environ.get(provider.token_env, "").strip() if provider.token_env else "")
    if provider.requires_token and not token:
        raise ValueError("天地图影像需要 Key；可在设置中填写或设置 MYGPR_TIANDITU_TOKEN。")


def infer_source_crs(bounds: tuple[float, float, float, float], declared_crs: str = "") -> str:
    text = str(declared_crs or "").strip()
    if text:
        try:
            return canonical_crs_text(text, strict=True)
        except ValueError as exc:
            raise ValueError(f"无法解析工程坐标系：{text}") from exc
    left, bottom, right, top = bounds
    if -180.0 <= left <= 180.0 and -180.0 <= right <= 180.0 and -90.0 <= bottom <= 90.0 and -90.0 <= top <= 90.0:
        return "EPSG:4326"
    raise ValueError("工程坐标系未设置，且坐标范围不像经纬度；无法叠加在线底图。")


def transform_bounds_to_web_mercator(
    bounds: tuple[float, float, float, float], source_crs: str
) -> tuple[float, float, float, float]:
    if Transformer is None or CRS is None:
        raise RuntimeError("在线底图坐标转换需要 pyproj。")
    source = CRS.from_user_input(canonical_crs_text(source_crs, strict=True))
    transformer = Transformer.from_crs(source, WEB_MERCATOR_CRS, always_xy=True)
    left, bottom, right, top = transformer.transform_bounds(*bounds, densify_pts=21)
    values = np.asarray([left, bottom, right, top], dtype=np.float64)
    if not np.isfinite(values).all() or right <= left or top <= bottom:
        raise ValueError("工程空间范围无效，无法加载在线底图。")
    limit = WEB_MERCATOR_HALF_WORLD
    return (
        float(np.clip(left, -limit, limit)),
        float(np.clip(bottom, -limit, limit)),
        float(np.clip(right, -limit, limit)),
        float(np.clip(top, -limit, limit)),
    )




def is_geographic_crs(value: str) -> bool:
    if not value or CRS is None:
        return str(value).upper() in {"EPSG:4326", "WGS84", "WGS 84"}
    try:
        return bool(CRS.from_user_input(canonical_crs_text(value, strict=True)).is_geographic)
    except (CRSError, ProjError, TypeError, ValueError):
        return False


def transform_xy(
    x_values: np.ndarray | list[float],
    y_values: np.ndarray | list[float],
    source_crs: str,
    target_crs: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform coordinate arrays while preserving float64 and NaN positions."""
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("X/Y 坐标数组长度不一致。")
    if not source_crs or not target_crs or str(source_crs) == str(target_crs):
        return x.copy(), y.copy()
    if Transformer is None:
        raise RuntimeError("坐标转换需要 pyproj。")
    source = canonical_crs_text(source_crs, strict=True)
    target = canonical_crs_text(target_crs, strict=True)
    transformer = Transformer.from_crs(source, target, always_xy=True)
    out_x = np.full(x.shape, np.nan, dtype=np.float64)
    out_y = np.full(y.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        if int(np.count_nonzero(finite)) == 1:
            index = int(np.flatnonzero(finite)[0])
            tx, ty = transformer.transform(float(x.flat[index]), float(y.flat[index]))
            out_x.flat[index] = float(tx)
            out_y.flat[index] = float(ty)
        else:
            tx, ty = transformer.transform(x[finite], y[finite])
            out_x[finite] = np.asarray(tx, dtype=np.float64)
            out_y[finite] = np.asarray(ty, dtype=np.float64)
    return out_x, out_y

def choose_zoom(
    bounds_3857: tuple[float, float, float, float],
    *,
    pixel_width: int,
    pixel_height: int,
    provider: BasemapProvider,
    max_tiles: int,
) -> int:
    left, bottom, right, top = bounds_3857
    width = max(right - left, 1.0)
    height = max(top - bottom, 1.0)
    target_resolution = max(width / max(pixel_width, 256), height / max(pixel_height, 256), 1e-9)
    world_width = WEB_MERCATOR_HALF_WORLD * 2.0
    estimated = int(math.floor(math.log2(world_width / (TILE_SIZE * target_resolution))))
    zoom = int(np.clip(estimated, provider.min_zoom, provider.max_zoom))
    while zoom > provider.min_zoom and tile_count_for_bounds(bounds_3857, zoom) > max_tiles:
        zoom -= 1
    return zoom


def tile_range_for_bounds(
    bounds_3857: tuple[float, float, float, float], zoom: int
) -> tuple[int, int, int, int]:
    left, bottom, right, top = bounds_3857
    n = 1 << int(zoom)
    world = WEB_MERCATOR_HALF_WORLD * 2.0

    def tile_x(x: float) -> int:
        return int(math.floor((x + WEB_MERCATOR_HALF_WORLD) / world * n))

    def tile_y(y: float) -> int:
        return int(math.floor((WEB_MERCATOR_HALF_WORLD - y) / world * n))

    x0 = int(np.clip(tile_x(left), 0, n - 1))
    x1 = int(np.clip(tile_x(np.nextafter(right, left)), 0, n - 1))
    y0 = int(np.clip(tile_y(top), 0, n - 1))
    y1 = int(np.clip(tile_y(np.nextafter(bottom, top)), 0, n - 1))
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def tile_count_for_bounds(bounds_3857: tuple[float, float, float, float], zoom: int) -> int:
    x0, x1, y0, y1 = tile_range_for_bounds(bounds_3857, zoom)
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def tile_mosaic_extent(x0: int, x1: int, y0: int, y1: int, zoom: int) -> tuple[float, float, float, float]:
    n = 1 << int(zoom)
    world = WEB_MERCATOR_HALF_WORLD * 2.0
    left = x0 / n * world - WEB_MERCATOR_HALF_WORLD
    right = (x1 + 1) / n * world - WEB_MERCATOR_HALF_WORLD
    top = WEB_MERCATOR_HALF_WORLD - y0 / n * world
    bottom = WEB_MERCATOR_HALF_WORLD - (y1 + 1) / n * world
    return float(left), float(right), float(bottom), float(top)


def _cache_root() -> Path:
    root = Path(get_app_data_dir()) / "basemap_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _provider_cache_namespace(provider: BasemapProvider) -> str:
    fingerprint = hashlib.sha256(
        f"{provider.provider_id}\0{provider.url_template}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{provider.provider_id}-{fingerprint}"


def _tile_cache_path(provider: BasemapProvider, zoom: int, x: int, y: int, cache_root: Path) -> Path:
    return cache_root / _provider_cache_namespace(provider) / str(zoom) / str(x) / f"{y}.png"


def _prune_tile_cache(cache_root: Path, *, max_bytes: int = BASEMAP_CACHE_MAX_BYTES) -> int:
    """Remove oldest cached tiles until the global cache is within *max_bytes*."""
    if max_bytes < 0 or not cache_root.exists():
        return 0
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for path in cache_root.rglob("*.png"):
        try:
            stat = path.stat()
        except OSError:
            continue
        size = max(int(stat.st_size), 0)
        total += size
        entries.append((float(stat.st_mtime), size, path))
    removed = 0
    for _mtime, size, path in sorted(entries):
        if total <= max_bytes:
            break
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
        total -= size
        removed += 1
    return removed


def _read_cached_tile(path: Path, ttl_days: int) -> Image.Image | None:
    if not path.is_file():
        return None
    age_seconds = max(time.time() - path.stat().st_mtime, 0.0)
    if age_seconds > max(int(ttl_days), 1) * 86400:
        return None
    try:
        with Image.open(path) as image:
            return image.convert("RGBA")
    except (OSError, UnidentifiedImageError):
        path.unlink(missing_ok=True)
        return None


def _download_tile(url: str, target: Path) -> Image.Image:
    request = Request(
        url,
        headers={
            "User-Agent": "MyGPR/0.9.28 (desktop field geophysics basemap viewer)",
            "Accept": "image/png,image/jpeg,image/webp,*/*;q=0.5",
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            data = response.read(MAX_TILE_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"底图瓦片下载失败：{exc}") from exc
    if len(data) > MAX_TILE_RESPONSE_BYTES:
        raise ValueError("底图瓦片响应过大，已拒绝。")
    try:
        with Image.open(BytesIO(data)) as image:
            rgba = image.convert("RGBA")
            if rgba.width > 1024 or rgba.height > 1024 or rgba.width <= 0 or rgba.height <= 0:
                raise ValueError(f"底图瓦片尺寸异常：{rgba.width}×{rgba.height}")
            rgba.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("底图服务返回的不是有效图像。") from exc
    buffer = BytesIO()
    rgba.save(buffer, format="PNG", optimize=True)
    atomic_write_bytes(target, buffer.getvalue())
    return rgba


def fetch_viewport_preview(
    bounds: tuple[float, float, float, float],
    *,
    source_crs: str,
    pixel_width: int,
    pixel_height: int,
    settings: OnlineBasemapSettings,
    cache_root: str | Path | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> BasemapPreview:
    """Fetch only the tiles required by the current project viewport."""
    validate_settings(settings)
    provider = resolved_provider(settings)
    source = infer_source_crs(bounds, source_crs)
    bounds_3857 = transform_bounds_to_web_mercator(bounds, source)
    max_tiles = int(np.clip(settings.max_tiles, 1, 64))
    zoom = choose_zoom(
        bounds_3857,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        provider=provider,
        max_tiles=max_tiles,
    )
    x0, x1, y0, y1 = tile_range_for_bounds(bounds_3857, zoom)
    total = (x1 - x0 + 1) * (y1 - y0 + 1)
    if total > max_tiles:
        raise ValueError(f"当前范围需要 {total} 张瓦片，超过安全上限 {max_tiles}。")
    token = settings.token.strip() or (os.environ.get(provider.token_env, "").strip() if provider.token_env else "")
    root = Path(cache_root) if cache_root is not None else _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    _prune_tile_cache(root)
    tile_images: dict[tuple[int, int], Image.Image] = {}
    cached_count = 0
    completed = 0
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            if cancel_requested is not None and cancel_requested():
                from core.job_manager import JobCancelled

                raise JobCancelled("在线底图加载已取消")
            path = _tile_cache_path(provider, zoom, x, y, root)
            image = _read_cached_tile(path, provider.cache_ttl_days)
            if image is not None:
                cached_count += 1
            else:
                url = provider.url_template.format(z=zoom, x=x, y=y, token=token)
                image = _download_tile(url, path)
            if image.size != (TILE_SIZE, TILE_SIZE):
                image = image.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.BILINEAR)
            tile_images[(x, y)] = image
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, f"加载在线底图 {completed}/{total}")

    mosaic = Image.new(
        "RGBA",
        ((x1 - x0 + 1) * TILE_SIZE, (y1 - y0 + 1) * TILE_SIZE),
        (242, 244, 247, 255),
    )
    for (x, y), image in tile_images.items():
        mosaic.paste(image, ((x - x0) * TILE_SIZE, (y - y0) * TILE_SIZE))
    array = np.asarray(mosaic.convert("RGB"), dtype=np.uint8)
    _prune_tile_cache(root)
    return BasemapPreview(
        array=array,
        extent=tile_mosaic_extent(x0, x1, y0, y1, zoom),
        crs=WEB_MERCATOR_CRS,
        attribution=provider.attribution,
        provider_id=provider.provider_id,
        zoom=zoom,
        tile_count=total,
        cached_tile_count=cached_count,
    )


__all__ = [
    "BASEMAP_CACHE_MAX_BYTES",
    "BASEMAP_SETTINGS_SCHEMA",
    "BasemapPreview",
    "BasemapProvider",
    "DEFAULT_MAX_TILES",
    "OnlineBasemapSettings",
    "PROVIDERS",
    "WEB_MERCATOR_CRS",
    "choose_zoom",
    "fetch_viewport_preview",
    "infer_source_crs",
    "is_geographic_crs",
    "load_settings",
    "resolved_provider",
    "save_settings",
    "tile_count_for_bounds",
    "tile_range_for_bounds",
    "transform_bounds_to_web_mercator",
    "transform_xy",
    "validate_settings",
]
