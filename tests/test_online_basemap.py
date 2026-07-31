from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest

from core.online_basemap import (
    OnlineBasemapSettings,
    _prune_tile_cache,
    fetch_viewport_preview,
    load_settings,
    save_settings,
    tile_count_for_bounds,
    transform_bounds_to_web_mercator,
    transform_xy,
    validate_settings,
)


def _png_bytes() -> bytes:
    image = Image.new("RGB", (256, 256), (40, 90, 130))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _TileHandler(BaseHTTPRequestHandler):
    payload = _png_bytes()
    requests = 0

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        type(self).requests += 1
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, _format: str, *_args) -> None:
        return


def test_settings_round_trip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "online_basemap.json"
    settings = OnlineBasemapSettings(
        enabled=True,
        provider_id="custom_xyz",
        custom_url="https://tiles.example/{z}/{x}/{y}.png",
        custom_attribution="Example imagery",
        max_tiles=18,
    )
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded == settings
    validate_settings(loaded)

    loaded.custom_url = "http://tiles.example/{z}/{x}/{y}.png"
    with pytest.raises(ValueError, match="HTTPS"):
        validate_settings(loaded)


def test_tile_math_is_bounded_for_small_project_extent() -> None:
    bounds = transform_bounds_to_web_mercator((104.0, 30.0, 104.01, 30.01), "EPSG:4326")
    assert tile_count_for_bounds(bounds, 16) <= 16


def test_viewport_fetch_uses_cache_and_returns_web_mercator_preview(tmp_path: Path) -> None:
    _TileHandler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TileHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        settings = OnlineBasemapSettings(
            enabled=True,
            provider_id="custom_xyz",
            custom_url=f"http://{host}:{port}/{{z}}/{{x}}/{{y}}.png",
            custom_attribution="Local test tiles",
            max_tiles=16,
        )
        preview = fetch_viewport_preview(
            (104.0, 30.0, 104.002, 30.002),
            source_crs="EPSG:4326",
            pixel_width=640,
            pixel_height=480,
            settings=settings,
            cache_root=tmp_path / "cache",
        )
        assert preview.crs == "EPSG:3857"
        assert preview.array.dtype == np.uint8
        assert preview.array.ndim == 3 and preview.array.shape[2] == 3
        assert 1 <= preview.tile_count <= 16
        assert preview.cached_tile_count == 0
        first_request_count = _TileHandler.requests
        assert first_request_count == preview.tile_count

        cached = fetch_viewport_preview(
            (104.0, 30.0, 104.002, 30.002),
            source_crs="EPSG:4326",
            pixel_width=640,
            pixel_height=480,
            settings=settings,
            cache_root=tmp_path / "cache",
        )
        assert cached.cached_tile_count == cached.tile_count
        assert _TileHandler.requests == first_request_count
    finally:
        server.shutdown()
        server.server_close()



def test_custom_sources_use_isolated_cache_namespaces(tmp_path: Path) -> None:
    class DistinctTileHandler(_TileHandler):
        requests = 0

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            type(self).requests += 1
            colour = (180, 20, 20) if self.path.startswith("/red/") else (20, 40, 180)
            image = Image.new("RGB", (256, 256), colour)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), DistinctTileHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        cache = tmp_path / "cache"
        common = dict(
            enabled=True,
            provider_id="custom_xyz",
            custom_attribution="Local test tiles",
            max_tiles=16,
        )
        blue = fetch_viewport_preview(
            (104.0, 30.0, 104.002, 30.002),
            source_crs="EPSG:4326",
            pixel_width=640,
            pixel_height=480,
            settings=OnlineBasemapSettings(
                custom_url=f"http://{host}:{port}/blue/{{z}}/{{x}}/{{y}}.png", **common
            ),
            cache_root=cache,
        )
        requests_after_blue = DistinctTileHandler.requests
        red = fetch_viewport_preview(
            (104.0, 30.0, 104.002, 30.002),
            source_crs="EPSG:4326",
            pixel_width=640,
            pixel_height=480,
            settings=OnlineBasemapSettings(
                custom_url=f"http://{host}:{port}/red/{{z}}/{{x}}/{{y}}.png", **common
            ),
            cache_root=cache,
        )
        assert DistinctTileHandler.requests > requests_after_blue
        assert not np.array_equal(blue.array, red.array)
        namespaces = [path for path in cache.iterdir() if path.is_dir()]
        assert len(namespaces) == 2
    finally:
        server.shutdown()
        server.server_close()


def test_tile_cache_prunes_oldest_files_to_capacity(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    paths = [cache / "source" / "1" / "0" / f"{index}.png" for index in range(3)]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]) * 100)
        path.touch()
    removed = _prune_tile_cache(cache, max_bytes=150)
    assert removed == 2
    assert sum(path.stat().st_size for path in cache.rglob("*.png")) <= 150


def test_transform_xy_preserves_float64_and_nan_positions() -> None:
    x = np.asarray([104.0, np.nan, 104.1], dtype=np.float64)
    y = np.asarray([30.0, 30.1, np.nan], dtype=np.float64)
    out_x, out_y = transform_xy(x, y, "EPSG:4326", "EPSG:3857")
    assert out_x.dtype == np.float64
    assert out_y.dtype == np.float64
    assert np.isfinite(out_x[0]) and np.isfinite(out_y[0])
    assert np.isnan(out_x[1]) and np.isnan(out_y[1])
    assert np.isnan(out_x[2]) and np.isnan(out_y[2])
