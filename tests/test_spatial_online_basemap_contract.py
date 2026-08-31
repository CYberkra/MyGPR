from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_spatial_page_exposes_basemap_and_prefetch_controls() -> None:
    text = (ROOT / "ui" / "pages" / "spatial_page.py").read_text(encoding="utf-8")
    assert "底图" in text
    assert "预下载测线区域" in text
    assert "basemap_prefetch_requested" in text
    assert "_on_basemap_changed" in text
    assert "已配准底图" in text
    assert "原始坐标（未配准底图）" in text


def test_gis_import_accepts_mbtiles_as_offline_raster() -> None:
    gis_text = (ROOT / "core" / "gis_layers.py").read_text(encoding="utf-8")
    assert '".mbtiles"' in gis_text
    assert "MBTiles" in gis_text
