from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_spatial_page_exposes_real_basemap_and_trace_linkage_controls() -> None:
    text = (ROOT / "ui" / "field_panels" / "spatial_page.py").read_text(encoding="utf-8")
    assert "真实地貌底图" in text
    assert "配置在线底图" in text
    assert "加载当前范围" in text
    assert "GeoTIFF / MBTiles" in text
    assert "spatial_selected_trace" in text
    assert "processing_live_trace_index = best_trace" in text
    assert "_action_spatial_screenshot" in text
    assert "canvas.figure.savefig" in text


def test_gis_import_accepts_mbtiles_as_offline_raster() -> None:
    gis_text = (ROOT / "core" / "gis_layers.py").read_text(encoding="utf-8")
    spatial_text = (ROOT / "ui" / "field_panels" / "spatial_page.py").read_text(encoding="utf-8")
    assert '".mbtiles"' in gis_text
    assert "*.mbtiles" in spatial_text
