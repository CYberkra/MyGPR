from pathlib import Path
import numpy as np
from core.crs_contract import CRSDefinition
from core.gis_cache import GISCacheKey, GISCacheManager
from core.gis_layers import GISRasterPreview


def test_crs_definition_is_canonical_wkt():
    definition = CRSDefinition.parse("EPSG:4326")
    assert definition.authority == "EPSG:4326"
    assert "GEOGCRS" in definition.wkt or "GEOGCS" in definition.wkt


def test_gis_preview_cache_roundtrip(tmp_path: Path):
    manager = GISCacheManager(tmp_path)
    key = GISCacheKey("abc", "EPSG:4326", 1, 100)
    calls = []
    def load():
        calls.append(1)
        return GISRasterPreview(np.ones((2, 2), dtype=np.float32), (0, 1, 0, 1), "EPSG:4326", is_dem=True)
    first = manager.get_or_create_raster(key, load)
    second = manager.get_or_create_raster(key, load)
    assert len(calls) == 1
    assert np.array_equal(first.array, second.array)
