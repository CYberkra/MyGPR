from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.gis_layers import GISLayerStore


def test_geojson_layer_is_copied_registered_and_transformed(tmp_path: Path) -> None:
    source = tmp_path / "survey_line.geojson"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"line_id": "L01"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[104.0, 30.0], [104.001, 30.0]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    project = tmp_path / "project"
    store = GISLayerStore(project)

    layer = store.import_layer(source, role="survey_line")
    assert layer.kind == "vector"
    assert layer.crs == "EPSG:4326"
    assert layer.geometry_type == "LineString"
    assert (project / layer.source_path).exists()
    assert store.list_layers()[0].role == "survey_line"

    features = store.load_vector(layer, target_crs="EPSG:32648")
    assert len(features) == 1
    coords = features[0].coordinates[0]
    assert coords.shape == (2, 2)
    assert coords[1, 0] > coords[0, 0]
    assert coords[0, 0] > 100000

    store.update(layer.layer_id, visible=False, opacity=0.4)
    updated = store.get(layer.layer_id)
    assert updated.visible is False
    assert updated.opacity == 0.4
    store.remove(layer.layer_id)
    assert store.list_layers() == []
    assert not (project / layer.source_path).exists()


def test_geotiff_dem_uses_real_georeferencing_and_single_band_semantics(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    source = tmp_path / "dem.tif"
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=4,
        width=5,
        count=1,
        dtype="float32",
        crs="EPSG:32648",
        transform=from_origin(500000.0, 3300000.0, 2.0, 2.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)

    store = GISLayerStore(tmp_path / "project")
    layer = store.import_layer(source, role="dem")
    preview = store.load_raster_preview(layer, max_size=100)

    assert layer.kind == "raster"
    assert layer.metadata["is_dem"] is True
    assert preview.is_dem is True
    assert preview.array.shape == (4, 5)
    assert preview.extent == pytest.approx((500000.0, 500010.0, 3299992.0, 3300000.0))
    assert preview.crs == "EPSG:32648"


def test_gis_import_cancellation_rolls_back_staging_directory(tmp_path: Path) -> None:
    from core.job_manager import JobCancelled

    source = tmp_path / "large.geojson"
    source.write_text(
        '{"type":"FeatureCollection","features":[]}' + (" " * (12 * 1024 * 1024)),
        encoding="utf-8",
    )
    project = tmp_path / "project_cancel"
    store = GISLayerStore(project)
    cancel = False

    def progress(current: int, _total: int, _message: str) -> None:
        nonlocal cancel
        if current > 0:
            cancel = True

    with pytest.raises(JobCancelled):
        store.import_layer(
            source,
            cancel_requested=lambda: cancel,
            progress_callback=progress,
        )

    assert store.list_layers() == []
    layer_dirs = list((project / "spatial" / "gis_layers").glob("*"))
    assert layer_dirs == []
