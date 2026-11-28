from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.figure import Figure

from core.field_project_store import FieldLineRecord, FieldProjectStore
from core.gis_layers import GISLayerStore
from core.gis_map_export import export_project_plan_map
from core.job_manager import JobCancelled
from core.trajectory_model import TrajectoryModel, TrajectoryPoint


def _project(tmp_path: Path) -> FieldProjectStore:
    store = FieldProjectStore.create_empty(
        tmp_path / "project",
        name="Field GIS",
        coordinate_system="EPSG:32648",
    )
    store.upsert_line(FieldLineRecord("L01", "Line 1"))
    store.save_trajectory(
        "L01",
        TrajectoryModel(
            [
                TrajectoryPoint(distance_m=0.0, x=500000.0, y=3300000.0, z=100.0),
                TrajectoryPoint(distance_m=10.0, x=500010.0, y=3300004.0, z=101.0),
            ]
        ),
    )
    return store


@pytest.fixture
def raster_project(tmp_path: Path) -> FieldProjectStore:
    rasterio = pytest.importorskip("rasterio")
    store = _project(tmp_path)
    source = tmp_path / "elevation.tif"
    elevation = np.arange(16, dtype=np.float32).reshape(4, 4)

    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs=store.manifest.coordinate_system,
        transform=rasterio.transform.from_origin(499990, 3300020, 5, 5),
    ) as dataset:
        dataset.write(elevation, 1)

    GISLayerStore(store.root).import_layer(source)
    return store


@pytest.fixture
def exported_raster_counts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    counts: list[int] = []
    savefig = Figure.savefig

    def count_rasters(figure, *args, **kwargs):
        savefig(figure, *args, **kwargs)
        counts.append(sum(len(axes.images) for axes in figure.axes))

    monkeypatch.setattr(Figure, "savefig", count_rasters)
    return counts


def test_headless_plan_map_export_writes_real_project_coordinates(tmp_path: Path) -> None:
    store = _project(tmp_path)
    output = store.root / "spatial" / "plan.png"
    progress = []

    result = export_project_plan_map(
        store,
        output,
        selected_line="L01",
        progress_callback=lambda current, total, message: progress.append((current, total, message)),
    )

    assert result == output
    assert output.exists() and output.stat().st_size > 1000
    assert progress[-1][0] == progress[-1][1]


def test_plan_map_cancellation_does_not_replace_previous_file(tmp_path: Path) -> None:
    store = _project(tmp_path)
    output = store.root / "spatial" / "plan.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"previous")

    with pytest.raises(JobCancelled):
        export_project_plan_map(store, output, cancel_requested=lambda: True)

    assert output.read_bytes() == b"previous"
    assert not list(output.parent.glob(".plan.*.tmp.png"))


def test_plan_map_raster_export_reuses_cached_preview(
    raster_project: FieldProjectStore,
    exported_raster_counts: list[int],
) -> None:
    first = raster_project.root / "spatial" / "first.png"
    second = raster_project.root / "spatial" / "second.png"

    with patch.object(
        GISLayerStore,
        "load_raster_preview",
        autospec=True,
        side_effect=GISLayerStore.load_raster_preview,
    ) as direct_read:
        assert export_project_plan_map(raster_project, first) == first
        direct_read.assert_called_once()

        assert export_project_plan_map(raster_project, second) == second
        direct_read.assert_called_once()

    assert exported_raster_counts == [1, 1]
    assert first.stat().st_size > 1000
    assert second.stat().st_size > 1000


def test_plan_map_raster_export_reads_directly_when_cache_raises_oserror(
    raster_project: FieldProjectStore,
    exported_raster_counts: list[int],
) -> None:
    output = raster_project.root / "spatial" / "fallback.png"

    with (
        patch.object(
            GISLayerStore,
            "load_raster_preview_cached",
            autospec=True,
            side_effect=OSError("GIS cache unavailable"),
        ) as cached_read,
        patch.object(
            GISLayerStore,
            "load_raster_preview",
            autospec=True,
            side_effect=GISLayerStore.load_raster_preview,
        ) as direct_read,
    ):
        assert export_project_plan_map(raster_project, output) == output

    cached_read.assert_called_once()
    direct_read.assert_called_once()
    assert direct_read.call_args == cached_read.call_args
    assert exported_raster_counts == [1]
    assert output.stat().st_size > 1000
