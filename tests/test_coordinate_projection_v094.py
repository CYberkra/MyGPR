from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from core.coordinate_projection import infer_3deg_zone_from_longitude, project_lonlat_to_xy, resolve_projection_spec
from core.field_project_store import FieldProjectStore


def test_resolve_cgcs2000_3deg_zone_39() -> None:
    spec = resolve_projection_spec("CGCS2000 / 3-degree GK Zone 39", mean_longitude=116.3)
    assert spec.epsg == 4527
    assert spec.zone == 39


def test_infer_yingshan_zone_from_longitude() -> None:
    assert infer_3deg_zone_from_longitude(106.8) == 36
    spec = resolve_projection_spec("", mean_longitude=106.8)
    assert spec.epsg == 4524
    assert spec.is_auto is True


def test_project_lonlat_to_engineering_xy() -> None:
    x, y, spec = project_lonlat_to_xy([116.3, 116.3001], [39.9, 39.9001], coordinate_system="CGCS2000 / 3-degree GK Zone 39")
    assert spec.epsg == 4527
    assert x.shape == (2,)
    assert y.shape == (2,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert float(abs(x[0])) > 1000.0
    assert float(abs(y[0])) > 1000.0


def test_sidecar_import_writes_projected_trajectory(tmp_path: Path) -> None:
    project = FieldProjectStore.create_empty(
        tmp_path / "project",
        name="projection-test",
        coordinate_system="CGCS2000 / 3-degree GK Zone 39",
    )
    line = project.import_line_file("L01", Path("sample_data/gui_sidecar_all_data_main.csv"), name="projection-line")
    assert line.rtk_status == "已投影"
    trajectory = project.load_trajectory("L01")
    assert len(trajectory.points) == 12
    point = trajectory.points[0]
    assert point.coordinate_system.startswith("CGCS2000 / 3-degree GK Zone 39")
    assert abs(point.x - point.longitude) > 1000.0
    assert abs(point.y - point.latitude) > 1000.0
    assert abs(point.longitude - 116.3) < 1e-4
    assert abs(point.latitude - 39.9) < 1e-4
    manifest = json.loads((project.root / "raw" / "L01" / "import_manifest.json").read_text(encoding="utf-8"))
    assert manifest["projection"]["status"] == "ok"
    assert manifest["projection"]["epsg"] == 4527
