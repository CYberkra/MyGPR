from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.coordinate_projection import resolve_projection_spec
from core.field_project_models import FieldLineRecord, validate_line_id
from core.field_project_operations import import_line_data
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet


def test_cgcs2000_zone_39_uses_zone_epsg_not_cm_epsg() -> None:
    spec = resolve_projection_spec("CGCS2000 / 3-degree GK Zone 39", mean_longitude=106.0)
    assert spec.zone == 39
    assert spec.epsg == 4527


def test_line_id_validation_rejects_path_like_values() -> None:
    assert validate_line_id("L09_30") == "L09_30"
    for bad in ["../L01", "L01/evil", "", "9Line", "CON"]:
        with pytest.raises(ValueError):
            validate_line_id(bad)


def test_failed_formal_import_rolls_back_manifest_and_raw_dir(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="rollback-test")
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("not,a,gpr,file\n", encoding="utf-8")

    with pytest.raises(Exception):
        import_line_data(store, bad_csv, line_id="L01", name="bad-line")

    assert store.list_lines() == []
    assert not (store.root / "raw" / "L01").exists()


def test_import_rejects_unsafe_line_id_without_creating_paths(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="safe-id-test")
    src = Path("sample_data/gui_sidecar_all_data_main.csv")
    with pytest.raises(ValueError):
        import_line_data(store, src, line_id="../escape", name="bad")
    assert not (tmp_path / "escape").exists()
    assert store.list_lines() == []


def test_missing_trajectory_does_not_return_demo(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="no-demo-trajectory")
    store.upsert_line(FieldLineRecord("L01", "line-without-trajectory"))
    dataset = GPRDataSet.from_matrix("L01", np.ones((64, 128), dtype=np.float32), length_m=10.0)
    store.save_gpr_dataset("L01", dataset)
    with pytest.raises(FileNotFoundError):
        store.load_trajectory("L01")


def test_transpose_manifest_records_axis_warning(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="transpose-warning")
    store.upsert_line(FieldLineRecord("L01", "risk-line"))
    dataset = GPRDataSet.from_matrix("L01", np.ones((300, 20), dtype=np.float32), length_m=25.0, time_window_ns=700.0)
    store.save_gpr_dataset("L01", dataset)
    store.transpose_gpr_dataset("L01")
    manifest = (store.root / "raw" / "L01" / "orientation_fix_manifest.json").read_text(encoding="utf-8")
    assert "axis_rebuild_policy" in manifest
    assert "axis_warning" in manifest
