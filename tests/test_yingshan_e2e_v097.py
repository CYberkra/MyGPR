from __future__ import annotations

import csv
import shutil
from pathlib import Path

from core.coordinate_projection import resolve_projection_spec
from core.field_project_operations import batch_import_line_data, infer_line_identity_from_filename
from core.field_project_store import FieldProjectStore


def test_yingshan_line_l_and_x_filename_inference() -> None:
    l1 = infer_line_identity_from_filename("LineL1origin(36).csv", fallback_index=5)
    x1 = infer_line_identity_from_filename("LineX1origin(36).csv", fallback_index=6)
    assert l1.line_id == "L01_36"
    assert l1.name == "L1号测线（36）"
    assert l1.confidence == "high"
    assert x1.line_id == "X1_36"
    assert x1.name == "X1号测线（36）"
    assert x1.confidence == "high"


def test_auto_cgcs2000_3deg_gk_marks_projection_auto() -> None:
    spec = resolve_projection_spec("CGCS2000 / 3-degree GK", mean_longitude=106.8)
    assert spec.zone == 36
    assert spec.epsg == 4524
    assert spec.is_auto is True
    assert "auto" in spec.name.lower()


def test_batch_import_preserves_l_and_x_semantic_line_ids(tmp_path: Path) -> None:
    source = Path("sample_data/gui_sidecar_all_data_main.csv")
    l1 = tmp_path / "LineL1origin(36).csv"
    x1 = tmp_path / "LineX1origin(36).csv"
    shutil.copy2(source, l1)
    shutil.copy2(source, x1)
    store = FieldProjectStore.create_empty(tmp_path / "project", name="semantic-lines", coordinate_system="CGCS2000 / 3-degree GK")

    summary = batch_import_line_data(store, [l1, x1])

    assert summary.succeeded == 2
    lines = {line.line_id: line for line in store.list_lines()}
    assert "L01_36" in lines
    assert "X1_36" in lines
    assert lines["L01_36"].gpr_dataset_path
    assert lines["X1_36"].gpr_dataset_path


def test_target_save_does_not_accept_embedded_illegal_line_id(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="target-line-safety")
    store.save_targets(
        "L01",
        [
            {
                "target_id": "T-01",
                "line_id": "../evil",
                "distance_m": 1.0,
                "depth_m": 0.5,
                "type": "测试目标",
            }
        ],
    )
    rows = list(csv.DictReader(store.targets_path("L01").open("r", encoding="utf-8-sig")))
    assert rows[0]["line_id"] == "L01"
