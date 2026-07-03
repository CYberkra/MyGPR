from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.field_processing_bridge import run_registered_method
from core.processing_artifact_index import latest_processing_artifact
from core.target_source_binding import artifact_target_source, bind_target_to_source, raw_target_source


def _read_targets_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_targets_persist_raw_source_binding(tmp_path: Path) -> None:
    store = FieldProjectStore.create(tmp_path / "project")
    dataset = GPRDataSet.synthetic("L03", rows=40, cols=32, length_m=16.0)
    source = raw_target_source(dataset, line_id="L03")
    target = bind_target_to_source(
        {
            "target_id": "T-99",
            "line_id": "L03",
            "mileage": 4.2,
            "depth": 1.1,
            "type": "疑似管线",
            "confidence": "★★★☆☆",
            "status": "待确认",
        },
        source,
    )
    path = store.save_targets("L03", [target])
    rows = _read_targets_csv(path)
    assert rows[0]["source_result_id"] == "L03_raw"
    assert rows[0]["source_mode"] == "raw"
    assert rows[0]["source_method_id"] == "raw"
    loaded = store.load_targets("L03")
    assert loaded[0]["source_result_id"] == "L03_raw"
    assert loaded[0]["source_method_name"] == "原始 B-scan"


def test_targets_persist_specific_processed_artifact_binding(tmp_path: Path) -> None:
    store = FieldProjectStore.create(tmp_path / "project")
    dataset = GPRDataSet.synthetic("L03", rows=60, cols=40, length_m=20.0)
    output, manifest = run_registered_method(dataset, "dewow", {"window": 21})
    data_path, _params_path = store.save_processed_line(
        "L03",
        np.asarray(output.matrix),
        {
            "method": "dewow",
            "method_name": "去低频漂移 dewow",
            "params": {"window": 21},
            "manifest": manifest,
            "input_dataset": dataset.to_metadata(),
        },
    )
    record = latest_processing_artifact(store.root, "L03")
    assert record is not None
    source = artifact_target_source(record)
    target = bind_target_to_source(
        {"target_id": "T-02", "line_id": "L03", "mileage": 7.5, "depth": 1.4},
        source,
    )
    path = store.save_targets("L03", [target])
    rows = _read_targets_csv(path)
    assert rows[0]["source_result_id"] == data_path.stem
    assert rows[0]["source_mode"] == "processed"
    assert rows[0]["source_data_path"].endswith(".npy")
    assert rows[0]["source_manifest_path"].endswith(".json")
    assert rows[0]["source_method_id"] == "dewow"
    assert rows[0]["source_method_name"] == "去低频漂移 dewow"


def test_targets_can_bind_time_to_depth_display_compare_artifact(tmp_path: Path) -> None:
    store = FieldProjectStore.create(tmp_path / "project")
    dataset = GPRDataSet.synthetic("L03", rows=60, cols=40, length_m=20.0)
    output, manifest = run_registered_method(dataset, "time_to_depth", {})
    store.save_processed_line(
        "L03",
        np.asarray(output.matrix),
        {
            "method": "time_to_depth",
            "method_name": "时间-深度转换",
            "params": {},
            "manifest": manifest,
            "input_dataset": dataset.to_metadata(),
        },
    )
    record = latest_processing_artifact(store.root, "L03")
    assert record is not None
    source = artifact_target_source(record)
    assert source.source_mode == "display_compare"
    target = bind_target_to_source({"target_id": "T-03", "line_id": "L03", "mileage": 9.0, "depth": 2.0}, source)
    rows_path = store.save_targets("L03", [target])
    rows = _read_targets_csv(rows_path)
    assert rows[0]["source_artifact_role"] == "display_compare_transform"
    assert "time_to_depth" in rows[0]["source_axis_transform"]
