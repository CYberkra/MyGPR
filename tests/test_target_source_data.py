from __future__ import annotations

from pathlib import Path

import numpy as np

from core.field_processing_bridge import run_registered_method
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.processing_artifact_index import latest_processing_artifact
from core.target_source_binding import artifact_target_source, raw_target_source
from core.target_source_data import resolve_target_source_view


def test_raw_target_source_view_uses_raw_matrix_and_time_axis(tmp_path: Path) -> None:
    dataset = GPRDataSet.synthetic("L03", rows=48, cols=36, length_m=18.0)
    source = raw_target_source(dataset, line_id="L03")
    view = resolve_target_source_view(project_root=tmp_path, source=source, raw_dataset=dataset)
    assert view.matrix.shape == dataset.matrix.shape
    assert view.vertical_axis_label == "时间 (ns)"
    assert np.allclose(view.distance_axis_m, dataset.distance_axis_m)


def test_processed_target_source_view_loads_saved_artifact_matrix(tmp_path: Path) -> None:
    store = FieldProjectStore.create(tmp_path / "project")
    dataset = GPRDataSet.synthetic("L03", rows=64, cols=40, length_m=20.0)
    output, manifest = run_registered_method(dataset, "dewow", {"window": 21})
    store.save_processed_line(
        "L03",
        output.matrix,
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
    view = resolve_target_source_view(project_root=store.root, source=source, raw_dataset=dataset)
    assert view.matrix.shape == output.matrix.shape
    assert view.vertical_axis_label == "时间 (ns)"
    assert source.source_mode == "processed"
    assert not np.allclose(view.matrix, dataset.matrix)


def test_time_to_depth_source_view_uses_depth_axis_contract(tmp_path: Path) -> None:
    store = FieldProjectStore.create(tmp_path / "project")
    dataset = GPRDataSet.synthetic("L03", rows=64, cols=40, length_m=20.0)
    output, manifest = run_registered_method(dataset, "time_to_depth", {})
    store.save_processed_line(
        "L03",
        output.matrix,
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
    view = resolve_target_source_view(project_root=store.root, source=source, raw_dataset=dataset)
    assert source.source_mode == "display_compare"
    assert view.vertical_axis_label == "深度 (m)"
    assert view.uses_depth_axis
    assert "坐标轴转换" in view.source_note


def test_processed_artifact_save_paths_are_unique_within_one_second(tmp_path: Path) -> None:
    store = FieldProjectStore.create(tmp_path / "project")
    dataset = GPRDataSet.synthetic("L03", rows=24, cols=20, length_m=10.0)
    first, _ = store.save_processed_line("L03", dataset.matrix, {"method": "raw_copy", "manifest": {"method_id": "raw_copy"}})
    second, _ = store.save_processed_line("L03", dataset.matrix * 0.5, {"method": "raw_copy", "manifest": {"method_id": "raw_copy"}})
    assert first != second
    assert first.exists()
    assert second.exists()
