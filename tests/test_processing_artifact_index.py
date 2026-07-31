from __future__ import annotations

from pathlib import Path

import numpy as np

from core.field_project_store import FieldProjectStore
from tests.field_project_test_utils import create_test_project
from core.gpr_data_model import GPRDataSet
from core.field_processing_bridge import run_registered_method
from core.processing_artifact_index import index_processing_artifacts, latest_processing_artifact
from core.storage_uri import is_h5_uri, resolve_h5_uri


def test_processing_artifact_index_reads_saved_manifest(tmp_path: Path) -> None:
    store = create_test_project(tmp_path / "project")
    dataset = GPRDataSet.synthetic("L03", rows=80, cols=48, length_m=24.0)
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
    records = index_processing_artifacts(store.root, line_id="L03")
    assert records
    record = records[0]
    assert record.line_id == "L03"
    assert record.method_id == "dewow"
    assert record.status == "success"
    assert is_h5_uri(record.data_path)
    assert record.manifest_path.endswith(".json")
    assert record.output_shape == tuple(output.matrix.shape)
    assert not record.is_display_compare_transform
    h5_path, dataset_path = resolve_h5_uri(store.root, record.data_path)
    assert h5_path.exists()
    import h5py
    with h5py.File(h5_path, "r") as handle:
        assert dataset_path in handle


def test_processing_artifact_index_marks_time_to_depth_as_display_compare(tmp_path: Path) -> None:
    store = create_test_project(tmp_path / "project")
    dataset = GPRDataSet.synthetic("L03", rows=80, cols=48, length_m=24.0)
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
    assert record.method_id == "time_to_depth"
    assert record.is_display_compare_transform
    assert record.axis_transform is not None
    assert record.axis_transform["kind"] == "time_to_depth"
