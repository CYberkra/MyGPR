from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.field_project_models import FieldLineRecord
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.field_processing_bridge import run_registered_method
from core.processing_artifact_index import index_processing_artifacts


def test_processing_manifest_contains_traceability_hashes() -> None:
    dataset = GPRDataSet.synthetic("L01", rows=80, cols=64, length_m=30.0)
    output, manifest = run_registered_method(dataset, "dewow", {})

    assert output.line_id == "L01"
    assert manifest["schema"] == "mygpr.processing_manifest.v2"
    assert manifest["line_id"] == "L01"
    assert manifest["method_id"] == "dewow"
    assert manifest["input_shape"] == [80, 64]
    assert manifest["output_shape"] == list(output.matrix.shape)
    assert len(manifest["input_data_sha256"]) == 64
    assert len(manifest["output_data_sha256"]) == 64
    assert "input_dataset" in manifest
    assert "output_dataset" in manifest
    assert isinstance(manifest["output_finite_summary"], dict)


def test_processed_artifacts_use_timestamped_params_and_exact_index_mapping(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="processing-traceability")
    store.upsert_line(FieldLineRecord("L01", "line-1"))
    dataset = GPRDataSet.synthetic("L01", rows=80, cols=64, length_m=30.0)
    store.save_gpr_dataset("L01", dataset)

    output1, manifest1 = run_registered_method(dataset, "dewow", {})
    data1, params1 = store.save_processed_line(
        "L01",
        output1.matrix,
        {
            "method": "dewow",
            "method_name": "去低频漂移 dewow",
            "params": {},
            "manifest": manifest1,
            "input_dataset": dataset.to_metadata(),
        },
    )
    output2, manifest2 = run_registered_method(dataset, "subtracting_average_2D", {"ntraces": 9})
    data2, params2 = store.save_processed_line(
        "L01",
        output2.matrix,
        {
            "method": "subtracting_average_2D",
            "method_name": "平均背景去除",
            "params": {"ntraces": 9},
            "manifest": manifest2,
            "input_dataset": dataset.to_metadata(),
        },
    )

    assert data1.exists() and data2.exists()
    assert params1.exists() and params2.exists()
    assert params1.name != "L01_params.json"
    assert params2.name != "L01_params.json"
    assert params1 != params2

    manifest_files = sorted((store.root / "processed" / "L01").glob("L01_processing_manifest_*.json"))
    assert len(manifest_files) == 2
    for path in manifest_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["save_schema"] == "mygpr.processing_save.v3"
        assert len(payload["output_data_sha256"]) == 64
        assert len(payload["params_sha256"]) == 64
        assert payload["params_path"].endswith(".json")

    records = index_processing_artifacts(store.root, line_id="L01")
    by_id = {record.artifact_id: record for record in records}
    assert by_id[data1.stem].method_id == "dewow"
    assert by_id[data2.stem].method_id == "subtracting_average_2D"
    assert by_id[data1.stem].params_path.endswith(params1.name)
    assert by_id[data2.stem].params_path.endswith(params2.name)
    assert by_id[data1.stem].output_data_sha256
    assert by_id[data2.stem].params_sha256


def test_save_processed_line_rejects_invalid_line_id(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="processing-traceability")
    with pytest.raises(ValueError):
        store.save_processed_line("../L01", np.ones((10, 10), dtype=np.float32), {"method": "dewow"})
