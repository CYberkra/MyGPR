from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.chunked_gpr_io import ImportCancelled
from core.field_project_operations import import_line_data
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.field_data_quality import evaluate_line_data_quality


def test_forced_large_npy_import_uses_chunked_hdf5_and_lazy_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import core.field_line_store as line_store

    monkeypatch.setattr(line_store, "LARGE_DATASET_THRESHOLD_BYTES", 1)
    source = tmp_path / "large.npy"
    matrix = np.arange(128 * 256, dtype=np.float32).reshape(128, 256)
    np.save(source, matrix, allow_pickle=False)
    store = FieldProjectStore.create_empty(tmp_path / "project", name="large")

    line = import_line_data(store, source, line_id="L01", name="large")
    stored = store.root / line.gpr_dataset_path
    assert stored.is_file() and stored.suffix == ".h5"
    import h5py
    with h5py.File(stored, "r") as handle:
        assert handle["/raw/bscan"].chunks is not None
        assert handle["/raw/bscan"].compression == "gzip"
    loaded = store.load_gpr_dataset("L01")
    assert type(loaded.matrix).__name__ == "HDF5ArrayProxy"
    assert loaded.matrix.dtype == np.float32
    np.testing.assert_allclose(loaded.matrix[[0, -1], :8], matrix[[0, -1], :8])


def test_single_file_cancel_rolls_back_manifest_and_raw_directory(tmp_path: Path) -> None:
    source = tmp_path / "cancel.csv"
    source.write_text("\n".join(",".join(str(r * 40 + c) for c in range(40)) for r in range(600)), encoding="utf-8")
    store = FieldProjectStore.create_empty(tmp_path / "project", name="cancel")
    checks = {"count": 0}

    def cancel_requested() -> bool:
        checks["count"] += 1
        return checks["count"] >= 3

    with pytest.raises(ImportCancelled):
        import_line_data(store, source, line_id="L09", cancel_requested=cancel_requested)

    assert not store.list_lines()
    assert not (store.root / "raw" / "L09").exists()
    assert not (store.root / "data" / "lines" / "L09.h5").exists()


def test_large_quality_check_uses_bounded_representative_sample() -> None:
    matrix = np.zeros((2100, 2100), dtype=np.float32)
    dataset = GPRDataSet.from_matrix("L01", matrix, length_m=100.0)
    report = evaluate_line_data_quality(dataset, None)
    assert report.sampled is True
    assert 0 < report.evaluated_value_count <= 4_000_000
