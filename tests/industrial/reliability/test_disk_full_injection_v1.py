from __future__ import annotations

import errno
from pathlib import Path

import numpy as np
import pytest

import core.hdf5_line_container as line_container
from core.field_project_models import FieldLineRecord
from core.field_project_operations import import_line_data
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet

pytestmark = [
    pytest.mark.industrial,
    pytest.mark.reliability,
    pytest.mark.requirement("REQ-STO-001", "REQ-STO-002"),
    pytest.mark.risk("RISK-DATA-LOSS", "RISK-HALF-COMMIT", "RISK-DISK-FULL"),
    pytest.mark.level("component"),
]


def _project(root: Path) -> FieldProjectStore:
    store = FieldProjectStore.create_empty(root, name="disk-full-injection")
    store.upsert_line(FieldLineRecord("L01", "Line 01"))
    matrix = np.arange(64 * 40, dtype=np.float32).reshape(64, 40)
    store.save_gpr_dataset(
        "L01",
        GPRDataSet.from_matrix("L01", matrix, length_m=20.0, time_window_ns=500.0),
    )
    return store


def _inject_enospc(monkeypatch: pytest.MonkeyPatch) -> None:
    """让所有 HDF5 矩阵写入抛 ENOSPC，模拟磁盘写满。"""

    def fail_write(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(line_container, "_write_matrix", fail_write)


def test_import_disk_full_rolls_back_manifest_and_raw_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "line.csv"
    source.write_text(
        "Number of Samples = 4,,\n"
        "Time windows (ns) = 100,,\n"
        "Number of Traces = 3,,\n"
        "Trace interval (m) = 0.1,,\n"
        + "\n".join("1,2,3,0.1,4" for _ in range(12)),
        encoding="utf-8",
    )
    store = FieldProjectStore.create_empty(tmp_path / "project", name="import-disk-full")
    _inject_enospc(monkeypatch)

    with pytest.raises(OSError, match="No space left on device"):
        import_line_data(store, source, line_id="L09")

    assert not store.list_lines()
    assert not (store.root / "raw" / "L09").exists()
    assert not (store.root / "data" / "lines" / "L09.h5").exists()
    store.close()

    reopened = FieldProjectStore.open(tmp_path / "project", access_mode="write")
    try:
        assert not store.list_lines()
        assert reopened.storage.transaction_journal.pending_paths() == ()
    finally:
        reopened.close()


def test_processing_artifact_disk_full_rolls_back_catalog_and_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    store = _project(root)
    _inject_enospc(monkeypatch)

    with pytest.raises(OSError, match="No space left on device"):
        store.save_processed_line(
            "L01",
            np.ones((64, 40), dtype=np.float32),
            {"method": "noop", "params": {}},
        )

    assert store.storage.transaction_journal.pending_paths() == ()
    assert not store.storage.catalog.list_artifacts(line_id="L01")
    store.close()

    reopened = FieldProjectStore.open(root, access_mode="write")
    try:
        assert reopened.storage.catalog.list_artifacts(line_id="L01") == []
        assert reopened.storage.transaction_journal.pending_paths() == ()
    finally:
        reopened.close()


def test_processing_artifact_disk_full_leaves_container_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _project(tmp_path / "project")
    before = np.asarray(store.load_gpr_dataset("L01").matrix, dtype=np.float32).copy()
    _inject_enospc(monkeypatch)

    with pytest.raises(OSError, match="No space left on device"):
        store.save_processed_line(
            "L01",
            np.ones((64, 40), dtype=np.float32),
            {"method": "noop", "params": {}},
        )

    # 原始数据容器必须保持可读且未被损坏
    after = store.load_gpr_dataset("L01").matrix
    np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    store.close()
