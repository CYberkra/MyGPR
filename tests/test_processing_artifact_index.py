from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.project_storage_backend import LegacyProjectStorageBackend
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


def test_processing_artifact_index_returns_newest_first(tmp_path: Path) -> None:
    """回归：GUI 消费端约定索引 0 = 最新成果。

    此前 window_mixins/processing_page 取 artifacts[-1] 而数据源按 created_at DESC
    （最新在前），导致"最新成果"实为最旧、自动预览与二次处理默认输入指错。
    本测试锁定数据源契约：index[0] 必须是最新一条。
    """
    import time

    store = create_test_project(tmp_path / "project")
    dataset = GPRDataSet.synthetic("L03", rows=80, cols=48, length_m=24.0)
    output1, manifest1 = run_registered_method(dataset, "dewow", {"window": 21})
    store.save_processed_line(
        "L03",
        output1.matrix,
        {"method": "dewow", "method_name": "去低频漂移 dewow",
         "params": {"window": 21}, "manifest": manifest1,
         "input_dataset": dataset.to_metadata()},
    )
    time.sleep(0.05)  # created_at 毫秒精度，隔开两次写入避免同刻
    output2, manifest2 = run_registered_method(dataset, "time_to_depth", {})
    store.save_processed_line(
        "L03",
        np.asarray(output2.matrix),
        {"method": "time_to_depth", "method_name": "时间-深度转换",
         "params": {}, "manifest": manifest2,
         "input_dataset": dataset.to_metadata()},
    )
    records = index_processing_artifacts(store.root, line_id="L03")
    assert len(records) == 2
    assert records[0].method_id == "time_to_depth"  # 最新在前
    assert records[1].method_id == "dewow"
    latest = latest_processing_artifact(store.root, "L03")
    assert latest is not None
    assert latest.method_id == "time_to_depth"


def test_legacy_storage_load_processing_artifact(tmp_path: Path) -> None:
    """回归：legacy 存储项目（无 SQLite catalog）成果可读回。

    此前 adapter 无条件调 storage.load_processing_artifact，legacy 后端没有该方法
    → 旧项目成果预览/二次处理 AttributeError（P0-3）。本测试锁定 npy+旁车读回契约。
    """
    store = create_test_project(tmp_path / "project")
    # 模拟旧项目：存储后端降级为 legacy（npy + 旁车），目录结构不变
    store.storage = LegacyProjectStorageBackend(store.root, store.manifest)
    dataset = GPRDataSet.synthetic("L03", rows=80, cols=48, length_m=24.0)
    output, manifest = run_registered_method(dataset, "dewow", {"window": 21})
    store.save_processed_line(
        "L03",
        output.matrix,
        {"method": "dewow", "method_name": "去低频漂移 dewow",
         "params": {"window": 21}, "manifest": manifest,
         "input_dataset": dataset.to_metadata()},
    )
    record = latest_processing_artifact(store.root, "L03")
    assert record is not None
    assert record.data_path.endswith(".npy")  # legacy npy 布局
    loaded = store.storage.load_processing_artifact("L03", record.artifact_id)
    assert loaded is not None
    assert tuple(loaded.matrix.shape) == tuple(output.matrix.shape)
    assert loaded.time_window_ns == pytest.approx(dataset.time_window_ns)
    assert loaded.dielectric_constant == pytest.approx(dataset.dielectric_constant)
    # 未知成果抛 FileNotFoundError（而非 AttributeError）
    with pytest.raises(FileNotFoundError):
        store.storage.load_processing_artifact(
            "L03", "L03_processed_doesnotexist_000000_000000")
