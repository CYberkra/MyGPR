# -*- coding: utf-8 -*-
"""HDF5 容器持久性回归测试（fix/hdf5-durability）。

背景：2026-09-02 处理保存中断导致 L01/L09 容器损坏（bad layout message），
write_processing_artifact 曾对活容器 in-place r+ 写入。P3-2 起改为 per-artifact
sidecar 文件（<stem>.artifacts/<artifact_id>.h5），回归契约：
1. 处理保存走 sidecar 整文件原子替换，容器本体一个字节不动；
2. 写入后读回验证失败时临时文件被丢弃、sidecar 不被发布，容器仍然逐位不变；
3. 损坏容器 load_raw_dataset 抛出带可操作提示的 RuntimeError。
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from core.gpr_data_model import GPRDataSet
from core.hdf5_line_container import (
    artifacts_dir_path,
    initialize_line_container,
    load_processing_dataset,
    load_raw_dataset,
    locate_processing_artifact,
    write_processing_artifact,
    write_raw_dataset,
)


def _dataset(line_id: str = "L01") -> GPRDataSet:
    matrix = np.arange(32 * 24, dtype=np.float32).reshape(32, 24)
    return GPRDataSet.from_matrix(line_id, matrix, length_m=10.0, time_window_ns=250.0)


def _write_container(path: Path) -> None:
    initialize_line_container(path, project_id="p", line_id="L01")
    write_raw_dataset(path, _dataset(), project_id="p", line_id="L01")


def test_raw_write_readback_roundtrip(tmp_path: Path) -> None:
    container = tmp_path / "L01.h5"
    _write_container(container)
    ds = load_raw_dataset(container, line_id="L01")
    assert ds.matrix.shape == (32, 24)
    assert np.allclose(np.asarray(ds.distance_axis_m), _dataset().distance_axis_m)


def test_artifact_write_publishes_sidecar_and_container_byte_stable(tmp_path: Path) -> None:
    container = tmp_path / "L01.h5"
    _write_container(container)
    before = container.read_bytes()
    result = write_processing_artifact(
        container,
        artifact_id="L01_art",
        matrix=np.ones((32, 24), dtype=np.float32),
        manifest={"method_id": "noop"},
        params={},
    )
    assert result["sha256"]
    # 容器逐位不变：保存成品不触碰测线容器
    assert container.read_bytes() == before
    # sidecar 已发布且内部布局与旧容器内嵌布局一致
    sidecar = artifacts_dir_path(container) / "L01_art.h5"
    assert result["h5_file"] == str(sidecar)
    with h5py.File(sidecar, "r", libver="latest", swmr=True) as handle:
        assert "processing/artifacts/L01_art/bscan" in handle
        assert handle["processing/artifacts/L01_art"].attrs["status"] == "committed"
    assert locate_processing_artifact(container, "L01_art")[0] == sidecar
    # raw 数据仍在容器内且完好；成品矩阵经双读接口回读一致
    assert load_raw_dataset(container, line_id="L01").matrix.shape == (32, 24)
    ds = load_processing_dataset(container, artifact_id="L01_art")
    assert np.array_equal(np.asarray(ds.matrix), np.ones((32, 24), dtype=np.float32))


def test_artifact_write_failure_preserves_original_container(tmp_path: Path, monkeypatch) -> None:
    """验证阶段抛错 → sidecar 临时文件被丢弃，容器逐位不变。"""
    import core.hdf5_line_container as mod

    container = tmp_path / "L01.h5"
    _write_container(container)
    original_bytes = container.read_bytes()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("写回验证失败：模拟损坏")

    monkeypatch.setattr(mod, "_verify_written_artifact", _boom)
    with pytest.raises(RuntimeError, match="写回验证失败"):
        write_processing_artifact(
            container,
            artifact_id="L01_art",
            matrix=np.ones((32, 24), dtype=np.float32),
            manifest={"method_id": "noop"},
            params={},
        )
    # 容器逐位不变且仍然可读
    assert container.read_bytes() == original_bytes
    load_raw_dataset(container, line_id="L01")
    # 没有残留 .tmp 文件（容器兄弟目录与 sidecar 目录内都没有）
    assert list(container.parent.glob("*.tmp")) == []
    assert list(artifacts_dir_path(container).glob("*.tmp")) == []



def test_artifact_write_rejects_duplicate_id(tmp_path: Path) -> None:
    container = tmp_path / "L01.h5"
    _write_container(container)
    kwargs = dict(
        artifact_id="L01_dup",
        matrix=np.ones((32, 24), dtype=np.float32),
        manifest={"method_id": "noop"},
        params={},
    )
    write_processing_artifact(container, **kwargs)
    with pytest.raises(FileExistsError):
        write_processing_artifact(container, **kwargs)


def test_corrupted_container_load_raises_with_actionable_hint(tmp_path: Path) -> None:
    """模拟 2026-09-02 损坏：写坏容器字节 → load 报错信息含恢复指引。"""
    container = tmp_path / "L01.h5"
    container.parent.mkdir(parents=True, exist_ok=True)
    # 真 HDF5 容器 + 末尾截断（最常见的写入中断形态）
    _write_container(container)
    data = bytearray(container.read_bytes())
    container.write_bytes(bytes(data[: len(data) // 3]))
    with pytest.raises((OSError, RuntimeError, KeyError)) as excinfo:
        load_raw_dataset(container, line_id="L01")
    # 契约走全链路：GUI 的 friendly_error_message 必须把该异常映射为可操作中文提示
    # （backend CI 无 PyQt6，friendly_error_message 的 import 链需要 Qt → 跳过映射断言）
    pytest.importorskip("PyQt6")
    from ui.controllers.backend_controller import friendly_error_message
    mapped = friendly_error_message(excinfo.value)
    assert "数据文件损坏" in mapped
    assert "重新导入" in mapped


def test_data_uri_opens_sidecar_directly(tmp_path: Path) -> None:
    """backend 返回的 data_uri 中文件部分指向 sidecar，URI 直开可读到矩阵。"""
    from core.storage_uri import make_h5_uri, resolve_h5_uri

    container = tmp_path / "L01.h5"
    _write_container(container)
    result = write_processing_artifact(
        container,
        artifact_id="L01_art",
        matrix=np.ones((32, 24), dtype=np.float32),
        manifest={"method_id": "noop"},
        params={},
    )
    sidecar = artifacts_dir_path(container) / "L01_art.h5"
    uri = make_h5_uri(sidecar, result["dataset_path"])
    file_path, dataset_path = resolve_h5_uri(tmp_path, uri)
    assert file_path == sidecar
    assert dataset_path == result["dataset_path"]
    with h5py.File(file_path, "r") as handle:
        assert dataset_path in handle


def test_sidecar_and_legacy_container_group_coexist(tmp_path: Path) -> None:
    """双读并存：legacy 容器内嵌组 + 新 sidecar 同时可见；delete 双删。"""
    from core.hdf5_line_container import (
        delete_processing_artifact,
        list_processing_artifact_ids,
        locate_processing_artifact,
    )

    container = tmp_path / "L01.h5"
    _write_container(container)
    matrix = np.ones((32, 24), dtype=np.float32)
    # 手工在容器内造 legacy 内嵌组（旧版本布局，绕过新 sidecar 写路径）
    with h5py.File(container, "r+", libver="latest") as handle:
        group = handle.require_group("processing/artifacts/L01_old")
        group.create_dataset("bscan", data=matrix)
        group.attrs["status"] = "committed"
        group.attrs["manifest_json"] = "{}"
    # 新写一个 sidecar
    result = write_processing_artifact(
        container,
        artifact_id="L01_new",
        matrix=np.zeros((32, 24), dtype=np.float32),
        manifest={"method_id": "noop"},
        params={},
    )
    assert result["h5_file"] == str(artifacts_dir_path(container) / "L01_new.h5")

    ids = list_processing_artifact_ids(container)
    assert ids == ["L01_new", "L01_old"], ids
    # 双读：legacy 走容器、sidecar 走独立文件
    assert locate_processing_artifact(container, "L01_old")[0] == container
    assert locate_processing_artifact(container, "L01_new")[0] == artifacts_dir_path(container) / "L01_new.h5"
    # 双删：legacy 删容器组（r+ 打开，不触碰其他字节段）、sidecar unlink
    delete_processing_artifact(container, "L01_old")
    delete_processing_artifact(container, "L01_new")
    assert list_processing_artifact_ids(container) == []
    assert not (artifacts_dir_path(container) / "L01_new.h5").exists()
