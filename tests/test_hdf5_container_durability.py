# -*- coding: utf-8 -*-
"""HDF5 容器持久性回归测试（fix/hdf5-durability）。

背景：2026-09-02 处理保存中断导致 L01/L09 容器损坏（bad layout message），
write_processing_artifact 曾对活容器 in-place r+ 写入。回归契约：
1. 处理保存走整文件原子替换，中途失败原文件完好可读；
2. 写入后读回验证失败时临时文件被丢弃、原文件不被发布；
3. 损坏容器 load_raw_dataset 抛出带可操作提示的 RuntimeError。
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from core.gpr_data_model import GPRDataSet
from core.hdf5_line_container import (
    initialize_line_container,
    load_raw_dataset,
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


def test_artifact_write_publishes_readable_container(tmp_path: Path) -> None:
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
    # 已发布容器可读且包含 artifact
    with h5py.File(container, "r", libver="latest", swmr=True) as handle:
        assert "processing/artifacts/L01_art/bscan" in handle
        assert handle["processing/artifacts/L01_art"].attrs["status"] == "committed"
    # raw 数据在整文件替换后仍然完好
    assert load_raw_dataset(container, line_id="L01").matrix.shape == (32, 24)
    assert len(container.read_bytes()) > len(before)


def test_artifact_write_failure_preserves_original_container(tmp_path: Path, monkeypatch) -> None:
    """验证阶段抛错 → 临时文件被丢弃，原容器字节不变（不被坏文件覆盖）。"""
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
    # 原文件未被替换，且仍然可读
    assert container.read_bytes() == original_bytes
    load_raw_dataset(container, line_id="L01")
    # 没有残留 .tmp 兄弟文件
    assert list(container.parent.glob("*.tmp")) == []


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
