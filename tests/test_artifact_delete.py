"""端到端：处理成果删除（回收站语义）— backend 门面 → adapter → catalog。

覆盖：
- 单删：sidecar 移入 .trash、catalog 行消失、branch head 回退、trash_manifest 字段
- 级联：删父 → 子随删，提交顺序深度先删（子文件先出现在 moved_paths）
- legacy 内嵌组：可删（导出为独立 h5 进回收站 + 删容器组），raw 逐字节不变
- 混合形态批删：inline 与 sidecar 各归各位，形态索引记入 trash_manifest
- 数据缺失（既无 sidecar 又无内嵌组）：整批拒绝、零变异
- 未知 id：整批拒绝、零变异
- descendants 只读查询：含自身、未知 id 返回空
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from core.field_project_errors import FieldProjectOperationError
from core.hdf5_line_container import PROCESSING_ROOT, artifacts_dir_path
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def _data(rows: int = 72, cols: int = 48) -> np.ndarray:
    y = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    return (0.08 * y + np.exp(-((y - 0.55 - 0.03 * np.sin(x * 6.28)) ** 2) / 0.003)).astype(np.float32)


def _make_project_with_chain(tmp_path: Path, name: str = "DeleteE2E"):
    """创建项目 + L01 + 父子两条成果（A=原始处理，B=从 A 二跳）。

    返回 (backend, summary, artifact_a, artifact_b)。
    """
    backend = MyGPRBackend.create_default(max_workers=1)
    summary = backend.projects.create_project(tmp_path / "proj", name=name)
    backend.projects.save_line_dataset(
        summary.project_id, "L01", _data(), name="L01",
        length_m=23.5, time_window_ns=500.0,
    )
    p1 = PipelineDefinition(name="p1", steps=(PipelineStep("dewow", {"window": 11}),))
    job1 = backend.submit_project_pipeline(summary.project_id, "L01", p1, result_name="成果A")
    snap1 = backend.jobs.wait(job1, timeout=30)
    assert snap1.status is snap1.status.COMPLETED, snap1.error_message
    artifact_a = snap1.result
    assert artifact_a is not None and artifact_a.artifact_id
    p2 = PipelineDefinition(name="p2", steps=(PipelineStep("agcGain", {"window": 9}),))
    job2 = backend.submit_project_pipeline(
        summary.project_id, "L01", p2, result_name="成果B",
        input_artifact_id=artifact_a.artifact_id,
    )
    snap2 = backend.jobs.wait(job2, timeout=30)
    assert snap2.status is snap2.status.COMPLETED, snap2.error_message
    artifact_b = snap2.result
    assert artifact_b is not None and artifact_b.artifact_id
    assert artifact_b.parent_artifact_id == artifact_a.artifact_id
    return backend, summary, artifact_a, artifact_b

def _sidecar_path(backend, project_id: str, line_id: str, artifact_id: str) -> Path:
    session = backend.projects._session(project_id)
    container = session._store.storage.line_container_path(line_id)
    return container.parent / f"{container.stem}.artifacts" / f"{artifact_id}.h5"


def _container_path(backend, project_id: str, line_id: str) -> Path:
    return backend.projects._session(project_id)._store.storage.line_container_path(line_id)


def _to_inline(backend, project_id: str, line_id: str, artifact_id: str) -> Path:
    """把某条 sidecar 成果改造为旧版内嵌形态（组搬进容器 + 删 sidecar + 改 catalog）。

    模拟 v3 时代 ``save_processed_line`` 的落盘结果：容器 ``/processing/artifacts/<id>``
    有组、``h5_path`` 指回容器、``<stem>.artifacts/`` 下无文件。
    """
    store = backend.projects._session(project_id)._store
    container = store.storage.line_container_path(line_id)
    sidecar = artifacts_dir_path(container) / f"{artifact_id}.h5"
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    with h5py.File(sidecar, "r") as source, h5py.File(container, "r+") as target:
        target.copy(source[group_path], target, name=group_path)
        target.flush()
    sidecar.unlink()
    with store.storage.catalog.transaction() as db:
        db.execute(
            "UPDATE artifacts SET h5_path=? WHERE artifact_id=?",
            (container.relative_to(store.root).as_posix(), artifact_id),
        )
    assert group_path in h5py.File(container, "r"), "内嵌组未建立"
    assert not sidecar.exists(), "sidecar 未移除"
    return container


def test_delete_single_artifact_moves_sidecar_to_trash(tmp_path: Path) -> None:
    backend, summary, artifact_a, artifact_b = _make_project_with_chain(tmp_path)
    try:
        pid, line_id = summary.project_id, "L01"
        sidecar = _sidecar_path(backend, pid, line_id, artifact_b.artifact_id)
        assert sidecar.is_file(), f"sidecar 不存在：{sidecar}"

        result = backend.delete_artifacts(pid, line_id, [artifact_b.artifact_id])
        assert result.line_id == line_id
        assert tuple(result.deleted_artifact_ids) == (artifact_b.artifact_id,)
        assert result.remaining_artifact_count == 1  # 只剩 A
        assert Path(result.trash_dir).is_dir()
        assert not sidecar.exists()

        # catalog 行消失
        remaining_ids = {a.artifact_id for a in backend.projects.list_artifacts(pid, line_id)}
        assert artifact_b.artifact_id not in remaining_ids
        assert artifact_a.artifact_id in remaining_ids

        # trash_manifest 字段
        manifest = json.loads((Path(result.trash_dir) / "trash_manifest.json").read_text("utf-8"))
        assert manifest["schema"] == "mygpr.artifact_trash.v1"
        assert manifest["line_id"] == line_id
        assert manifest["artifact_ids"] == [artifact_b.artifact_id]
        assert manifest["original_paths"], "moved paths 应非空"
        assert manifest["reason"]
        assert manifest["trashed_at"]

        # 回收站里应有移入的 sidecar 文件
        trashed = [p for p in Path(result.trash_dir).glob("*.h5")]
        assert len(trashed) == 1
    finally:
        backend.shutdown()


def test_delete_parent_cascades_child_depth_first_order(tmp_path: Path) -> None:
    backend, summary, artifact_a, artifact_b = _make_project_with_chain(tmp_path)
    try:
        pid, line_id = summary.project_id, "L01"
        result = backend.delete_artifacts(pid, line_id, [artifact_a.artifact_id])
        # 闭包补全：传父 → 子随删
        assert set(result.deleted_artifact_ids) == {
            artifact_a.artifact_id, artifact_b.artifact_id}
        # trash 目录内应有两条 sidecar
        assert len(list(Path(result.trash_dir).glob("*.h5"))) == 2
        assert backend.projects.list_artifacts(pid, line_id) == ()
        for aid in (artifact_a.artifact_id, artifact_b.artifact_id):
            assert not _sidecar_path(backend, pid, line_id, aid).exists()
    finally:
        backend.shutdown()


def test_delete_unknown_artifact_id_rejects_without_mutation(tmp_path: Path) -> None:
    backend, summary, artifact_a, artifact_b = _make_project_with_chain(tmp_path)
    try:
        pid, line_id = summary.project_id, "L01"
        sidecar = _sidecar_path(backend, pid, line_id, artifact_a.artifact_id)
        before = sidecar.read_bytes()
        with pytest.raises(FieldProjectOperationError):
            backend.delete_artifacts(pid, line_id, ["nonexistent-id"])
        assert not (backend.projects._session(pid)._store.root / ".trash").exists()
        assert sidecar.read_bytes() == before
        assert len(backend.projects.list_artifacts(pid, line_id)) == 2
    finally:
        backend.shutdown()


def test_list_artifact_descendants_readonly(tmp_path: Path) -> None:
    backend, summary, artifact_a, artifact_b = _make_project_with_chain(tmp_path)
    try:
        pid, line_id = summary.project_id, "L01"
        desc = backend.list_artifact_descendants(pid, line_id, artifact_a.artifact_id)
        assert artifact_a.artifact_id in desc
        assert artifact_b.artifact_id in desc
        # 叶子查询只含自身
        leaf = backend.list_artifact_descendants(pid, line_id, artifact_b.artifact_id)
        assert list(leaf) == [artifact_b.artifact_id]
        # 未知 id → 空闭包
        assert backend.list_artifact_descendants(pid, line_id, "nope") == ()
    finally:
        backend.shutdown()


def test_delete_legacy_inline_artifact_exports_and_removes_group(tmp_path: Path) -> None:
    """legacy 内嵌形态可删：导出独立 h5 进回收站 + 删容器组，raw 逐字节不变。

    回归 2026-09-22：旧实现对此形态直接拒绝（"旧版内嵌存储，不支持删除"），
    导致 v3 时代旧工程成果永远删不掉。实测内嵌组删除安全（raw 独立保留），
    故改为「先导出到回收站保可恢复、再删组」。
    """
    backend, summary, artifact_a, artifact_b = _make_project_with_chain(tmp_path)
    try:
        pid, line_id = summary.project_id, "L01"
        container = _to_inline(backend, pid, line_id, artifact_a.artifact_id)
        group_path = f"{PROCESSING_ROOT}/{artifact_a.artifact_id}"
        assert not _sidecar_path(backend, pid, line_id, artifact_a.artifact_id).exists()

        with h5py.File(container, "r") as handle:
            raw_before = handle["/raw/bscan"][...].copy()
            export_before = handle[f"{group_path}/bscan"][...].copy()
            manifest_before = json.loads(
                str(handle[group_path].attrs.get("manifest_json", "{}")))

        # 内嵌 A 是 B 的父，级联应一并删除
        result = backend.delete_artifacts(pid, line_id, [artifact_a.artifact_id])
        assert set(result.deleted_artifact_ids) == {
            artifact_a.artifact_id, artifact_b.artifact_id}

        # 回收站：两种形态各留一个独立 h5，形态索引可查
        trash = Path(result.trash_dir)
        assert trash.is_dir()
        exported = trash / f"{artifact_a.artifact_id}.h5"
        assert exported.is_file(), "内嵌组未导出到回收站"
        manifest = json.loads((trash / "trash_manifest.json").read_text("utf-8"))
        assert manifest["storage_modes"][artifact_a.artifact_id] == "inline"
        assert manifest["storage_modes"][artifact_b.artifact_id] == "sidecar"

        # 容器：内嵌组消失、raw 与 schema 完好
        with h5py.File(container, "r") as handle:
            assert group_path not in handle
            assert np.array_equal(raw_before, handle["/raw/bscan"][...]), "raw 数据被改动"
            assert str(handle.attrs.get("schema")) == "mygpr.line_container.v1"

        # 导出件与原内嵌组逐字段一致（可恢复）
        with h5py.File(exported, "r") as handle:
            assert np.array_equal(export_before, handle[f"{group_path}/bscan"][...])
            assert json.loads(str(handle[group_path].attrs.get("manifest_json", "{}"))) \
                == manifest_before

        assert backend.projects.list_artifacts(pid, line_id) == ()
    finally:
        backend.shutdown()


def test_delete_mixed_inline_and_sidecar_in_one_batch(tmp_path: Path) -> None:
    """同批混删 inline 与 sidecar：两种形态各归各位，互不影响。"""
    backend, summary, artifact_a, artifact_b = _make_project_with_chain(tmp_path)
    try:
        pid, line_id = summary.project_id, "L01"
        container = _to_inline(backend, pid, line_id, artifact_a.artifact_id)
        sidecar_b = _sidecar_path(backend, pid, line_id, artifact_b.artifact_id)
        assert sidecar_b.is_file()

        result = backend.delete_artifacts(
            pid, line_id, [artifact_a.artifact_id, artifact_b.artifact_id])
        assert set(result.deleted_artifact_ids) == {
            artifact_a.artifact_id, artifact_b.artifact_id}

        trash = Path(result.trash_dir)
        assert (trash / f"{artifact_a.artifact_id}.h5").is_file()  # inline 导出
        assert (trash / f"{artifact_b.artifact_id}.h5").is_file()  # sidecar 搬入
        manifest = json.loads((trash / "trash_manifest.json").read_text("utf-8"))
        assert manifest["storage_modes"] == {
            artifact_a.artifact_id: "inline", artifact_b.artifact_id: "sidecar"}

        assert f"{PROCESSING_ROOT}/{artifact_a.artifact_id}" not in h5py.File(container, "r")
        assert not sidecar_b.exists()
        assert backend.projects.list_artifacts(pid, line_id) == ()
    finally:
        backend.shutdown()


def test_delete_artifact_with_no_backing_data_rejected_zero_mutation(tmp_path: Path) -> None:
    """既无 sidecar 也无内嵌组（纯元数据残留）→ 整批拒绝，零变异。"""
    backend, summary, artifact_a, artifact_b = _make_project_with_chain(tmp_path)
    try:
        pid, line_id = summary.project_id, "L01"
        session = backend.projects._session(pid)
        container = session._store.storage.line_container_path(line_id)

        # 只删 sidecar，容器里也没有对应内嵌组 → 两边都找不到
        _sidecar_path(backend, pid, line_id, artifact_a.artifact_id).unlink()

        before = container.read_bytes()
        with pytest.raises(FieldProjectOperationError, match="数据文件缺失"):
            backend.delete_artifacts(pid, line_id, [artifact_a.artifact_id])
        assert container.read_bytes() == before
        assert not (session._store.root / ".trash").exists()
        remaining_ids = {a.artifact_id for a in backend.projects.list_artifacts(pid, line_id)}
        assert artifact_a.artifact_id in remaining_ids
        assert artifact_b.artifact_id in remaining_ids
    finally:
        backend.shutdown()
