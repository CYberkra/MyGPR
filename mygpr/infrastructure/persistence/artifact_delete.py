#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processing-artifact deletion into the project trash (user-facing 删除成果).

Design decisions (2026-09-06, user-approved):

- 删除语义 = 移入项目 ``.trash`` 回收站（与 delete_project_line 同一模式），
  不做永久删除；恢复 = 把文件手工移回 + 手工修 catalog（超出本模块范围）。
- 支持两种落盘形态，**语义统一**：
  1. sidecar（``<container>.artifacts/<artifact_id>.h5``）——直接把文件 move 进 trash；
  2. legacy 内嵌组（容器 ``/processing/artifacts/<id>``）——先把该组原样导出为
     trash 中的独立 ``<id>.h5``（组路径与 attr 逐字段保留，可直接搬回容器恢复），
     再从容器里删除该组。
  两者都进同一个 ``trash_root``，``trash_manifest.json`` 用 ``storage_mode`` 字段
  区分，恢复脚本据此决定"搬回容器"还是"搬回 sidecar 目录"。
- **零变异预检**：先判定全部目标的可删除性与形态，任一不可删则整批拒绝，
  一个字节都不动，避免出现"半删"状态。
- 并发：调用方（adapter mixin）必须持有 session RLock 后再进入本模块；
  模块自身不加锁（与 IntermediateCleanupMixin 同一约定）。
- Windows 文件占用（删除当前预览成果时 WinError 5）不在此吞掉，
  原样抛出让 backend_controller 映射为"文件被占用"提示。

变更记录：2026-09-22 —— 解禁 legacy 内嵌组删除。原实现在预检阶段一刀切拒绝
「非 sidecar 形态」，导致旧工程（v3 时代 ``save_processed_line`` 写内嵌组）的
成果永远删不掉，用户只能手工改 HDF5。实测确认内嵌组删除是安全的：容器
``raw`` 组独立保留、逐字节不变，schema 与可读性均不受影响。故改为导出+删除。
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from core.field_project_errors import FieldProjectOperationError
from core.field_project_models import validate_line_id
from core.hdf5_line_container import (
    PROCESSING_ROOT,
    artifacts_dir_path,
    locate_processing_artifact,
)
from core.processing_artifact_index import ProcessingArtifactRecord, index_processing_artifacts
from core.storage_primitives import atomic_write_json, utc_now

#: 内嵌组导出/删除时允许吞掉的 HDF5 访问异常（损坏文件交由上层报"数据文件缺失"）。
_H5_READ_ERRORS = (OSError, RuntimeError, TypeError, ValueError, KeyError)

STORAGE_MODE_SIDECAR = "sidecar"
STORAGE_MODE_INLINE = "inline"


@dataclass(frozen=True)
class _ArtifactTarget:
    """预检产物：一条待删成果的落盘形态与来源路径。"""

    artifact_id: str
    storage_mode: str  # STORAGE_MODE_SIDECAR / STORAGE_MODE_INLINE
    container: Path


def _export_inline_group(container: Path, artifact_id: str, destination: Path) -> bool:
    """把容器内嵌组原样导出为独立 h5，返回是否成功。

    组路径保持 ``/processing/artifacts/<artifact_id>`` 不变：恢复时只需
    ``dst.copy(src[group_path], dst, name=group_path)`` 搬回容器即可，
    attr（manifest_json / params_json / status）随组一并复制。
    """
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(container, "r", libver="latest") as source:
            if f"{group_path}/bscan" not in source:
                return False
            with h5py.File(destination, "w", libver="latest") as output:
                output.copy(source[group_path], output, name=group_path)
                output.flush()
    except _H5_READ_ERRORS:
        destination.unlink(missing_ok=True)
        return False
    return True


def _delete_inline_group(container: Path, artifact_id: str) -> bool:
    """删除容器内嵌组；容器零改动的场景（组不存在）返回 False。"""
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    if not container.is_file():
        return False
    try:
        with h5py.File(container, "r", libver="latest", swmr=True) as handle:
            if group_path not in handle:
                return False
    except _H5_READ_ERRORS:
        return False
    with h5py.File(container, "r+", libver="latest") as handle:
        if group_path not in handle:
            return False
        del handle[group_path]
        handle.attrs["updated_at"] = utc_now()
        handle.flush()
    return True


if TYPE_CHECKING:  # pragma: no cover - 仅类型标注，避免运行时环
    from core.field_project_store import FieldProjectStore


def collect_artifact_descendants(
    records: list[ProcessingArtifactRecord],
    root_artifact_id: str,
) -> tuple[str, ...]:
    """Return ``root_artifact_id`` plus its transitive children (blood line).

    ``records`` 应为该测线的完整成果索引（含 parent_artifact_id）。
    未知 root 直接返回空元组，由调用方决定报错语义。
    """
    children: dict[str, list[str]] = {}
    known: set[str] = set()
    for record in records:
        known.add(record.artifact_id)
        parent = str(record.parent_artifact_id or "")
        if parent:
            children.setdefault(parent, []).append(record.artifact_id)
    if root_artifact_id not in known:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    # 广度优先收集（根在前、叶在后）；提交阶段必须 reversed() 深度先删——
    # catalog.delete_artifact 把 branch head 回退到被删行的 parent，
    # 先删父会留下指向已删除父的悬空 head。
    stack = [root_artifact_id]
    while stack:
        current = stack.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        stack.extend(children.get(current, []))
    return tuple(ordered)


def _resolve_delete_closure(
    root: Path,
    line_id: str,
    artifact_ids: list[str] | tuple[str, ...],
) -> list[str]:
    """Expand the requested ids into a full descendant closure (root first).

    未知 id 直接报错（整批拒绝语义由此保证）。
    """
    records = index_processing_artifacts(root, line_id)
    known = {record.artifact_id for record in records}
    unknown = [aid for aid in artifact_ids if aid not in known]
    if unknown:
        raise FieldProjectOperationError(
            f"未找到处理成果：{', '.join(unknown[:4])}"
            + ("…" if len(unknown) > 4 else "")
        )
    full_ids: list[str] = []
    seen: set[str] = set()
    for aid in artifact_ids:
        for descendant in collect_artifact_descendants(records, aid):
            if descendant not in seen:
                seen.add(descendant)
                full_ids.append(descendant)
    return full_ids


def _classify_delete_targets(
    container: Path,
    full_ids: list[str],
) -> dict[str, "_ArtifactTarget"]:
    """Pre-flight: classify every target as sidecar / inline, zero mutation.

    只做只读探测，不产生任何副作用——任一不可删即整批拒绝，
    避免出现"半删"状态（部分成果已移入回收站、部分还在）。
    """
    sidecar_dir = artifacts_dir_path(container)
    targets: dict[str, _ArtifactTarget] = {}
    for aid in full_ids:
        if (sidecar_dir / f"{aid}.h5").is_file():
            targets[aid] = _ArtifactTarget(aid, STORAGE_MODE_SIDECAR, container)
            continue
        try:
            located, _ = locate_processing_artifact(container, aid)
        except FileNotFoundError:
            located = None
        if located == container:
            # 旧版内嵌组：容器可读且确含该组才放行（导出+删除在提交阶段做）。
            targets[aid] = _ArtifactTarget(aid, STORAGE_MODE_INLINE, container)
            continue
        raise FieldProjectOperationError(
            f"成果 {aid} 的数据文件缺失，无法删除；整批操作已取消。"
        )
    return targets


def _trash_one_artifact(
    target: "_ArtifactTarget",
    sidecar_dir: Path,
    trash_root: Path,
) -> tuple[str, str]:
    """Move (sidecar) or export-then-remove (inline) one artifact into the trash.

    返回 ``(形态, 记录进 manifest 的原位置字符串)``。
    """
    aid = target.artifact_id
    destination = trash_root / f"{aid}.h5"
    if target.storage_mode == STORAGE_MODE_SIDECAR:
        source = sidecar_dir / f"{aid}.h5"
        shutil.move(str(source), str(destination))
        return STORAGE_MODE_SIDECAR, source.as_posix()
    # 内嵌形态：先导出成回收站里的独立文件（保证可恢复），再删容器组。
    # 导出失败意味着容器里读不出该组——此时抛错中止，已处理的条目保持原样
    # （catalog 尚未删行），用户可重试；不静默丢数据。
    if not _export_inline_group(target.container, aid, destination):
        raise FieldProjectOperationError(
            f"成果 {aid} 位于旧版内嵌存储但无法导出（容器可能已损坏），"
            f"已中止；先前条目未做修改，请手工检查 {target.container}。"
        )
    _delete_inline_group(target.container, aid)
    return STORAGE_MODE_INLINE, f"{target.container.as_posix()}::{PROCESSING_ROOT}/{aid}"


def delete_artifacts_with_trash(
    store: "FieldProjectStore",
    line_id: str,
    artifact_ids: list[str] | tuple[str, ...],
    *,
    reason: str = "用户删除成果",
) -> dict[str, object]:
    """Move artifacts into the project trash (sidecar file or legacy inline group).

    预检（零变异）→ 全部通过后逐个移动/导出 + catalog 删行 → 写 trash_manifest。
    返回 dict 摘要（adapter 层再转 DTO）。
    """
    store.assert_writable()
    safe_line_id = validate_line_id(line_id)
    if not artifact_ids:
        raise FieldProjectOperationError("未选择要删除的成果")
    root = store.root.resolve()

    full_ids = _resolve_delete_closure(root, safe_line_id, artifact_ids)
    container = store.storage.line_container_path(safe_line_id)
    sidecar_dir = artifacts_dir_path(container)
    targets = _classify_delete_targets(container, full_ids)

    # 提交阶段：建 trash 目录 → 逐条移/导出 → 删 catalog 行。
    stamp = utc_now().replace(":", "").replace("+", "_")
    root_label = str(artifact_ids[0]).replace("/", "_")[:32]
    trash_root = root / ".trash" / "artifacts" / f"{stamp}_{root_label}"
    trash_root.mkdir(parents=True, exist_ok=False)
    catalog = store.storage.catalog
    moved: list[str] = []
    modes: dict[str, str] = {}
    # 深度先删：catalog.delete_artifact 把 branch head 回退到被删行的 parent，
    # 先删父会留下指向已删除父的悬空 head。
    for aid in reversed(full_ids):
        mode, origin = _trash_one_artifact(targets[aid], sidecar_dir, trash_root)
        moved.append(origin)
        modes[aid] = mode
        catalog.delete_artifact(aid)

    atomic_write_json(trash_root / "trash_manifest.json", {
        "schema": "mygpr.artifact_trash.v1",
        "line_id": safe_line_id,
        # 形态索引：sidecar 条目搬回 ``<container>.artifacts/``，
        # inline 条目用 h5py 把组 copy 回容器（组路径原样保留）。
        "storage_modes": modes,
        "artifact_ids": full_ids,
        "original_paths": moved,
        "reason": reason,
        "trashed_at": utc_now(),
        "container": container.relative_to(root).as_posix(),
        "sidecar_dir": sidecar_dir.relative_to(root).as_posix(),
    })
    store.append_log(
        f"成果移入回收站 {safe_line_id}: count={len(full_ids)}, "
        f"inline={sum(1 for m in modes.values() if m == STORAGE_MODE_INLINE)}, "
        f"reason={reason}, trash={trash_root}"
    )
    deleted = set(full_ids)
    remaining = [
        record for record in index_processing_artifacts(root, safe_line_id)
        if record.artifact_id not in deleted
    ]
    return {
        "line_id": safe_line_id,
        "deleted_artifact_ids": tuple(full_ids),
        "trash_dir": str(trash_root),
        "remaining_artifact_count": len(remaining),
        "moved_paths": tuple(moved),
    }
