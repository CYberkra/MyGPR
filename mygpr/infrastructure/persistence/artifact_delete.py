#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processing-artifact deletion into the project trash (user-facing 删除成果).

Design decisions (2026-09-06, user-approved):

- 删除语义 = 移入项目 ``.trash`` 回收站（与 delete_project_line 同一模式），
  不做永久删除；恢复 = 把 sidecar 手工移回 + 手工修 catalog（超出本模块范围）。
- 仅支持 sidecar 落盘形态（``<container>.artifacts/<artifact_id>.h5``）。
  legacy 内嵌组（旧容器 /processing/artifacts/<id>）在预检阶段直接拒绝，
  **零变异**：整批一个都不动，避免出现"半删"状态。
- 并发：调用方（adapter mixin）必须持有 session RLock 后再进入本模块；
  模块自身不加锁（与 IntermediateCleanupMixin 同一约定）。
- Windows 文件占用（删除当前预览成果时 WinError 5）不在此吞掉，
  原样抛出让 backend_controller 映射为"文件被占用"提示。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from core.field_project_errors import FieldProjectOperationError
from core.field_project_models import validate_line_id
from core.hdf5_line_container import artifacts_dir_path, locate_processing_artifact
from core.processing_artifact_index import ProcessingArtifactRecord, index_processing_artifacts
from core.storage_primitives import atomic_write_json, utc_now


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


def delete_artifacts_with_trash(
    store: "FieldProjectStore",
    line_id: str,
    artifact_ids: list[str] | tuple[str, ...],
    *,
    reason: str = "用户删除成果",
) -> dict[str, object]:
    """Move artifacts (and only their sidecar files) into the project trash.

    预检（零变异）→ 全部通过后逐个移动 + catalog 删行 → 写 trash_manifest。
    返回 dict 摘要（adapter 层再转 DTO）。
    """
    store.assert_writable()
    safe_line_id = validate_line_id(line_id)
    if not artifact_ids:
        raise FieldProjectOperationError("未选择要删除的成果")
    root = store.root.resolve()

    records = index_processing_artifacts(root, safe_line_id)
    by_id = {record.artifact_id: record for record in records}
    unknown = [aid for aid in artifact_ids if aid not in by_id]
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

    # 预检：全部必须为 sidecar 形态（legacy 组 / 缺文件 → 整批拒绝）。
    sidecars: dict[str, Path] = {}
    for aid in full_ids:
        container = store.storage.line_container_path(safe_line_id)
        candidate = artifacts_dir_path(container) / f"{aid}.h5"
        if not candidate.is_file():
            try:
                located, _ = locate_processing_artifact(container, aid)
            except FileNotFoundError:
                located = None
            if located == container:
                raise FieldProjectOperationError(
                    f"成果 {aid} 为旧版内嵌存储（非独立文件），不支持删除；"
                    "整批操作已取消，未修改任何文件。"
                )
            raise FieldProjectOperationError(
                f"成果 {aid} 的数据文件缺失，无法删除；整批操作已取消。"
            )
        sidecars[aid] = candidate

    # 提交阶段：trash 目录 → 移文件 → 删 catalog 行。
    stamp = utc_now().replace(":", "").replace("+", "_")
    root_label = str(artifact_ids[0]).replace("/", "_")[:32]
    trash_root = root / ".trash" / "artifacts" / f"{stamp}_{root_label}"
    trash_root.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    catalog = store.storage.catalog
    for aid in reversed(full_ids):
        source = sidecars[aid]
        target = trash_root / source.name
        shutil.move(str(source), str(target))
        moved.append(source.as_posix())
        catalog.delete_artifact(aid)

    atomic_write_json(trash_root / "trash_manifest.json", {
        "schema": "mygpr.artifact_trash.v1",
        "line_id": safe_line_id,
        "artifact_ids": full_ids,
        "original_paths": moved,
        "reason": reason,
        "trashed_at": utc_now(),
    })
    store.append_log(
        f"成果移入回收站 {safe_line_id}: count={len(full_ids)}, "
        f"reason={reason}, trash={trash_root}"
    )
    remaining = [
        record for record in index_processing_artifacts(root, safe_line_id)
        if record.artifact_id not in seen
    ]
    return {
        "line_id": safe_line_id,
        "deleted_artifact_ids": tuple(full_ids),
        "trash_dir": str(trash_root),
        "remaining_artifact_count": len(remaining),
        "moved_paths": tuple(moved),
    }
