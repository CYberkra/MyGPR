#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Permanent deletion of saved results and their actual dependency closure."""
from __future__ import annotations

import json
from collections import deque
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
from core.storage_primitives import utc_now
from core.permanent_delete import checked_delete_path, delete_journal_path, permanently_delete_paths, protected_source_paths

#: 内嵌组导出/删除时允许吞掉的 HDF5 访问异常（损坏文件交由上层报"数据文件缺失"）。
_H5_READ_ERRORS = (OSError, RuntimeError, TypeError, ValueError, KeyError)

STORAGE_MODE_SIDECAR = "sidecar"
STORAGE_MODE_INLINE = "inline"
STORAGE_MODE_FILE = "file"


@dataclass(frozen=True)
class _ArtifactTarget:
    """预检产物：一条待删成果的落盘形态与来源路径。"""

    artifact_id: str
    storage_mode: str  # STORAGE_MODE_SIDECAR / STORAGE_MODE_INLINE
    container: Path


def _delete_inline_group(container: Path, artifact_id: str) -> bool:
    """删除容器内嵌组；容器零改动的场景（组不存在）返回 False。"""
    group_path = f"{PROCESSING_ROOT}/{artifact_id}"
    if not container.is_file():
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


def _dependency_parent(record: ProcessingArtifactRecord) -> str:
    if record.params.get("lineage_version") == 2:
        return str(record.params.get("step_input_artifact_id") or "")
    return str(record.parent_artifact_id or "")


def collect_artifact_descendants(
    records: list[ProcessingArtifactRecord],
    root_artifact_id: str,
) -> tuple[str, ...]:
    """Return ``root_artifact_id`` plus its transitive children (blood line).

    v2 records use their explicit saved-input edge. Historical records retain
    the conservative run-input dependency; their step order is never inferred.
    ``records`` 应为该测线的完整成果索引（含 parent_artifact_id）。
    未知 root 直接返回空元组，由调用方决定报错语义。
    """
    children: dict[str, list[str]] = {}
    known: set[str] = set()
    for record in records:
        known.add(record.artifact_id)
        parent = _dependency_parent(record)
        if parent:
            children.setdefault(parent, []).append(record.artifact_id)
    if root_artifact_id not in known:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    # 广度优先收集（根在前、叶在后）；提交阶段必须 reversed() 深度先删——
    # catalog.delete_artifact 把 branch head 回退到被删行的 parent，
    # 先删父会留下指向已删除父的悬空 head。
    stack = deque([root_artifact_id])
    while stack:
        current = stack.popleft()
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
    # Overlapping user selections may list a leaf before its ancestor. Order
    # the union by dependencies before reversed() commits any file/catalog move.
    parents = {record.artifact_id: _dependency_parent(record) for record in records}
    children: dict[str, list[str]] = {}
    queue: deque[str] = deque()
    for aid in full_ids:
        parent = parents[aid]
        if parent in seen:
            children.setdefault(parent, []).append(aid)
        else:
            queue.append(aid)
    ordered: list[str] = []
    while queue:
        aid = queue.popleft()
        ordered.append(aid)
        queue.extend(children.get(aid, []))
    if len(ordered) != len(full_ids):
        raise FieldProjectOperationError("处理成果来源存在循环，已取消删除，请先核对项目谱系。")
    return ordered


def _classify_delete_targets(
    container: Path,
    full_ids: list[str],
    root: Path | None = None,
) -> dict[str, "_ArtifactTarget"]:
    """Pre-flight: classify every target as sidecar / inline, zero mutation.

    只做只读探测；预检失败时不开始删除。
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
            # 容器可读且确含该组才放行。
            targets[aid] = _ArtifactTarget(aid, STORAGE_MODE_INLINE, container)
            continue
        # A prior unlink may have completed before its catalog update failed.
        if root is not None and delete_journal_path(root, f"artifact-{aid}").is_file():
            targets[aid] = _ArtifactTarget(aid, STORAGE_MODE_SIDECAR, container)
            continue
        raise FieldProjectOperationError(
            f"成果 {aid} 的数据文件缺失，无法删除；整批操作已取消。"
        )
    return targets


def _validate_artifact_metadata_paths(root: Path, line_id: str, full_ids: list[str]) -> None:
    for record in index_processing_artifacts(root, line_id):
        if record.artifact_id not in full_ids:
            continue
        for value in (record.params_path, record.manifest_path):
            if value:
                path = checked_delete_path(root, root / value)
                if not path.is_relative_to(root / 'processed' / line_id):
                    raise FieldProjectOperationError(f'成果元数据不属于当前测线：{path}')


def delete_artifacts_permanently(
    store: "FieldProjectStore", line_id: str, artifact_ids: list[str] | tuple[str, ...],
    *, reason: str = "用户删除成果",
) -> dict[str, object]:
    """Delete leaves first, updating catalog after each durable file deletion."""
    store.assert_writable()
    safe_line_id = validate_line_id(line_id)
    if not artifact_ids:
        raise FieldProjectOperationError("未选择要删除的成果")
    root = store.root.resolve()
    full_ids = _resolve_delete_closure(root, safe_line_id, artifact_ids)
    hybrid = store.storage.is_hybrid
    container = checked_delete_path(root, root / 'data' / 'lines' / f'{safe_line_id}.h5')
    sidecar_dir = checked_delete_path(root, artifacts_dir_path(container))
    if hybrid:
        targets = _classify_delete_targets(container, full_ids, root)
    else:
        records_by_id = {r.artifact_id: r for r in index_processing_artifacts(root, safe_line_id)}
        targets = {aid: _ArtifactTarget(aid, STORAGE_MODE_FILE,
                   checked_delete_path(root, root / records_by_id[aid].data_path)) for aid in full_ids}
    if any(t.storage_mode == STORAGE_MODE_INLINE for t in targets.values()):
        for source in protected_source_paths(root):
            if source.exists() and container.samefile(source):
                raise FieldProjectOperationError('成果容器也是原始导入源文件，已拒绝修改源数据。')
    # Check every id/path before the first irreversible operation.
    for aid in full_ids:
        delete_journal_path(root, f"artifact-{aid}")
        checked_delete_path(root, sidecar_dir / f"{aid}.h5")
    _validate_artifact_metadata_paths(root, safe_line_id, full_ids)
    deleted: list[str] = []
    paths: list[str] = []
    try:
        for aid in reversed(full_ids):
            target = targets[aid]
            records = index_processing_artifacts(root, safe_line_id)
            record = next(r for r in records if r.artifact_id == aid)
            shared = {value for r in records if r.artifact_id != aid
                      for value in (r.params_path, r.manifest_path) if value}
            metadata = [root / value for value in (record.params_path, record.manifest_path)
                        if value and value not in shared]
            metadata.append(root / 'processed' / safe_line_id / f'{aid}.artifact')
            latest = root / 'processed' / safe_line_id / f'{safe_line_id}_params.json'
            if latest.is_file() and json.loads(latest.read_text(encoding='utf-8')).get('artifact_id') == aid:
                metadata.append(latest)
            def finalize(aid=aid, target=target, record=record, records=records):
                if target.storage_mode == STORAGE_MODE_INLINE:
                    _delete_inline_group(container, aid)
                with store.mutation('delete-result-registration'):
                    if hybrid:
                        store.storage.catalog.delete_artifact(aid)
                    line = store.get_line(safe_line_id)
                    pointer = f'processed/{safe_line_id}/{aid}.artifact'
                    if line.processed_result in (pointer, record.data_path):
                        remaining = [r for r in records if r.artifact_id != aid]
                        newest = max(remaining, key=lambda r: r.created_at, default=None)
                        line.processed_result = ((f'processed/{safe_line_id}/{newest.artifact_id}.artifact' if hybrid else newest.data_path) if newest else '')
                        line.params_path = newest.params_path if newest else ''
                        line.processing_status = '已完成' if newest else '已导入'
                        store.upsert_line(line)
            files = [sidecar_dir / f"{aid}.h5"] if target.storage_mode == STORAGE_MODE_SIDECAR else []
            if target.storage_mode == STORAGE_MODE_FILE:
                files.append(target.container)
            files.extend(metadata)
            paths.extend(permanently_delete_paths(root, files, key=f"artifact-{aid}", finalize=finalize))
            deleted.append(aid)
    except Exception as exc:
        raise FieldProjectOperationError(
            f"已永久删除 {len(deleted)} 项成果，当前项未完成，其余 {len(full_ids)-len(deleted)-1} 项未执行。{exc}"
        ) from exc
    store.append_log(f"成果永久删除 {safe_line_id}: count={len(deleted)}, reason={reason}")
    return {"line_id": safe_line_id, "deleted_artifact_ids": tuple(deleted),
            "trash_dir": "", "remaining_artifact_count": len(index_processing_artifacts(root, safe_line_id)),
            "moved_paths": tuple(paths)}


# Compatibility for callers importing the former name: semantics are permanent.
delete_artifacts_with_trash = delete_artifacts_permanently
