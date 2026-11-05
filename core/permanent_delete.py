"""Direct managed-file deletion with durable progress, never recovery copies."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

from core.field_project_errors import FieldProjectOperationError
from core.storage_primitives import atomic_write_json


def protected_source_paths(root: Path) -> set[Path]:
    """Original imports are never owned deletion targets, even inside a project."""
    from core.source_file_registry import load_source_registry
    root = root.resolve()
    sources = {Path(r.source_path).expanduser().resolve() for r in load_source_registry(root) if r.source_path}
    for manifest in (root / 'raw').glob('*/import_manifest.json'):
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        if payload.get('source_path'):
            sources.add(Path(payload['source_path']).expanduser().resolve())
    retained = root / 'metadata' / 'protected_originals.json'
    if retained.is_file():
        sources.update(Path(p).resolve() for p in json.loads(retained.read_text(encoding='utf-8')))
    return sources


def remove_owned_path(path: Path, protected: set[Path]) -> None:
    """Leave original source files and their ancestor directories in place."""
    if path.resolve() in protected:
        return
    if path.is_dir():
        if any(p.is_relative_to(path.resolve()) for p in protected):
            for child in path.iterdir():
                remove_owned_path(child, protected)
            if not any(path.iterdir()):
                path.rmdir()
        else:
            shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def checked_delete_path(root: Path, path: Path) -> Path:
    """Reject escaped paths and links, including Windows junction ancestors."""
    root = root.resolve()
    path = path.absolute()
    if not path.is_relative_to(root) or path == root:
        raise FieldProjectOperationError(f"删除路径超出项目范围：{path}")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or part.is_junction():
            raise FieldProjectOperationError(f"删除范围包含目录链接：{part}")
    if not path.resolve().is_relative_to(root):
        raise FieldProjectOperationError(f"删除路径超出项目范围：{path}")
    if path.is_dir():
        # rglob does not descend through symlinks; reject junctions before
        # traversal can enter them (important on Windows).
        work = [path]
        while work:
            for child in work.pop().iterdir():
                if child.is_symlink() or child.is_junction():
                    raise FieldProjectOperationError(f"删除范围包含目录链接：{child}")
                if child.is_dir():
                    work.append(child)
    return path


def delete_journal_path(root: Path, key: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
        raise FieldProjectOperationError("删除对象编号无效")
    return checked_delete_path(root, root / ".delete_operations" / f"{key}.json")


def permanently_delete_paths(
    root: Path, paths: Iterable[Path], *, key: str, finalize: Callable[[], None],
) -> tuple[str, ...]:
    """Resume an interrupted operation; metadata completion is part of success.

    The journal contains paths and progress only. Failed operations retain it
    for an explicit retry; no file contents are copied or moved elsewhere.
    """
    root = root.resolve()
    journal = delete_journal_path(root, key)
    planned = list(dict.fromkeys(checked_delete_path(root, p) for p in paths))
    protected = protected_source_paths(root)
    # Preserve provenance when a line's registration/import manifest is removed.
    internal_sources = sorted(str(p) for p in protected if p.is_relative_to(root))
    if internal_sources:
        retained = checked_delete_path(root, root / 'metadata' / 'protected_originals.json')
        atomic_write_json(retained, internal_sources)
    state: dict[str, Any] = {"schema": "mygpr.permanent_delete.v1", "paths": [p.relative_to(root).as_posix() for p in planned],
             "completed": [], "state": "deleting"}
    if journal.is_file():
        saved = json.loads(journal.read_text(encoding="utf-8"))
        if saved.get("schema") != state["schema"]:
            raise FieldProjectOperationError(f"删除进度记录无效：{journal}")
        # Recompute ownership on retry. A shared/latest metadata path may now
        # belong to a newer result; old journal paths must never expand scope.
        state['completed'] = list(saved.get('completed') or [])
    journal.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(journal, state)
    try:
        for path in planned:
            # A previous rmtree may have removed only part of a directory.
            remove_owned_path(path, protected)
            relative = path.relative_to(root).as_posix()
            if relative not in state["completed"]:
                state["completed"].append(relative)
            atomic_write_json(journal, state)
        state["state"] = "metadata_pending"
        atomic_write_json(journal, state)
        finalize()
    except Exception as exc:
        state["state"] = "incomplete"
        state["error"] = str(exc)
        atomic_write_json(journal, state)
        remaining = [p.relative_to(root).as_posix() for p in planned if p.exists()]
        raise FieldProjectOperationError(
            f"永久删除未完成：{exc}；已完成 {len(state['completed'])} 个路径，"
            f"残留：{', '.join(remaining) or '登记待更新'}。可重试此删除；进度：{journal}"
        ) from exc
    journal.unlink()
    return tuple(str(p) for p in planned)


def delete_project_tree(root: Path) -> None:
    """Retain project identity on partial failure so a project can be retried."""
    protected = protected_source_paths(root)
    manifest = json.loads((root / 'project.json').read_text(encoding='utf-8'))
    marker_path = root / 'metadata' / 'project_root_marker.json'
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    deferred = {'project.json', 'metadata', '.delete_operations',
                'catalog.sqlite', 'catalog.sqlite-wal', 'catalog.sqlite-shm'}
    paths = [p for p in root.iterdir() if p.name not in deferred]

    def finalize():
        for name in ('catalog.sqlite-wal', 'catalog.sqlite-shm', 'catalog.sqlite', 'metadata', 'project.json'):
            remove_owned_path(root / name, protected)

    try:
        permanently_delete_paths(root, paths, key='project', finalize=finalize)
        journal_dir = root / '.delete_operations'
        if journal_dir.exists():
            remove_owned_path(journal_dir, protected)
        if not any(root.iterdir()):
            root.rmdir()
    except Exception:
        # Only metadata is reconstructed. Irreversibly removed data stays removed.
        atomic_write_json(root / 'project.json', manifest)
        atomic_write_json(marker_path, marker)
        atomic_write_json(root / 'metadata' / 'protected_originals.json', sorted(str(p) for p in protected))
        raise
