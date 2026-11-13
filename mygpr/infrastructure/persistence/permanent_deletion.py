"""Managed paths and original-source safety for destructive project operations."""

from typing import Any


class PermanentDeletionMixin:
    _lock: Any
    _store: Any

    def delete_permanently(self) -> str:
        from core.field_project_operations import delete_project_permanently
        with self._lock:
            result = delete_project_permanently(self._store)
            return result.deleted_path

    def validate_deletion(self, kind: str, targets, line_id: str = '') -> str:
        from core.permanent_delete import checked_delete_path, protected_source_paths
        from core.field_project_operations import _assert_safe_project_root_for_removal
        from mygpr.infrastructure.persistence.artifact_delete import (
            _resolve_delete_closure, _classify_delete_targets,
        )
        with self._lock:
            self._store.assert_writable()
            root = self._store.root
            protected = protected_source_paths(root)
            if kind == 'project':
                path = _assert_safe_project_root_for_removal(self._store)
                checked_delete_path(root.parent, path)
                paths = sorted(p for p in root.rglob('*') if p.is_file() and
                         p.relative_to(root).parts[0] not in ('logs', '.delete_operations') and
                         p.name not in ('catalog.sqlite', 'catalog.sqlite-wal', 'catalog.sqlite-shm') and
                         not p.name.startswith('.mygpr.lock'))
            elif kind == 'artifacts':
                ids = _resolve_delete_closure(root, line_id, targets)
                if self._store.storage.is_hybrid:
                    container = self._store.storage.line_container_path(line_id)
                    _classify_delete_targets(container, ids, root)
                    paths = [container, self._store.storage.line_artifacts_dir(line_id)]
                else:
                    paths = [root / 'processed' / line_id]
            elif kind == 'lines':
                paths = [root / folder / target for target in targets for folder in ('raw', 'processed')]
                for target in targets:
                    paths.append(root / 'metadata' / 'markers' / f'{target}.json')
                    if self._store.storage.is_hybrid:
                        paths.extend([self._store.storage.line_container_path(target),
                                      self._store.storage.line_artifacts_dir(target)])
                    paths.extend((root / 'targets').glob(f'{target}_*'))
                    paths.extend((root / 'spatial').glob(f'{target}_*'))
            else:
                paths = [root / 'spatial' / 'results' / target for target in targets]
            states = []
            for path in paths:
                if path != root:
                    checked_delete_path(root, path)
                info = path.stat() if path.exists() else None
                states.append((str(path), info.st_size if info else None, info.st_mtime_ns if info else None))
            return repr((states, sorted(str(p) for p in protected)))
