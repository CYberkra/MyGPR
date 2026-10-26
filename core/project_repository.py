#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical project session, transaction and compatibility boundary."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Iterator

from core.schema_registry import DEFAULT_SCHEMA_REGISTRY, UnsupportedSchemaError
from core.storage_primitives import (
    FileTransaction,
    ProjectAccessMode,
    ProjectLock,
    ProjectReadOnlyError,
    recover_file_transactions,
)


def _has_future_schema(path: Path, expected_family: str) -> bool:
    """Probe compatibility without migration, quarantine or recovery writes."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        family, version = DEFAULT_SCHEMA_REGISTRY.parse(payload.get("schema"))
        return family == expected_family and version > DEFAULT_SCHEMA_REGISTRY.definition(family).current_version
    except (OSError, ValueError, TypeError, UnsupportedSchemaError):
        # Interrupted current/corrupt manifests retain the normal recovery path.
        return False


@dataclass
class ProjectSession:
    root: Path
    lock: ProjectLock
    recovery_results: tuple[dict[str, object], ...] = ()
    _local: threading.local = field(default_factory=threading.local, init=False, repr=False)

    @property
    def read_only(self) -> bool:
        return self.lock.read_only

    @property
    def writable(self) -> bool:
        return self.lock.writable

    def assert_writable(self) -> None:
        self.lock.assert_writable()

    def recover_pending_file_transactions(self) -> None:
        if not self.writable or self.lock.reentrant:
            return
        self.recovery_results = recover_file_transactions(self.root)
        failures = [item for item in self.recovery_results if not bool(item.get("success"))]
        if failures:
            details = "; ".join(str(item.get("message") or item.get("transaction_id")) for item in failures)
            raise RuntimeError(f"项目文件事务恢复失败：{details}")

    @contextmanager
    def transaction(self, label: str) -> Iterator[FileTransaction]:
        self.assert_writable()
        active = getattr(self._local, 'file_transaction', None)
        if active is not None:
            yield active
            return
        with FileTransaction(self.root, label=label) as transaction:
            yield transaction
            transaction.commit()

    @contextmanager
    def use_file_transaction(self, transaction: FileTransaction) -> Iterator[None]:
        """Keep nested manifest writes in the encompassing project mutation."""
        if transaction.root != self.root.resolve():
            raise ValueError('file transaction belongs to another project')
        previous = getattr(self._local, 'file_transaction', None)
        self._local.file_transaction = transaction
        try:
            yield
        finally:
            self._local.file_transaction = previous

    def close(self) -> None:
        self.lock.release()

    def __enter__(self) -> "ProjectSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ProjectRepository:
    """Factory for the single project access/locking implementation.

    Existing ``FieldProjectStore`` and legacy ``ProjectService`` remain API
    adapters during migration, but both use this session and durable write
    primitive.  New code must depend on this repository rather than creating
    lock files or atomic-write helpers of its own.
    """

    @staticmethod
    def open_compatible_session(
        root: str | Path, *, manifest_name: str, schema_family: str,
        mode: str | ProjectAccessMode = ProjectAccessMode.AUTO, recover_stale: bool = False,
    ) -> ProjectSession:
        manifest_path = Path(root).resolve() / manifest_name
        access = ProjectAccessMode.READ_ONLY if _has_future_schema(manifest_path, schema_family) else mode
        session = ProjectRepository.open_session(root, mode=access, recover_stale=recover_stale, recover_transactions=False)
        opened = False
        try:
            # Recheck under the writer lock before any transaction recovery.
            if not session.read_only and _has_future_schema(manifest_path, schema_family):
                session.close()
                session = ProjectRepository.open_session(root, mode=ProjectAccessMode.READ_ONLY)
            session.recover_pending_file_transactions()
            opened = True
            return session
        finally:
            if not opened:
                session.close()

    @staticmethod
    def open_session(
        root: str | Path,
        *,
        mode: str | ProjectAccessMode = ProjectAccessMode.AUTO,
        recover_stale: bool = False,
        allow_reentrant: bool = True,
        recover_transactions: bool = True,
    ) -> ProjectSession:
        access = mode if isinstance(mode, ProjectAccessMode) else ProjectAccessMode(str(mode))
        resolved = Path(root).resolve()
        lock = ProjectLock(
            resolved,
            mode=access,
            recover_stale=recover_stale,
            allow_reentrant=allow_reentrant,
        ).acquire()
        session = ProjectSession(resolved, lock)
        opened = False
        try:
            if recover_transactions:
                session.recover_pending_file_transactions()
            opened = True
            return session
        finally:
            if not opened:
                session.close()


__all__ = ["ProjectAccessMode", "ProjectReadOnlyError", "ProjectRepository", "ProjectSession"]
