#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical project session, transaction and compatibility boundary."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from core.storage_primitives import (
    FileTransaction,
    ProjectAccessMode,
    ProjectLock,
    ProjectReadOnlyError,
    recover_file_transactions,
)


@dataclass
class ProjectSession:
    root: Path
    lock: ProjectLock
    recovery_results: tuple[dict[str, object], ...] = ()

    @property
    def read_only(self) -> bool:
        return self.lock.read_only

    @property
    def writable(self) -> bool:
        return self.lock.writable

    def assert_writable(self) -> None:
        self.lock.assert_writable()

    @contextmanager
    def transaction(self, label: str) -> Iterator[FileTransaction]:
        self.assert_writable()
        with FileTransaction(self.root, label=label) as transaction:
            yield transaction
            transaction.commit()

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
    def open_session(
        root: str | Path,
        *,
        mode: str | ProjectAccessMode = ProjectAccessMode.AUTO,
        recover_stale: bool = False,
        allow_reentrant: bool = True,
    ) -> ProjectSession:
        access = mode if isinstance(mode, ProjectAccessMode) else ProjectAccessMode(str(mode))
        resolved = Path(root).resolve()
        lock = ProjectLock(
            resolved,
            mode=access,
            recover_stale=recover_stale,
            allow_reentrant=allow_reentrant,
        ).acquire()
        recovery_results: tuple[dict[str, object], ...] = ()
        if lock.writable and not lock.reentrant:
            recovery_results = recover_file_transactions(resolved)
            failures = [item for item in recovery_results if not bool(item.get("success"))]
            if failures:
                lock.release()
                details = "; ".join(str(item.get("message") or item.get("transaction_id")) for item in failures)
                raise RuntimeError(f"项目文件事务恢复失败：{details}")
        return ProjectSession(resolved, lock, recovery_results)


__all__ = ["ProjectAccessMode", "ProjectReadOnlyError", "ProjectRepository", "ProjectSession"]
