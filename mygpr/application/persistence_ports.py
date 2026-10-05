#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable-write boundary for application services.

Why this exists
---------------
``core.storage_primitives`` is the authoritative atomic-write implementation
(``fsync`` on the file, ``os.replace``, then ``fsync`` on the directory). The
``application`` layer may not import ``core`` — see
``config/architecture_policy.toml`` ``[layers.application]`` — so before this
module each service re-implemented the same discipline locally. By 2026-10-05
that meant **four** hand-rolled variants inside one layer:

- ``processing/workbench_service.py`` — full discipline, self-documented as a
  mirror awaiting a port;
- ``grid/service.py`` — ``mkstemp`` + replace, **no fsync at all**;
- bare ``write_text`` calls in ``autotune/evidence.py``,
  ``processing/analysis_service.py`` and ``interpretation/edit_service.py``
  with no atomic semantics at all.

The split is the real problem: a reader cannot tell which of those call sites
is durable. Services now depend on this Protocol instead; the implementation
lives in ``mygpr.infrastructure.persistence.durable_write``.

What this deliberately does NOT do
----------------------------------
It is not a filesystem abstraction and it does not pretend the application
layer knows about ``os.replace``. It states the *durability contract* — a
returned path means the bytes are on disk and will survive an abrupt process
death — and leaves the mechanism to the adapter. Keeping the surface this
narrow is what makes the adapter a thirty-line class instead of a wrapper zoo.

Scope note: this module covers the ``application`` layer only. The eight
near-duplicate ``tmp.replace(...)`` sites under ``core/`` can already import
the authority and are tracked separately.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class DurableWritePort(Protocol):
    """Atomic, fsync-backed publish of a file's contents."""

    def write_bytes(self, path: Path, data: bytes) -> Path:
        """Publish ``data`` at ``path`` atomically and durably.

        Returns the destination path. Raises on failure — a write that cannot
        be made durable must not be reported as success, because callers treat
        a returned path as "the artifact now exists on disk".
        """
        ...

    def write_text(self, path: Path, text: str, *, encoding: str = "utf-8") -> Path:
        """Text convenience wrapper over :meth:`write_bytes`."""
        ...

    def write_json(self, path: Path, payload: Any) -> Path:
        """Serialise ``payload`` as UTF-8 JSON and publish it durably."""
        ...


__all__ = ["DurableWritePort"]
