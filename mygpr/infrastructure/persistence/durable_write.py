#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infrastructure adapter binding ``DurableWritePort`` to the ``core`` authority.

This is the *only* place where the application layer's durability contract
meets an actual mechanism. ``core.storage_primitives`` is the single
authoritative implementation (unique temp name + file ``fsync`` +
``os.replace`` + directory ``fsync``); delegating rather than reimplementing
is the entire point — see ``mygpr/application/persistence_ports.py`` for why
the boundary exists.

Importing ``core`` from here is legal: ``[layers.infrastructure]`` allows
``core`` as a local import prefix, and the whole reason the port sits in
``application`` is that ``core`` may not be imported from there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core import storage_primitives

from mygpr.application.persistence_ports import DurableWritePort


class CoreDurableWrite:
    """``DurableWritePort`` backed by ``core.storage_primitives``.

    Thin by design. Every method is a straight delegation: if a future
    requirement needs real logic here (retry, quota, progress), it belongs in
    ``storage_primitives`` where the other eight callers can benefit too.
    """

    def write_bytes(self, path: Path, data: bytes) -> Path:
        return storage_primitives.atomic_write_bytes(path, data)

    def write_text(self, path: Path, text: str, *, encoding: str = "utf-8") -> Path:
        return storage_primitives.atomic_write_text(path, text, encoding=encoding)

    def write_json(self, path: Path, payload: Any) -> Path:
        return storage_primitives.atomic_write_json(path, payload)


def default_durable_write() -> DurableWritePort:
    """Return the process-wide durable writer."""
    return CoreDurableWrite()


__all__ = ["CoreDurableWrite", "default_durable_write"]
