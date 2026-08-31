"""Shared helper for legacy compatibility facades.

The historical ``PythonModule`` package remains import-compatible while the
canonical implementations live under ``mygpr.infrastructure``.
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, MutableMapping


def reexport(namespace: MutableMapping[str, Any], module_name: str) -> ModuleType:
    """Copy all historical non-dunder exports from *module_name* into *namespace*."""
    module = import_module(module_name)
    for name, value in vars(module).items():
        if not name.startswith("__"):
            namespace.setdefault(name, value)
    return module


__all__ = ["reexport"]
