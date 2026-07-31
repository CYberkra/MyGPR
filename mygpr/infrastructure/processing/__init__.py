"""Concrete processing adapters with cycle-safe lazy exports.

The infrastructure package must remain importable from historical algorithm
facades without eagerly importing the legacy registry back into itself.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FileBackedBlockPipelineExecutor": ("mygpr.infrastructure.processing.block_executor", "FileBackedBlockPipelineExecutor"),
    "LegacyProcessingCatalog": ("mygpr.infrastructure.processing.legacy_adapter", "LegacyProcessingCatalog"),
    "LegacyProcessingExecutor": ("mygpr.infrastructure.processing.legacy_adapter", "LegacyProcessingExecutor"),
    "CompositeProcessingCatalog": ("mygpr.infrastructure.processing.native_adapter", "CompositeProcessingCatalog"),
    "CompositeProcessingExecutor": ("mygpr.infrastructure.processing.native_adapter", "CompositeProcessingExecutor"),
    "NativeProcessingCatalog": ("mygpr.infrastructure.processing.native_adapter", "NativeProcessingCatalog"),
    "NativeProcessingExecutor": ("mygpr.infrastructure.processing.native_adapter", "NativeProcessingExecutor"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
