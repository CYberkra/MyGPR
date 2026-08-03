#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public processing-method registry compatibility facade.

The large declarative tables are split by responsibility under
``core.method_registry_groups`` and ``core.method_registry_metadata``.  This
module preserves the historical public API used by the current Qt frontend,
CLI, tests, and compatibility adapters.

Since v0.9.37, the **single source of truth** is ``NATIVE_ALGORITHMS`` in
``mygpr.infrastructure.processing.algorithms.methods``.  This module
projects it into the legacy ``PROCESSING_METHODS`` dict for backward
compatibility, overlaying UI metadata from ``METHOD_METADATA``.
"""
from __future__ import annotations

from typing import Any

from core.algorithm_specs import AlgorithmCatalog
from core.method_registry_bindings import HAS_PYWAVELETS  # noqa: F401 — re-exported for tests
from core.method_registry_bindings import *  # noqa: F401,F403 — re-export legacy method_* callables for cli_batch and processing_engine
from core.method_registry_metadata import (
    AUTO_TUNE_STAGE_BY_METHOD,
    METHOD_CATEGORY_LABELS,
    METHOD_DISPLAY_NAMES,
    METHOD_METADATA,
    METHOD_TAGS,
    PREFERRED_METHOD_ORDER,
)
from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS


def _legacy_adapter(native_func: Any) -> Any:
    """Wrap a native ``(data, params)`` callable into ``(data, **kwargs)`` for the legacy engine."""
    def wrapper(data: Any, **kwargs: Any) -> Any:
        return native_func(data, kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Build PROCESSING_METHODS from NATIVE_ALGORITHMS (single source of truth)
# with UI metadata overlays from METHOD_METADATA.
#
# ``func`` is intentionally sourced from the legacy method_registry_bindings
# (imported via wildcard above) so that ProcessingEngine keeps its historical
# behavior.  The native callable is preserved under ``native_func`` for
# NativeProcessingExecutor to use.
# ---------------------------------------------------------------------------
PROCESSING_METHODS: dict[str, dict[str, Any]] = {}

# Build a lookup of legacy method_* callables by method_id.
_legacy_callables: dict[str, Any] = {}
for _name, _value in list(globals().items()):
    if _name.startswith("method_") and callable(_value):
        _legacy_callables[_name.replace("method_", "", 1)] = _value

# Methods whose legacy implementation is hardcoded in ProcessingEngine._run_legacy_adapter.
# For these, ``func`` must remain non-callable so the engine dispatches to the
# historical implementation rather than the native callable.
_LEGACY_ADAPTER_METHODS = frozenset({
    "compensatingGain",
    "agcGain",
    "subtracting_average_2D",
    "running_average_2D",
})

for method_id, algorithm in NATIVE_ALGORITHMS.items():
    meta = METHOD_METADATA.get(method_id, {})

    # Convert parameter_schema from dict to legacy list-of-dicts format.
    param_schema = algorithm.parameter_schema or {}
    params_list: list[dict[str, Any]] = []
    for key, value in param_schema.items():
        if isinstance(value, dict):
            entry = dict(value)
            entry.setdefault("name", key)
            params_list.append(entry)
        else:
            params_list.append({"name": str(key)})

    # Derive module name from the callable's __module__.
    if method_id in _LEGACY_ADAPTER_METHODS:
        func = method_id  # string sentinel → ProcessingEngine falls back to _run_legacy_adapter
    else:
        func = _legacy_callables.get(method_id, _legacy_adapter(algorithm.function))
    if hasattr(func, "__module__") and func.__module__:
        module_name = func.__module__.split(".")[-1]
    else:
        module_name = method_id

    PROCESSING_METHODS[method_id] = {
        "name": str(meta.get("display_name") or algorithm.name or method_id),
        "type": "native",
        "module": module_name,
        "func": func,
        "native_func": algorithm.function,
        "params": params_list,
        "auto_tune_enabled": bool(algorithm.auto_tune_family),
        "auto_tune_family": str(algorithm.auto_tune_family or ""),
        "auto_tune_stage": str(algorithm.auto_tune_stage or algorithm.auto_tune_family or ""),
        "category": str(meta.get("category") or algorithm.category or "experimental"),
        "maturity": str(meta.get("maturity") or "experimental"),
        "visibility": str(meta.get("visibility") or "public"),
        "implementation_version": str(algorithm.implementation_version or "native-1.0"),
        "description": str(meta.get("description") or ""),
    }

# Overlay AUTO_TUNE_STAGE_BY_METHOD (some methods have stage overrides).
for method_id, stage in AUTO_TUNE_STAGE_BY_METHOD.items():
    if method_id in PROCESSING_METHODS:
        PROCESSING_METHODS[method_id]["auto_tune_stage"] = stage

# Build AlgorithmCatalog from the unified PROCESSING_METHODS.
ALGORITHM_CATALOG = AlgorithmCatalog.from_legacy(PROCESSING_METHODS)


def get_algorithm_spec(method_key: str):
    return ALGORITHM_CATALOG.get(method_key)


def get_field_approved_method_keys() -> list[str]:
    return [spec.algorithm_id for spec in ALGORITHM_CATALOG.production()]


def get_research_method_keys() -> list[str]:
    return [spec.algorithm_id for spec in ALGORITHM_CATALOG.research()]


def is_public_method(method_key: str) -> bool:
    """Whether a method should appear in the public GUI lists."""
    method = PROCESSING_METHODS.get(method_key, {})
    if not method or str(method_key).startswith("_"):
        return False
    return str(method.get("visibility", "public")) == "public"


def get_method_display_name(method_key: str) -> str:
    """Return unified user-facing method name."""
    method = PROCESSING_METHODS.get(method_key, {})
    return str(
        METHOD_DISPLAY_NAMES.get(method_key)
        or method.get("name")
        or method.get("display_name")
        or method_key
    )


def get_method_category(method_key: str) -> str:
    """Return internal category key for a method."""
    method = PROCESSING_METHODS.get(method_key, {})
    return str(method.get("category", "experimental"))


def get_auto_tune_stage(method_key: str) -> str:
    """Return stage-level auto-tune grouping for a method."""
    method = PROCESSING_METHODS.get(method_key, {})
    return str(
        method.get("auto_tune_stage")
        or AUTO_TUNE_STAGE_BY_METHOD.get(method_key)
        or method.get("auto_tune_family")
        or ""
    )


def get_method_category_label(method_key: str) -> str:
    """Return user-facing category label for a method."""
    category = get_method_category(method_key)
    return str(METHOD_CATEGORY_LABELS.get(category, category))


def get_public_method_keys() -> list[str]:
    """Return public method keys in preferred display order."""
    ordered = [key for key in PREFERRED_METHOD_ORDER if is_public_method(key)]
    tail = [
        key
        for key in PROCESSING_METHODS.keys()
        if key not in ordered and is_public_method(key)
    ]
    return ordered + tail


def get_public_methods_grouped_by_category() -> list[tuple[str, list[str]]]:
    """Return public methods grouped by category while preserving preferred order."""
    grouped: dict[str, list[str]] = {}
    for key in get_public_method_keys():
        category = get_method_category(key)
        grouped.setdefault(category, []).append(key)

    ordered_categories = []
    for key in get_public_method_keys():
        category = get_method_category(key)
        if category not in ordered_categories:
            ordered_categories.append(category)

    return [(category, grouped.get(category, [])) for category in ordered_categories]


__all__ = [
    "HAS_PYWAVELETS",
    "PROCESSING_METHODS",
    "METHOD_METADATA",
    "ALGORITHM_CATALOG",
    "METHOD_DISPLAY_NAMES",
    "PREFERRED_METHOD_ORDER",
    "METHOD_TAGS",
    "METHOD_CATEGORY_LABELS",
    "AUTO_TUNE_STAGE_BY_METHOD",
    "get_algorithm_spec",
    "get_field_approved_method_keys",
    "get_research_method_keys",
    "is_public_method",
    "get_method_display_name",
    "get_method_category",
    "get_auto_tune_stage",
    "get_method_category_label",
    "get_public_method_keys",
    "get_public_methods_grouped_by_category",
]
