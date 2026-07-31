#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public processing-method registry compatibility facade.

The large declarative tables are split by responsibility under
``core.method_registry_groups`` and ``core.method_registry_metadata``.  This
module preserves the historical public API used by the current Qt frontend,
CLI, tests, and compatibility adapters.
"""
from __future__ import annotations

from core.algorithm_specs import AlgorithmCatalog
from core.method_registry_bindings import *  # noqa: F401,F403
from core.method_registry_groups import (
    PROCESSING_METHODS_BACKGROUND_DENOISE,
    PROCESSING_METHODS_CALIBRATION,
    PROCESSING_METHODS_IMAGING,
    PROCESSING_METHODS_MOTION,
)
from core.method_registry_metadata import (
    AUTO_TUNE_STAGE_BY_METHOD,
    METHOD_CATEGORY_LABELS,
    METHOD_DISPLAY_NAMES,
    METHOD_METADATA,
    METHOD_TAGS,
    PREFERRED_METHOD_ORDER,
)

PROCESSING_METHODS = {
    **PROCESSING_METHODS_CALIBRATION,
    **PROCESSING_METHODS_BACKGROUND_DENOISE,
    **PROCESSING_METHODS_MOTION,
    **PROCESSING_METHODS_IMAGING,
}

for _method_key, _meta in METHOD_METADATA.items():
    if _method_key in PROCESSING_METHODS:
        PROCESSING_METHODS[_method_key].update(_meta)
        PROCESSING_METHODS[_method_key]["name"] = _meta["display_name"]

for _method_key, _stage in AUTO_TUNE_STAGE_BY_METHOD.items():
    if _method_key in PROCESSING_METHODS:
        PROCESSING_METHODS[_method_key]["auto_tune_stage"] = _stage

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
        or method.get("display_name")
        or method.get("name")
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
