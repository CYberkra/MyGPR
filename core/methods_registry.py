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

Lazy build（v0.9.38 启动性能）：import 本模块不再级联
``core.method_registry_bindings``（其顶层拉起全部 PythonModule 算法实现 →
scipy，importtime 实测 ~1.1s）——``PROCESSING_METHODS`` / ``ALGORITHM_CATALOG``
/ ``method_*`` callables 改为首次访问时经 PEP 562 ``__getattr__`` 一次性
构建。方法目录加载、处理链执行、CLI 等运行时场景照旧透明；仅 UI 启动
import 期不再为整个算法栈付费。
"""
from __future__ import annotations

import importlib.util
from typing import Any

from core.algorithm_specs import AlgorithmCatalog
from core.method_registry_metadata import (
    AUTO_TUNE_STAGE_BY_METHOD,
    METHOD_CATEGORY_LABELS,
    METHOD_DISPLAY_NAMES,
    METHOD_METADATA,
    METHOD_TAGS,
    PREFERRED_METHOD_ORDER,
)

# HAS_PYWAVELETS：零 import 判定。旧实现从 method_registry_bindings 顶层
# import（连带 pywt 与整个算法栈）；find_spec 只查存在性不执行模块。
# 语义差异仅在"pywt 存在但 import 即崩"的损坏环境，可接受。
HAS_PYWAVELETS = importlib.util.find_spec("pywt") is not None

# Methods whose legacy implementation is hardcoded in ProcessingEngine._run_legacy_adapter.
# For these, ``func`` must remain non-callable so the engine dispatches to the
# historical implementation rather than the native callable.
_LEGACY_ADAPTER_METHODS = frozenset({
    "compensatingGain",
    "agcGain",
    "subtracting_average_2D",
    "running_average_2D",
})


def _get_native_algorithms() -> dict[str, Any]:
    """Lazy import of NATIVE_ALGORITHMS（避免 core→mygpr.infrastructure 顶层依赖）。

    TODO: 1.1.0 前通过端口/延迟注入彻底消除此依赖，届时删除本函数与迁移豁免。
    """
    from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS
    return NATIVE_ALGORITHMS


def _legacy_adapter(native_func: Any) -> Any:
    """Wrap a native ``(data, params)`` callable into ``(data, **kwargs)`` for the legacy engine."""
    def wrapper(data: Any, **kwargs: Any) -> Any:
        return native_func(data, kwargs)
    return wrapper


def _metadata_records() -> dict[str, dict[str, Any]]:
    """元数据级方法记录（零 scipy）：目录/UI/查询助手共用。

    与 :func:`_build_registry` 的差异仅在于 ``func``/``native_func`` 不解析
    （None 占位，legacy-adapter 哨兵字符串照旧）——实现函数与 bindings 的
    导入（scipy 级联 ~1.1s）推迟到完整构建。算法键集、元数据字段与
    ALGORITHM_CATALOG 派生结果与完整构建完全一致。
    """
    if "PROCESSING_METHODS_META" not in globals():
        processing_methods: dict[str, dict[str, Any]] = {}
        for method_id, algorithm in _get_native_algorithms().items():
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

            # 元数据级不解析实现：func=None（spec.callable 回落 module 占位）。
            # legacy-adapter 哨兵保持字符串，与完整构建一致。
            if method_id in _LEGACY_ADAPTER_METHODS:
                func: Any = method_id  # string sentinel → engine falls back to _run_legacy_adapter
                module_name = method_id
            else:
                func = None
                module_name = method_id

            processing_methods[method_id] = {
                "name": str(meta.get("display_name") or algorithm.name or method_id),
                "type": "native",
                "module": module_name,
                "func": func,
                "native_func": None,
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
            if method_id in processing_methods:
                processing_methods[method_id]["auto_tune_stage"] = stage

        globals()["PROCESSING_METHODS_META"] = processing_methods
        globals()["ALGORITHM_CATALOG"] = AlgorithmCatalog.from_legacy(processing_methods)
    return globals()["PROCESSING_METHODS_META"]  # type: ignore[return-value]


def _metadata_catalog() -> AlgorithmCatalog:
    """元数据级目录入口（零 scipy；首次访问触发元数据构建）。"""
    if "ALGORITHM_CATALOG" not in globals():
        _metadata_records()
    return globals()["ALGORITHM_CATALOG"]  # type: ignore[return-value]


def method_metadata_records() -> dict[str, dict[str, Any]]:
    """Public accessor：UI 目录等元数据级消费者使用（不触发算法实现导入）。"""
    return _metadata_records()


def _build_registry() -> None:
    """一次性构建 PROCESSING_METHODS 与 ALGORITHM_CATALOG（见模块 docstring）。"""
    import core.method_registry_bindings as _bindings

    # 与旧 wildcard re-export 同语义：method_* / _method_* 前缀名
    #（bindings.__all__ 即按此两前缀生成）。
    legacy_callables: dict[str, Any] = {}
    for name in dir(_bindings):
        if name.startswith("method_") and callable(getattr(_bindings, name)):
            legacy_callables[name.replace("method_", "", 1)] = getattr(_bindings, name)

    processing_methods: dict[str, dict[str, Any]] = {}
    for method_id, algorithm in _get_native_algorithms().items():
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
            func: Any = method_id  # string sentinel → engine falls back to _run_legacy_adapter
        else:
            func = legacy_callables.get(method_id, _legacy_adapter(algorithm.function))
        if hasattr(func, "__module__") and func.__module__:
            module_name = func.__module__.split(".")[-1]
        else:
            module_name = method_id

        processing_methods[method_id] = {
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
        if method_id in processing_methods:
            processing_methods[method_id]["auto_tune_stage"] = stage

    globals()["PROCESSING_METHODS"] = processing_methods
    globals()["ALGORITHM_CATALOG"] = AlgorithmCatalog.from_legacy(processing_methods)


def _registry() -> dict[str, dict[str, Any]]:
    """模块内取值入口：首次调用触发构建（模块内全局名查找不走 __getattr__）。"""
    if "PROCESSING_METHODS" not in globals():
        _build_registry()
    return globals()["PROCESSING_METHODS"]  # type: ignore[return-value]


def __getattr__(name: str) -> Any:
    """PEP 562 惰性导出：PROCESSING_METHODS / ALGORITHM_CATALOG / method_* callables。

    兼容历史 ``from core.methods_registry import X``（cli_batch、tests）与
    属性访问；模块内部代码一律经 :func:`_registry` 取值。
    """
    if name in ("PROCESSING_METHODS", "ALGORITHM_CATALOG"):
        _registry()
        return globals()[name]
    if name.startswith("method_"):
        import core.method_registry_bindings as _bindings
        return getattr(_bindings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_algorithm_spec(method_key: str):
    if "ALGORITHM_CATALOG" not in globals():
        _registry()
    return globals()["ALGORITHM_CATALOG"].get(method_key)


def get_field_approved_method_keys() -> list[str]:
    if "ALGORITHM_CATALOG" not in globals():
        _registry()
    return [spec.algorithm_id for spec in globals()["ALGORITHM_CATALOG"].production()]


def get_research_method_keys() -> list[str]:
    if "ALGORITHM_CATALOG" not in globals():
        _registry()
    return [spec.algorithm_id for spec in globals()["ALGORITHM_CATALOG"].research()]


def is_public_method(method_key: str) -> bool:
    """Whether a method should appear in the public GUI lists."""
    methods = _registry()
    method = methods.get(method_key, {})
    if not method or str(method_key).startswith("_"):
        return False
    return str(method.get("visibility", "public")) == "public"


def get_method_display_name(method_key: str) -> str:
    """Return unified user-facing method name."""
    method = _registry().get(method_key, {})
    return str(
        METHOD_DISPLAY_NAMES.get(method_key)
        or method.get("name")
        or method.get("display_name")
        or method_key
    )


def get_method_category(method_key: str) -> str:
    """Return internal category key for a method."""
    method = _registry().get(method_key, {})
    return str(method.get("category", "experimental"))


def get_auto_tune_stage(method_key: str) -> str:
    """Return stage-level auto-tune grouping for a method."""
    method = _registry().get(method_key, {})
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
    methods = _registry()
    ordered = [key for key in PREFERRED_METHOD_ORDER if is_public_method(key)]
    tail = [
        key
        for key in methods.keys()
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
    "method_metadata_records",
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
