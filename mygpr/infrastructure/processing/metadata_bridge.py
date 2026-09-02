#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native 目录与 core 展示元数据之间的唯一桥接点。

收敛目标（任务 F 候选 2）：``NativeProcessingCatalog`` 不再经
Composite/Legacy 适配器取元数据，但 UI 可见的中文 display_name、category、
visibility、auto_tune_* 仍以 core 为单一事实来源——``ui/desktop_backend_facade.py``
与 ``core/methods_registry.py`` 都直接消费 ``core.method_registry_metadata``。
本桥集中所有 core → mygpr 的元数据读取，将来 core 侧收敛/消亡时只需改动本模块。

说明：``core.methods_registry.PROCESSING_METHODS`` 自 v0.9.37 起由
``NATIVE_ALGORITHMS`` 投影构建并叠加 ``METHOD_METADATA`` 与
``AUTO_TUNE_STAGE_BY_METHOD``；读取其元数据字段即等价于旧
``LegacyProcessingCatalog.get()`` 的覆盖逻辑，因此描述符输出天然与基线一致。
"""
from __future__ import annotations

from typing import Any, FrozenSet

from core.methods_registry import PROCESSING_METHODS, get_auto_tune_stage

# 旧 LegacyProcessingCatalog 视为全局变换的方法（native 侧不标 global_transform
# 的实现需保留该能力标记，见任务 F 收敛基线 fixture）。
_LEGACY_GLOBAL_TRANSFORM_METHODS = frozenset(
    {"kirchhoff_migration", "stolt_migration", "fk_filter"}
)


def legacy_overlay(method_id: str) -> dict[str, Any]:
    """返回 core 侧对该方法的展示元数据覆盖。

    等价于旧 ``LegacyProcessingCatalog.get()`` 中从 ``PROCESSING_METHODS``
    派生的字段；``capabilities`` 为 legacy 视角额外标记的能力集合。
    """
    raw = PROCESSING_METHODS.get(str(method_id)) or {}
    stage = get_auto_tune_stage(str(method_id))
    auto_tune_enabled = bool(raw.get("auto_tune_enabled", False))
    legacy_capabilities: set[str] = {"ndarray"}
    if auto_tune_enabled:
        legacy_capabilities.add("auto_tune")
    if stage == "motion_comp":
        legacy_capabilities.add("trace_metadata")
    if str(method_id) in _LEGACY_GLOBAL_TRANSFORM_METHODS:
        legacy_capabilities.add("global_transform")
    return {
        "name": str(raw.get("name") or method_id),
        "category": str(raw.get("category") or "experimental"),
        "visibility": str(raw.get("visibility") or "public"),
        "auto_tune_enabled": auto_tune_enabled,
        "auto_tune_family": str(raw.get("auto_tune_family") or ""),
        "auto_tune_stage": stage,
        "maturity": str(raw.get("maturity") or "experimental"),
        "legacy_capabilities": frozenset(legacy_capabilities),
    }


def metadata_fields(method_id: str) -> tuple[dict[str, Any], FrozenSet[str]]:
    """兼容便捷接口：返回 (元数据覆盖 dict, legacy 能力集合)。

    供目录实现合并使用；单独存在以避免把能力集合塞进公开 metadata dict。
    """
    overlay = legacy_overlay(method_id)
    capabilities = overlay.pop("legacy_capabilities")
    return overlay, capabilities


__all__ = [
    "_LEGACY_GLOBAL_TRANSFORM_METHODS",
    "legacy_overlay",
    "metadata_fields",
]
