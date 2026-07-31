#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Product-mode toggles for field and developer UI surfaces.

The default MyGPR package opens as a field exploration and positioning product.
Developer tools remain available only when explicitly enabled.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Mapping

RESEARCH_UI_ENV = "MYGPR_ENABLE_RESEARCH_UI"
PRODUCT_MODE_ENV = "MYGPR_PRODUCT_MODE"
_TRUE_VALUES = {"1", "true", "yes", "y", "on", "research", "dev", "developer", "development"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "prod", "production", "field", "engineering"}


def _normalized(value: str | None) -> str:
    return (value or "").strip().lower()


def is_research_ui_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether research-only UI entry points should be visible.

    Default is False.  Enable explicitly with one of:

    - ``MYGPR_ENABLE_RESEARCH_UI=1``
    - ``MYGPR_PRODUCT_MODE=research`` or ``dev``
    """

    source = os.environ if env is None else env
    explicit = _normalized(source.get(RESEARCH_UI_ENV))
    if explicit:
        return explicit in _TRUE_VALUES
    mode = _normalized(source.get(PRODUCT_MODE_ENV))
    if mode in _FALSE_VALUES:
        return False
    return mode in _TRUE_VALUES


def build_workspaces(env: Mapping[str, str] | None = None) -> "OrderedDict[str, str]":
    """Build the top-level workspaces for the current product mode."""

    workspaces: "OrderedDict[str, str]" = OrderedDict(
        [
            ("data_management", "项目管理"),
            ("processing_lab", "测线处理"),
            ("interpretation", "界面标注"),
            ("spatial", "空间成果"),
            ("delivery", "成果报告"),
        ]
    )
    if is_research_ui_enabled(env):
        # Keep the development-only page next to processing, where its dry-run
        # validation semantics make sense, but never show it in field mode.
        workspaces = OrderedDict(
            [
                ("data_management", "项目管理"),
                ("processing_lab", "测线处理"),
                ("simulation_validation", "仿真验证"),
                ("interpretation", "界面标注"),
                ("spatial", "空间成果"),
                ("delivery", "成果报告"),
            ]
        )
    return workspaces


__all__ = [
    "PRODUCT_MODE_ENV",
    "RESEARCH_UI_ENV",
    "build_workspaces",
    "is_research_ui_enabled",
]
