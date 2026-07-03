#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processing-panel wording and guardrails for the field workbench.

This module deliberately contains no processing execution logic.  It keeps UI
wording and field-workbench exposure rules in one small place so the main window
file does not become the only policy holder.
"""

from __future__ import annotations

PROCESSING_SETTINGS_TITLE = "处理设置"
PROCESSING_CATEGORY_LABEL = "算法分类"
PROCESSING_METHOD_LABEL = "选择算法"
PROCESSING_PARAMS_TITLE = "参数设置"
PROCESSING_OPERATION_TITLE = "处理操作"
PARAM_RECOMMEND_BUTTON_TEXT = "✣  推荐当前参数"

# These phrases are product wording guardrails requested for the MyGPR field UI.
FORBIDDEN_USER_VISIBLE_PHRASES = (
    "单算法处理",
    "重采样类",
)

DISPLAY_COMPARE_CAPABILITY_NOTE = (
    "显示与对比能力先作为测线处理页或成果页的子面板接入；"
    "time_to_depth 属于坐标轴/显示变换相关能力，保留在算法入口中并记录输出轴信息。"
)


def assert_processing_wording(text: str) -> None:
    """Raise if a field UI text contains a forbidden product phrase."""
    for phrase in FORBIDDEN_USER_VISIBLE_PHRASES:
        if phrase in text:
            raise AssertionError(f"Forbidden field UI wording: {phrase}")


__all__ = [
    "DISPLAY_COMPARE_CAPABILITY_NOTE",
    "FORBIDDEN_USER_VISIBLE_PHRASES",
    "PARAM_RECOMMEND_BUTTON_TEXT",
    "PROCESSING_CATEGORY_LABEL",
    "PROCESSING_METHOD_LABEL",
    "PROCESSING_OPERATION_TITLE",
    "PROCESSING_PARAMS_TITLE",
    "PROCESSING_SETTINGS_TITLE",
    "assert_processing_wording",
]
