#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用小部件工厂（任务 F 候选 4）：分隔线等跨页复用件。

原来在 4 个页面各有一份 `_create_separator` 私有工厂（值硬编码
`#e0e0e0` 或 rgba 灰），浅色固定、无主题感知；统一收敛到这里，
以半透明中性灰实现——随主题底色自适应，视觉上等价于 style_spec §1
的分隔线形态（QFrame.HLine + Sunken）。
"""
from __future__ import annotations

from PyQt6.QtWidgets import QFrame

__all__ = ['make_h_separator']


def make_h_separator(*, alpha: int = 90) -> QFrame:
    """水平分隔线：QFrame.HLine + Sunken + 半透明中性灰（随主题自适应）。"""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet(f'color: rgba(128, 128, 128, {int(alpha)});')
    return line


def make_separator(vertical: bool = False, *, alpha: int = 90) -> QFrame:
    """分隔线工厂：HLine/VLine + Sunken + 半透明中性灰（随主题自适应）。

    收敛自各页面的私有 `_create_separator`（任务 F 候选 4）。
    """
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine if vertical else QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet(f'color: rgba(128, 128, 128, {int(alpha)});')
    return line
