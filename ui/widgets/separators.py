#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用小部件工厂（任务 F 候选 4）：分隔线、页标题等跨页复用件。

原来在 4 个页面各有一份 `_create_separator` 私有工厂（值硬编码
`#e0e0e0` 或 rgba 灰），浅色固定、无主题感知；统一收敛到这里，
以半透明中性灰实现——随主题底色自适应，视觉上等价于 style_spec §1
的分隔线形态（QFrame.HLine + Sunken）。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame

from qfluentwidgets import SubtitleLabel

from ui import constants

__all__ = ['make_h_separator', 'make_separator', 'make_page_title']


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


def make_page_title(text: str) -> SubtitleLabel:
    """页面标题：SubtitleLabel 微软雅黑 12pt Bold 居中（SPEC §1）。

    收敛自 delivery/interpretation/jobs/processing/spatial 五页各自
    逐字重复的私有 `_page_title`（UI 一致性收敛轮）。
    """
    label = SubtitleLabel(text)
    label.setFont(QFont(constants.FONT_FAMILY, 12, QFont.Weight.Bold))
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label
