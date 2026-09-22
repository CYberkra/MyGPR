# -*- coding: utf-8 -*-
"""项目页「不做 B-Scan 预览」回归测试（2026-09-22 移除决策）。

背景：项目页右列曾用纵向三卡 QSplitter（测线列表 / 处理成果 / 数据预览）。
实测预览区只得到 ~198px 卡片高 → B-Scan 绘图区 262px → 每采样 0.291px，
仅为可读阈值 0.45px 的 65%。数据 900 采样需 405px 绘图区，缺口 259px，
而两张表最多让出 135px——**结构上无解**，故移除预览，本页回归纯管理。

本文件锁定该结构，防止后续重构又塞回一个被压扁的预览：
  1. 页面内不得出现 BScanView；
  2. 不得再有 set_preview_bundle 接口（避免协作层继续喂数据）；
  3. 两张表都必须还在（腾出的空间归它们）。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from PyQt6.QtWidgets import QTableWidget  # noqa: E402

from ui.pages.project_page import ProjectPage  # noqa: E402
from ui.widgets.bscan_view import BScanView  # noqa: E402


@pytest.fixture
def page(qapp):
    p = ProjectPage()
    yield p
    p.close()


class TestNoBscanPreview:
    """项目页是管理页，不是查看页——不得内嵌 B-Scan。"""

    def test_no_bscan_view_child(self, page):
        assert page.findChildren(BScanView) == [], (
            '项目页不应内嵌 B-Scan：右列三卡分割后绘图区仅 262px，'
            '每采样 0.291px < 可读阈值 0.45px。查看请走处理页/解释页。')

    def test_no_preview_bundle_api(self, page):
        assert not hasattr(page, 'set_preview_bundle'), (
            '项目页不应保留 set_preview_bundle 接口，否则协作层会继续喂'
            '数据到不存在的预览控件。')

    def test_keeps_both_tables(self, page):
        tables = page.findChildren(QTableWidget)
        assert len(tables) >= 2, (
            '移除预览腾出的空间应归测线列表与处理成果两张表，'
            f'当前只找到 {len(tables)} 张表。')


class TestPreviewButtonsPointElsewhere:
    """既然本页无预览，按钮/菜单必须说清「去别处看」。"""

    def test_button_label_mentions_target_page(self, page):
        text = page.preview_artifact_btn.text()
        assert '处理页' in text, (
            f'成果按钮文案应指明去处（当前 {text!r}）——本页不再就地弹图，'
            '写成「预览所选」会让用户以为图片会出现在本页。')
