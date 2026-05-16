#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for workflow node card algorithm selector functionality."""

import sys
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QApplication, QToolButton, QListWidget, QListWidgetItem
from ui.workflow_canvas_cards import WorkflowNodeCard, candidate_methods_for_workflow_method
from core.workflow_data import WorkflowMethod
from core.methods_registry import get_method_display_name
from ui.gui_workflow_page import WorkflowPage


def test_gain_card_uses_algorithm_button():
    """测试38: gain 节点 full 卡片使用算法 QToolButton"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 应该使用 QToolButton
    assert hasattr(card, 'algorithm_button'), "卡片应该有 algorithm_button 属性"
    assert isinstance(card.algorithm_button, QToolButton), "算法选择器应该是 QToolButton"
    
    # 按钮尺寸
    assert card.algorithm_button.minimumWidth() >= 160, "按钮最小宽度应该 >= 160"
    assert card.algorithm_button.minimumHeight() >= 30, "按钮高度应该 >= 30"


def test_algorithm_button_emits_signal():
    """测试39: 点击 algorithm_button 会发出 algorithm_selector_requested 信号"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 验证信号存在
    assert hasattr(card, 'algorithm_selector_requested'), "卡片应该有 algorithm_selector_requested 信号"
    
    # 收集信号
    received_signals = []
    card.algorithm_selector_requested.connect(lambda row, pos: received_signals.append((row, pos)))
    
    # 模拟按钮点击
    card._on_algorithm_button_clicked()
    
    # 验证信号发出
    assert len(received_signals) == 1, "应该发出一次信号"
    assert received_signals[0][0] == 0, "信号应该包含正确的 row"


def test_candidates_include_gain_algorithms():
    """测试40: gain 节点候选算法包含所有增益算法"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    candidates = card._candidate_methods()
    expected_algorithms = ["sec_gain", "energy_decay_gain", "compensatingGain", "agcGain"]
    
    for algo in expected_algorithms:
        assert algo in candidates, f"候选算法应该包含 '{algo}'"


def test_canvas_has_algorithm_selector_handler():
    """测试41: WorkflowCanvasView 有算法选择器处理方法"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    from ui.workflow_canvas_cards import WorkflowCanvasView
    
    canvas = WorkflowCanvasView()
    
    # 验证方法存在
    assert hasattr(canvas, '_on_algorithm_selector_requested'), "_on_algorithm_selector_requested 方法应该存在"
    assert hasattr(canvas, '_apply_algorithm_from_popup'), "_apply_algorithm_from_popup 方法应该存在"
    assert hasattr(canvas, '_do_switch_algorithm'), "_do_switch_algorithm 方法应该存在"
    
    # 验证信号存在
    assert hasattr(canvas, 'algorithm_selector_requested'), "canvas 应该有 algorithm_selector_requested 信号"


def _select_method(page: WorkflowPage, method_id: str) -> int:
    """辅助函数：选择指定的方法"""
    for row, method in enumerate(page.config.methods):
        if method.method_id == method_id:
            page.step_list.setCurrentRow(row)
            return row
    raise AssertionError(f"method not found in workflow: {method_id}")


def test_workflow_page_inspector_sync_after_algorithm_switch():
    """测试42: 从画布层切换算法后，Inspector应该同步更新"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    page = WorkflowPage()
    gain_row = _select_method(page, "sec_gain")
    app.processEvents()
    
    method = page.config.methods[gain_row]
    original_method_id = method.method_id
    
    # 找到对应的卡片
    gain_proxy = next(proxy for proxy in page.workflow_canvas._scene.proxies if proxy.row == gain_row)
    gain_card = gain_proxy.widget()
    
    # 使用画布的 _do_switch_algorithm 方法直接切换
    page.workflow_canvas._do_switch_algorithm(gain_row, "agcGain")
    app.processEvents()
    
    # 验证 method 已更新
    assert method.method_id == "agcGain", f"method应该切换到 agcGain，实际是 {method.method_id}"
    
    # 验证 Inspector 的 method_combo 已同步
    assert page.method_combo.currentData() == "agcGain", f"Inspector应该同步到 agcGain"
    
    page.close()


def test_same_algorithm_no_change():
    """测试43: 切换到当前算法不会触发切换"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 直接调用 _switch_algorithm_from_card
    card._switch_algorithm_from_card("sec_gain")
    app.processEvents()
    
    # method_id 不应该改变
    assert method.method_id == "sec_gain", "当前算法不应该改变"


def test_successive_switches():
    """测试44: 连续切换 SEC -> AGC -> SEC 能成功"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 第一次切换到 AGC
    card._switch_algorithm_from_card("agcGain")
    app.processEvents()
    assert method.method_id == "agcGain", "第一次切换应该成功"
    
    # 第二次切换回 SEC
    card._switch_algorithm_from_card("sec_gain")
    app.processEvents()
    assert method.method_id == "sec_gain", "第二次切换应该成功"
