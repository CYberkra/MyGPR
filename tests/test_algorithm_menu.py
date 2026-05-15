#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for workflow node card algorithm menu functionality."""

import sys
from PyQt6.QtWidgets import QApplication, QToolButton, QMenu
from ui.workflow_canvas_cards import WorkflowNodeCard, candidate_methods_for_workflow_method
from core.workflow_data import WorkflowMethod
from core.methods_registry import get_method_display_name


def test_gain_card_algorithm_menu_button():
    """测试38: 迁移节点 full 卡片使用算法 QToolButton 而非 QComboBox"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 使用 migration 阶段，它有多个候选算法
    method = WorkflowMethod(
        method_id="kirchhoff_migration",
        stage_id="migration",
        category="migration",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 应该使用 QToolButton 而不是 QComboBox
    assert hasattr(card, 'algorithm_button'), "卡片应该有 algorithm_button 属性"
    assert isinstance(card.algorithm_button, QToolButton), "算法选择器应该是 QToolButton"
    assert not hasattr(card, 'algorithm_combo') or card.algorithm_combo is None, "不应该是 QComboBox"
    
    # 按钮应该有正确的样式
    assert card.algorithm_button.minimumWidth() >= 150, "按钮最小宽度应该 >= 150"
    assert card.algorithm_button.minimumHeight() >= 26, "按钮高度应该 >= 26"


def test_gain_node_has_algorithm_menu_button():
    """测试42: 真实 gain 节点(full卡片)应存在 algorithmButton，菜单包含所有候选算法"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 创建真实的 gain 阶段节点
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 卡片应存在 nodeAlgorithmButton
    assert hasattr(card, 'algorithm_button'), "gain 卡片应该有 algorithm_button 属性"
    assert isinstance(card.algorithm_button, QToolButton), "算法选择器应该是 QToolButton"
    assert card.algorithm_button is not None, "algorithm_button 不应该是 None"
    
    # 检查候选算法
    candidates = card._candidate_methods()
    expected_algorithms = ["sec_gain", "energy_decay_gain", "compensatingGain", "agcGain"]
    
    for algo in expected_algorithms:
        assert algo in candidates, f"候选算法应该包含 '{algo}'"


def test_gain_card_algorithm_menu_contains_candidates():
    """测试39: 算法菜单包含所有候选算法"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 使用 migration 阶段，它有多个候选算法
    method = WorkflowMethod(
        method_id="kirchhoff_migration",
        stage_id="migration",
        category="migration",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 获取候选算法
    candidates = card._candidate_methods()
    
    # 构建菜单并检查
    menu = card._build_algorithm_menu()
    menu_actions = [action.text() for action in menu.actions()]
    
    # 菜单应该包含所有候选算法的显示名
    expected_names = [get_method_display_name(key) for key in candidates]
    
    for name in expected_names:
        assert name in menu_actions, f"菜单应该包含 '{name}'"


def test_algorithm_menu_current_is_checked():
    """测试40: 当前算法的菜单项应该是 checked"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 测试 kirchhoff_migration 是当前算法
    method = WorkflowMethod(
        method_id="kirchhoff_migration",
        stage_id="migration",
        category="migration",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 构建菜单
    menu = card._build_algorithm_menu()
    
    # 找到 kirchhoff_migration 的 action 并检查它是否 checked
    found_checked = False
    for action in menu.actions():
        if action.data() == "kirchhoff_migration":
            assert action.isCheckable(), "kirchhoff_migration action 应该是 checkable"
            assert action.isChecked(), "kirchhoff_migration action 应该被 checked"
            found_checked = True
            break
    
    assert found_checked, "应该找到并 checked kirchhoff_migration action"


def test_algorithm_menu_switch_updates_method():
    """测试41: 点击算法菜单项后更新 method"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 开始于 kirchhoff_migration
    method = WorkflowMethod(
        method_id="kirchhoff_migration",
        stage_id="migration",
        category="migration",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 使用新的 _switch_algorithm_from_card 方法
    card._switch_algorithm_from_card("stolt_migration")
    
    # 验证 method 已更新
    assert method.method_id == "stolt_migration", f"应该切换到 stolt_migration，实际是 {method.method_id}"


def test_gain_card_menu_switch_changes_algorithm_and_params():
    """测试43: 从卡片菜单切换 gain 算法，验证 method、params 和 changed 信号"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 创建 sec_gain 节点
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 收集 changed 信号
    changed_rows = []
    card.changed.connect(changed_rows.append)
    
    # 使用新的 _switch_algorithm_from_card 方法
    card._switch_algorithm_from_card("agcGain")
    
    # 验证 method.method_id 已更新
    assert method.method_id == "agcGain", f"应该切换到 agcGain，实际是 {method.method_id}"
    
    # 验证 params 已按 agcGain metadata 刷新
    assert "window" in method.params, f"agcGain 应该有 window 参数，实际 params 是 {method.params}"
    
    # 验证 changed 信号发出
    assert len(changed_rows) > 0, "changed 信号应该发出"
    assert changed_rows[0] == 0, f"changed 信号应该传回 row=0，实际是 {changed_rows[0]}"


def test_build_algorithm_menu_returns_proper_menu():
    """测试44: _build_algorithm_menu 返回正确构建的 QMenu"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    menu = card._build_algorithm_menu()
    
    # 验证返回的是 QMenu
    assert isinstance(menu, QMenu), "_build_algorithm_menu 应该返回 QMenu"
    
    # 验证菜单包含所有候选算法
    candidates = card._candidate_methods()
    action_data_list = [action.data() for action in menu.actions()]
    
    for candidate in candidates:
        assert candidate in action_data_list, f"菜单应该包含候选算法 {candidate}"
    
    # 验证当前算法是 checked
    for action in menu.actions():
        if action.data() == "sec_gain":
            assert action.isChecked(), "当前算法应该被 checked"
            break
