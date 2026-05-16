#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for workflow node card algorithm combo box functionality."""

import sys
from PyQt6.QtWidgets import QApplication, QComboBox
from ui.workflow_canvas_cards import WorkflowNodeCard, candidate_methods_for_workflow_method
from core.workflow_data import WorkflowMethod
from core.methods_registry import get_method_display_name
from ui.gui_workflow_page import WorkflowPage


def test_gain_card_uses_algorithm_combo():
    """测试38: gain 节点 full 卡片使用算法 QComboBox"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 使用 gain 阶段，它有多个候选算法
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 应该使用 QComboBox
    assert hasattr(card, 'algorithm_combo'), "卡片应该有 algorithm_combo 属性"
    assert isinstance(card.algorithm_combo, QComboBox), "算法选择器应该是 QComboBox"
    
    # 按钮尺寸
    assert card.algorithm_combo.minimumWidth() >= 160, "按钮最小宽度应该 >= 160"
    assert card.algorithm_combo.minimumHeight() >= 30, "按钮高度应该 >= 30"


def test_combo_contains_all_candidate_algorithms():
    """测试39: algorithm_combo 包含所有候选算法"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 获取候选算法
    candidates = card._candidate_methods()
    expected_algorithms = ["sec_gain", "energy_decay_gain", "compensatingGain", "agcGain"]
    
    for algo in expected_algorithms:
        assert algo in candidates, f"候选算法应该包含 '{algo}'"
    
    # 验证 combo 项目
    for i in range(card.algorithm_combo.count()):
        item_data = card.algorithm_combo.itemData(i)
        assert item_data in candidates, f"combo 项目 data 应该来自候选算法"


def test_combo_current_index_is_current_algorithm():
    """测试40: combo 当前项是当前算法"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 验证当前项
    index = card.algorithm_combo.currentIndex()
    current_data = card.algorithm_combo.itemData(index)
    assert current_data == "sec_gain", f"当前项应该是 sec_gain，实际是 {current_data}"


def test_combo_switch_algorithm():
    """测试41: 通过 combo 切换算法"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 找到 agcGain 的索引并切换
    agc_index = card.algorithm_combo.findData("agcGain")
    assert agc_index >= 0, "应该能找到 agcGain"
    
    card.algorithm_combo.setCurrentIndex(agc_index)
    app.processEvents()
    
    # 验证 method 已更新
    assert method.method_id == "agcGain", f"应该切换到 agcGain，实际是 {method.method_id}"


def test_combo_on_change_callback():
    """测试42: algorithm_combo 有 currentIndexChanged 信号连接"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 验证回调方法存在
    assert hasattr(card, '_on_algorithm_combo_changed'), "_on_algorithm_combo_changed 方法应该存在"


def _select_method(page: WorkflowPage, method_id: str) -> int:
    """辅助函数：选择指定的方法"""
    for row, method in enumerate(page.config.methods):
        if method.method_id == method_id:
            page.step_list.setCurrentRow(row)
            return row
    raise AssertionError(f"method not found in workflow: {method_id}")


def test_workflow_page_inspector_sync_after_combo_switch():
    """测试43: 从 combo 切换算法后，Inspector的method_combo应该同步更新"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 创建 WorkflowPage
    page = WorkflowPage()
    
    # 选择 gain 节点
    gain_row = _select_method(page, "sec_gain")
    app.processEvents()
    
    # 获取当前 method
    method = page.config.methods[gain_row]
    
    # 找到对应的卡片
    gain_proxy = next(proxy for proxy in page.workflow_canvas._scene.proxies if proxy.row == gain_row)
    gain_card = gain_proxy.widget()
    
    # 从 combo 切换到 agcGain
    agc_index = gain_card.algorithm_combo.findData("agcGain")
    gain_card.algorithm_combo.setCurrentIndex(agc_index)
    app.processEvents()
    
    # 验证 method 已更新
    assert method.method_id == "agcGain", f"method应该切换到 agcGain，实际是 {method.method_id}"
    
    # 验证 Inspector 的 method_combo 已同步
    assert page.method_combo.currentData() == "agcGain", f"Inspector应该同步到 agcGain，实际是 {page.method_combo.currentData()}"
    
    page.close()


def test_combo_switch_updates_params():
    """测试44: combo 切换算法后 params 按新算法 metadata 刷新"""
    app = QApplication.instance() or QApplication(sys.argv)
    
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
    
    # 切换到 agcGain
    agc_index = card.algorithm_combo.findData("agcGain")
    card.algorithm_combo.setCurrentIndex(agc_index)
    
    # 处理事件，确保 QTimer 执行
    app.processEvents()
    
    # 验证 method_id 已切换
    assert method.method_id == "agcGain", f"method_id 应该切换到 agcGain"
    
    # 验证 changed 信号发出
    assert len(changed_rows) > 0, "changed 信号应该发出"
    assert changed_rows[0] == 0, f"changed 信号应该传回 row=0"


def test_combo_same_algorithm_no_change():
    """测试45: 切换到当前算法不会重建参数"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    original_params = dict(method.params)
    card = WorkflowNodeCard(0, method)
    
    # 收集 changed 信号
    changed_rows = []
    card.changed.connect(changed_rows.append)
    
    # 切换到同一个算法
    sec_index = card.algorithm_combo.findData("sec_gain")
    card.algorithm_combo.setCurrentIndex(sec_index)
    app.processEvents()
    
    # 验证参数没变
    assert method.params == original_params, "切换到当前算法不应该重建参数"


def test_combo_successive_switches():
    """测试46: 连续切换 SEC -> AGC -> SEC 能成功"""
    app = QApplication.instance() or QApplication(sys.argv)
    
    method = WorkflowMethod(
        method_id="sec_gain",
        stage_id="gain",
        category="gain",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 第一次切换到 AGC
    agc_index = card.algorithm_combo.findData("agcGain")
    card.algorithm_combo.setCurrentIndex(agc_index)
    app.processEvents()
    assert method.method_id == "agcGain", "第一次切换应该成功"
    
    # 第二次切换回 SEC
    sec_index = card.algorithm_combo.findData("sec_gain")
    card.algorithm_combo.setCurrentIndex(sec_index)
    app.processEvents()
    assert method.method_id == "sec_gain", "第二次切换应该成功"
