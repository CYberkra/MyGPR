def test_gain_card_algorithm_menu_button():
    """测试38: 迁移节点 full 卡片使用算法 QToolButton 而非 QComboBox"""
    import sys
    from PyQt6.QtWidgets import QApplication, QToolButton
    from ui.workflow_canvas_cards import WorkflowNodeCard
    from core.workflow_data import WorkflowMethod
    
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


def test_gain_card_algorithm_menu_contains_candidates():
    """测试39: 算法菜单包含所有候选算法"""
    import sys
    from PyQt6.QtWidgets import QApplication
    from ui.workflow_canvas_cards import WorkflowNodeCard
    from core.workflow_data import WorkflowMethod
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 使用 migration 阶段，它有多个候选算法
    method = WorkflowMethod(
        method_id="kirchhoff_migration",
        stage_id="migration",
        category="migration",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 检查菜单存在
    assert hasattr(card, 'algorithm_menu'), "卡片应该有 algorithm_menu 属性"
    assert card.algorithm_menu is not None, "算法菜单应该存在"
    
    # 获取候选算法
    candidates = card._candidate_methods()
    menu_actions = [action.text() for action in card.algorithm_menu.actions()]
    
    # 菜单应该包含所有候选算法的显示名
    from core.methods_registry import get_method_display_name
    expected_names = [get_method_display_name(key) for key in candidates]
    
    for name in expected_names:
        assert name in menu_actions, f"菜单应该包含 '{name}'"


def test_algorithm_menu_current_is_checked():
    """测试40: 当前算法的菜单项应该是 checked"""
    import sys
    from PyQt6.QtWidgets import QApplication
    from ui.workflow_canvas_cards import WorkflowNodeCard
    from core.workflow_data import WorkflowMethod
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 测试 kirchhoff_migration 是当前算法
    method = WorkflowMethod(
        method_id="kirchhoff_migration",
        stage_id="migration",
        category="migration",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 找到 kirchhoff_migration 的 action 并检查它是否 checked
    found_checked = False
    for action in card.algorithm_menu.actions():
        if action.data() == "kirchhoff_migration":
            assert action.isCheckable(), "kirchhoff_migration action 应该是 checkable"
            assert action.isChecked(), "kirchhoff_migration action 应该被 checked"
            found_checked = True
            break
    
    assert found_checked, "应该找到并 checked kirchhoff_migration action"


def test_algorithm_menu_switch_updates_method():
    """测试41: 点击算法菜单项后更新 method"""
    import sys
    from PyQt6.QtWidgets import QApplication
    from ui.workflow_canvas_cards import WorkflowNodeCard
    from core.workflow_data import WorkflowMethod
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 开始于 kirchhoff_migration
    method = WorkflowMethod(
        method_id="kirchhoff_migration",
        stage_id="migration",
        category="migration",
    )
    method.order = 1
    card = WorkflowNodeCard(0, method)
    
    # 模拟点击 stolt_migration
    for action in card.algorithm_menu.actions():
        if action.data() == "stolt_migration":
            # 手动触发 _on_algorithm_menu_triggered
            card._on_algorithm_menu_triggered(action)
            break
    
    # 验证 method 已更新
    assert method.method_id == "stolt_migration", f"应该切换到 stolt_migration，实际是 {method.method_id}"
