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


def test_gain_node_has_algorithm_menu_button():
    """测试42: 真实 gain 节点(full卡片)应存在 algorithmButton，菜单包含所有候选算法"""
    import sys
    from PyQt6.QtWidgets import QApplication, QToolButton
    from ui.workflow_canvas_cards import WorkflowNodeCard
    from core.workflow_data import WorkflowMethod
    from core.methods_registry import get_method_display_name
    
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
    
    # 菜单应包含所有候选算法
    menu = card.algorithm_button.menu()
    assert menu is not None, "按钮应该有菜单"
    
    # 检查候选算法
    expected_algorithms = ["sec_gain", "energy_decay_gain", "compensatingGain", "agcGain"]
    menu_data = [action.data() for action in menu.actions()]
    
    for algo in expected_algorithms:
        assert algo in menu_data, f"菜单应该包含 '{algo}'"
    
    # 验证当前算法(sec_gain)的 action 是 checked
    for action in menu.actions():
        if action.data() == "sec_gain":
            assert action.isCheckable(), "sec_gain action 应该是 checkable"
            assert action.isChecked(), "当前算法(sec_gain)应该被 checked"
            break


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


def test_gain_card_menu_switch_changes_algorithm_and_params():
    """测试43: 从卡片菜单切换 gain 算法，验证 method、params 和 changed 信号"""
    import sys
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QObject
    from ui.workflow_canvas_cards import WorkflowNodeCard
    from core.workflow_data import WorkflowMethod
    
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
    
    # 找到 agcGain action 并 trigger
    menu = card.algorithm_button.menu()
    agc_action = None
    for action in menu.actions():
        if action.data() == "agcGain":
            agc_action = action
            break
    
    assert agc_action is not None, "应该找到 agcGain action"
    card._on_algorithm_menu_triggered(agc_action)
    
    # 验证 method.method_id 已更新
    assert method.method_id == "agcGain", f"应该切换到 agcGain，实际是 {method.method_id}"
    
    # 验证 params 已按 agcGain metadata 刷新
    assert "window" in method.params, f"agcGain 应该有 window 参数，实际 params 是 {method.params}"
    
    # 验证 changed 信号发出
    assert len(changed_rows) > 0, "changed 信号应该发出"
    assert changed_rows[0] == 0, f"changed 信号应该传回 row=0，实际是 {changed_rows[0]}"
