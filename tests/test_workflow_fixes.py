#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试工作流修复：节点映射和链接初始化"""

import pytest
from core.workflow_data import (
    WorkflowConfig,
    WorkflowMethod,
    build_default_workflow_config,
    get_config_manager,
)


class TestWorkflowLinkInitialization:
    """测试 canvas links 初始化策略"""

    def test_default_template_has_linear_links(self):
        """测试1: 默认模板首次创建有线性主链"""
        config = build_default_workflow_config("high_quality_uav_gpr")
        methods = sorted(config.methods, key=lambda m: m.order)
        
        # 应该有连线
        assert len(config.canvas_links) > 0, "默认模板应该有连线"
        
        # 检查连线是否是线性链
        for i in range(len(methods) - 1):
            left = methods[i]
            right = methods[i + 1]
            found = False
            for link in config.canvas_links:
                if link.from_node == left.node_id and link.to_node == right.node_id:
                    found = True
                    break
            assert found, f"应该有从 {left.node_id} 到 {right.node_id} 的连线"

    def test_deleted_links_not_restored_after_save_reload(self):
        """测试2: 用户删除所有连线后，保存/重载后不会自动恢复"""
        config = build_default_workflow_config("high_quality_uav_gpr")
        initial_links = len(config.canvas_links)
        assert initial_links > 0
        
        # 用户删除所有连线
        config.canvas_links = []
        assert len(config.canvas_links) == 0
        
        # 调用 ensure_canvas_links (模拟 _render_steps)
        config.ensure_canvas_links()
        
        # 连线应该仍然为空（因为已经初始化过）
        assert len(config.canvas_links) == 0, "删除的连线不应该被自动恢复"
        
        # 保存并重新加载
        config_manager = get_config_manager()
        filepath = config_manager.save_config(config)
        
        loaded_config = config_manager.load_config(filepath)
        
        # 加载后的配置连线也应该为空
        assert len(loaded_config.canvas_links) == 0, "重载后删除的连线不应该被恢复"
        
        # 清理测试文件
        import os
        os.remove(filepath)

    def test_legacy_template_without_links_field_gets_default_links(self):
        """测试3: 旧模板没有 canvas_links 字段时能生成默认链接"""
        # 模拟旧模板数据（没有 canvas_links 和 _links_initialized 字段）
        legacy_data = {
            "version": "1.0",
            "name": "旧模板",
            "template_type": "user",
            "realtime_enabled": True,
            "methods": [
                {"category": "preprocessing", "method_id": "dc_shift", "enabled": True, "order": 0},
                {"category": "preprocessing", "method_id": "dewow", "enabled": True, "order": 1},
            ],
            "canvas_layout": {"nodes": {}},
            "created_at": "2024-01-01T00:00:00",
            "last_modified": "2024-01-01T00:00:00",
        }
        
        config = WorkflowConfig.from_dict(legacy_data)
        
        # 应该生成默认连线
        assert len(config.canvas_links) == 1, "旧模板应该生成默认连线"
        assert config.canvas_links[0].from_node is not None
        assert config.canvas_links[0].to_node is not None


class TestWorkflowResultMapping:
    """测试工作流运行结果节点映射"""

    def test_run_selected_updates_only_selected_node(self):
        """测试4: Run Selected 只更新选中节点"""
        config = build_default_workflow_config("high_quality_uav_gpr")
        
        # 确保所有节点状态都是 pending
        for method in config.methods:
            method.status = "pending"
        
        # 选择第2个节点（index=1）
        selected_method = config.methods[1]
        
        # 模拟运行结果（只有选中节点的输出）
        outputs = [{
            "method_key": selected_method.method_id,
            "node_id": selected_method.node_id,
            "elapsed_ms": 100.0,
            "data": None,
        }]
        
        # 更新状态
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication.instance() or QApplication(sys.argv)
        
        page = WorkflowPage()
        page.config = config
        page._last_run_methods = [selected_method]
        page.set_run_result(outputs, realtime=False)
        
        # 只有选中的节点应该是 success
        for i, method in enumerate(config.methods):
            if i == 1:
                assert method.status == "success", "选中的节点应该是 success"
            else:
                assert method.status != "success", f"未选中的节点 {i} 不应该是 success"

    def test_duplicate_method_ids_not_overwritten(self):
        """测试5: 两个相同 method_id 节点状态不互相覆盖"""
        config = WorkflowConfig()
        
        # 添加两个相同 method_id 的节点
        method1 = WorkflowMethod(category="preprocessing", method_id="dc_shift", order=0)
        method2 = WorkflowMethod(category="preprocessing", method_id="dc_shift", order=1)
        config.methods = [method1, method2]
        config.ensure_canvas_links()
        
        # 确保两个节点有不同的 node_id
        assert method1.node_id != method2.node_id, "两个节点应该有不同的 node_id"
        
        # 模拟运行第一个节点
        outputs = [{
            "method_key": "dc_shift",
            "node_id": method1.node_id,
            "elapsed_ms": 50.0,
            "data": None,
        }]
        
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication.instance() or QApplication(sys.argv)
        
        page = WorkflowPage()
        page.config = config
        page._last_run_methods = [method1]
        page.set_run_result(outputs, realtime=False)
        
        # 只有第一个节点应该是 success
        assert method1.status == "success", "第一个节点应该是 success"
        assert method2.status != "success", "第二个节点不应该是 success"

    def test_run_from_current_updates_only_from_current(self):
        """测试6: Run From Current 只更新从当前节点开始的状态"""
        config = build_default_workflow_config("high_quality_uav_gpr")
        
        # 确保所有节点状态都是 pending
        for method in config.methods:
            method.status = "pending"
        
        # 从第2个节点开始运行（index=1）
        start_index = 1
        methods_to_run = config.methods[start_index:]
        
        # 模拟运行结果
        outputs = []
        for method in methods_to_run:
            outputs.append({
                "method_key": method.method_id,
                "node_id": method.node_id,
                "elapsed_ms": 100.0,
                "data": None,
            })
        
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication.instance() or QApplication(sys.argv)
        
        page = WorkflowPage()
        page.config = config
        page._last_run_methods = methods_to_run
        page.set_run_result(outputs, realtime=False)
        
        # 检查状态（跳过隐藏节点）
        for i, method in enumerate(config.methods):
            if method.hidden:
                continue  # 忽略隐藏节点
            if i >= start_index:
                assert method.status == "success", f"从当前开始的节点 {i} 应该是 success"
            else:
                assert method.status != "success", f"当前节点之前的节点 {i} 不应该是 success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])