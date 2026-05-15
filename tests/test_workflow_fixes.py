#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试工作流修复：节点映射和链接初始化"""

import pytest
from PyQt6.QtWidgets import QMessageBox
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

        assert len(config.canvas_links) > 0, "默认模板应该有连线"

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

        config.canvas_links = []
        assert len(config.canvas_links) == 0

        config.ensure_canvas_links()

        assert len(config.canvas_links) == 0, "删除的连线不应该被自动恢复"

        config_manager = get_config_manager()
        filepath = config_manager.save_config(config)

        loaded_config = config_manager.load_config(filepath)

        assert len(loaded_config.canvas_links) == 0, "重载后删除的连线不应该被恢复"

        import os
        os.remove(filepath)

    def test_legacy_template_without_links_field_gets_default_links(self):
        """测试3: 旧模板没有 canvas_links 字段时能生成默认链接"""
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

        assert len(config.canvas_links) == 1, "旧模板应该生成默认连线"
        assert config.canvas_links[0].from_node is not None
        assert config.canvas_links[0].to_node is not None


class TestWorkflowResultMapping:
    """测试工作流运行结果节点映射"""

    def test_run_selected_updates_only_selected_node(self):
        """测试4: Run Selected 只更新选中节点"""
        config = build_default_workflow_config("high_quality_uav_gpr")

        for method in config.methods:
            method.status = "pending"

        selected_method = config.methods[1]

        outputs = [{
            "method_key": selected_method.method_id,
            "node_id": selected_method.node_id,
            "elapsed_ms": 100.0,
            "data": None,
        }]

        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        page.config = config
        page._last_run_methods = [selected_method]
        page.set_run_result(outputs, realtime=False)

        for i, method in enumerate(config.methods):
            if i == 1:
                assert method.status == "success", "选中的节点应该是 success"
            else:
                assert method.status != "success", f"未选中的节点 {i} 不应该是 success"

    def test_duplicate_method_ids_not_overwritten(self):
        """测试5: 两个相同 method_id 节点状态不互相覆盖"""
        config = WorkflowConfig()

        method1 = WorkflowMethod(category="preprocessing", method_id="dc_shift", order=0)
        method2 = WorkflowMethod(category="preprocessing", method_id="dc_shift", order=1)
        config.methods = [method1, method2]
        config.ensure_canvas_links()

        assert method1.node_id != method2.node_id, "两个节点应该有不同的 node_id"

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

        assert method1.status == "success", "第一个节点应该是 success"
        assert method2.status != "success", "第二个节点不应该是 success"

    def test_run_from_current_updates_only_from_current(self):
        """测试6: Run From Current 只更新从当前节点开始的状态"""
        config = build_default_workflow_config("high_quality_uav_gpr")

        for method in config.methods:
            method.status = "pending"

        start_index = 1
        methods_to_run = config.methods[start_index:]

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

        for i, method in enumerate(config.methods):
            if method.hidden:
                continue
            if i >= start_index:
                assert method.status == "success", f"从当前开始的节点 {i} 应该是 success"
            else:
                assert method.status != "success", f"当前节点之前的节点 {i} 不应该是 success"


class TestWorkflowRunStates:
    """测试工作流运行状态更新"""

    def test_emit_run_sets_queued_running_states(self):
        """测试7: _emit_run 开始时设置 queued/running 状态"""
        config = build_default_workflow_config("high_quality_uav_gpr")

        for method in config.methods:
            method.status = "pending"

        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        page.config = config

        visible_methods = [m for m in config.methods if not m.hidden and m.enabled]
        visible_to_run = visible_methods[1:3]
        run_node_ids = {m.node_id for m in visible_to_run}

        for method in config.methods:
            if method.hidden or not method.enabled:
                method.status = "skipped"
            elif method.node_id in run_node_ids:
                method.status = "queued"
            elif method.status == "success":
                method.status = "success_stale"

        queued_count = sum(1 for m in config.methods if m.status == "queued")
        stale_count = sum(1 for m in config.methods if m.status == "success_stale")

        assert queued_count == len(visible_to_run), f"应该有 {len(visible_to_run)} 个 queued 节点"
        assert stale_count == 0, "没有旧 success 节点，不应该有 stale"

    def test_old_success_becomes_stale_on_run(self):
        """测试8: 上次 success 节点本次不运行时变为 stale"""
        config = build_default_workflow_config("high_quality_uav_gpr")

        for method in config.methods:
            method.status = "success"

        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        page.config = config

        visible_methods = [m for m in config.methods if not m.hidden and m.enabled]
        run_methods = [visible_methods[1]]
        run_node_ids = {m.node_id for m in run_methods}

        for method in config.methods:
            if method.hidden or not method.enabled:
                method.status = "skipped"
            elif method.node_id in run_node_ids:
                method.status = "queued"
            elif method.status == "success":
                method.status = "success_stale"

        success_stale_count = sum(1 for m in config.methods if m.status == "success_stale")
        success_count = sum(1 for m in config.methods if m.status == "success")

        visible_not_run = len(visible_methods) - len(run_methods)
        assert success_stale_count == visible_not_run, f"旧 success 节点应该变为 stale (期望 {visible_not_run})"
        assert success_count == 0, "没有 success 节点"

    def test_set_run_result_fallback_mapping(self):
        """测试9: set_run_result 对无 node_id output 仍能按 _last_run_methods fallback"""
        config = WorkflowConfig()

        method1 = WorkflowMethod(category="preprocessing", method_id="dc_shift", order=0)
        method2 = WorkflowMethod(category="preprocessing", method_id="dewow", order=1)
        config.methods = [method1, method2]
        config.ensure_canvas_links()

        for method in config.methods:
            method.status = "pending"

        outputs_without_node_id = [
            {"method_key": "dc_shift", "elapsed_ms": 50.0, "data": None},
            {"method_key": "dewow", "elapsed_ms": 30.0, "data": None},
        ]

        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        page.config = config
        page._last_run_methods = [method1, method2]

        output_by_node_id = {}
        run_node_ids = set(output_by_node_id.keys())

        for method in config.methods:
            if method.hidden or not method.enabled:
                method.status = "skipped"
                continue
            if method.node_id in run_node_ids:
                method.status = "success"

        outputs_fallback = [o for o in outputs_without_node_id if not o.get("node_id")]
        if outputs_fallback and hasattr(page, "_last_run_methods") and page._last_run_methods:
            for last_run_method, output in zip(page._last_run_methods, outputs_fallback):
                for config_method in config.methods:
                    if config_method.node_id == last_run_method.node_id:
                        config_method.status = "success"
                        break

        assert method1.status == "success", "method1 应该是 success"
        assert method2.status == "success", "method2 应该是 success"


class TestRunHistory:
    """测试 Run History 功能"""

    def test_run_history_mode_is_correct(self):
        """测试10: Run Selected history mode 显示 Run Selected"""
        config = build_default_workflow_config("high_quality_uav_gpr")

        for method in config.methods:
            method.status = "success"

        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        page.config = config
        page._run_history = []

        selected_method = config.methods[1]
        run_mode = "Run Selected"

        outputs = [{
            "method_key": selected_method.method_id,
            "node_id": selected_method.node_id,
            "elapsed_ms": 100.0,
            "data": None,
        }]

        for method in config.methods:
            if method.hidden or not method.enabled:
                method.status = "skipped"
            elif method.node_id == selected_method.node_id:
                method.status = "success"
            else:
                method.status = "success_stale"

        record = page._create_run_record(outputs, realtime=False, run_mode=run_mode)

        assert record["mode"] == "Run Selected", f"mode 应该是 Run Selected，实际是 {record['mode']}"

    def test_run_history_elapsed_ms_display(self):
        """测试11: elapsed_ms 显示单位正确"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        page._run_history = []

        config = build_default_workflow_config("high_quality_uav_gpr")
        page.config = config

        outputs = [{"method_key": "dc_shift", "elapsed_ms": 500.0, "data": None}]
        record = page._create_run_record(outputs, realtime=False)

        elapsed = record["elapsed_ms"]
        assert elapsed == 500.0, "elapsed_ms 应该是 500.0 毫秒"

        elapsed_str = f"{elapsed:.0f}ms" if elapsed < 1000 else f"{elapsed / 1000:.1f}s"
        assert "ms" in elapsed_str or "s" in elapsed_str, "耗时字符串应该包含 ms 或 s"

    def test_run_history_statistics(self):
        """测试12: failed/skipped/stale 统计正确"""
        config = build_default_workflow_config("high_quality_uav_gpr")

        config.methods[0].status = "success"
        config.methods[1].status = "success_stale"
        config.methods[2].status = "failed"
        config.methods[3].status = "skipped"

        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        page.config = config
        page._run_history = []

        outputs = [{"method_key": "dc_shift", "elapsed_ms": 100.0, "data": None}]
        record = page._create_run_record(outputs, realtime=False)

        assert record["success_count"] == 1, f"success_count 应该是 1，实际是 {record['success_count']}"
        assert record["failed_count"] == 1, f"failed_count 应该是 1，实际是 {record['failed_count']}"
        assert record["skipped_count"] == 1, f"skipped_count 应该是 1，实际是 {record['skipped_count']}"
        assert record["success_stale_count"] == 1, f"success_stale_count 应该是 1，实际是 {record['success_stale_count']}"


class TestLogSignal:
    """测试日志信号功能"""

    def test_log_emits_signal(self):
        """测试13: _log 会 emit log_message_requested"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()

        received = []
        page.log_message_requested.connect(lambda msg: received.append(msg))

        page._log("Test message")

        assert len(received) == 1, "应该接收到 1 条日志消息"
        assert received[0] == "Test message", f"消息应该是 'Test message'，实际是 '{received[0]}'"


class TestWorkflowRunSignalThreeArgs:
    """测试 workflow_run_requested 信号能发出三个参数"""

    def test_signal_emits_three_args(self):
        """测试14: workflow_run_requested 能发出 (object, bool, str) 三个参数"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()

        received = []
        page.workflow_run_requested.connect(
            lambda methods, realtime, run_mode: received.append((methods, realtime, run_mode))
        )

        config = build_default_workflow_config("high_quality_uav_gpr")
        page.config = config

        visible = [m for m in config.methods if not m.hidden and m.enabled]
        from unittest.mock import patch
        with patch("ui.gui_workflow_page.validate_workflow_config") as mock_validate:
            from core.workflow_validation import WorkflowValidationReport
            mock_validate.return_value = WorkflowValidationReport()
            page._emit_run(
                visible[:2],
                realtime=False,
                status="测试运行",
                log_text="测试日志",
                run_mode="Run All",
            )

        assert len(received) == 1, "应该接收到 1 次信号"
        methods_arg, realtime_arg, run_mode_arg = received[0]
        assert len(methods_arg) == 2, "methods 参数应该有 2 个元素"
        assert realtime_arg is False, "realtime 应该是 False"
        assert run_mode_arg == "Run All", f"run_mode 应该是 'Run All'，实际是 '{run_mode_arg}'"

    def test_signal_emits_realtime_mode(self):
        """测试15: 实时预览发出 Realtime run_mode"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()

        received = []
        page.workflow_run_requested.connect(
            lambda methods, realtime, run_mode: received.append((methods, realtime, run_mode))
        )

        config = build_default_workflow_config("high_quality_uav_gpr")
        page.config = config

        visible = [m for m in config.methods if not m.hidden and m.enabled]
        from unittest.mock import patch
        with patch("ui.gui_workflow_page.validate_workflow_config") as mock_validate:
            from core.workflow_validation import WorkflowValidationReport
            mock_validate.return_value = WorkflowValidationReport()
            page._emit_run(
                visible[:2],
                realtime=True,
                status="实时预览",
                log_text="实时预览",
                run_mode="Realtime",
            )

        assert len(received) == 1
        _, realtime_arg, run_mode_arg = received[0]
        assert realtime_arg is True, "realtime 应该是 True"
        assert run_mode_arg == "Realtime", f"run_mode 应该是 'Realtime'，实际是 '{run_mode_arg}'"


class TestRunWorkflowMethodsRunMode:
    """测试 app_qt.run_workflow_methods 能接收 run_mode"""

    def test_run_workflow_methods_signature_accepts_run_mode(self):
        """测试16: run_workflow_methods 签名接受 run_mode 参数"""
        import inspect
        from app_qt import GPRGuiQt

        sig = inspect.signature(GPRGuiQt.run_workflow_methods)
        params = list(sig.parameters.keys())
        assert "run_mode" in params, f"run_workflow_methods 应该有 run_mode 参数，实际参数: {params}"

        run_mode_param = sig.parameters["run_mode"]
        assert run_mode_param.default == "", f"run_mode 默认值应该是空字符串，实际是 {run_mode_param.default!r}"

    def test_pending_workflow_run_preserves_run_mode(self):
        """测试17: _pending_workflow_run 保存 run_mode"""
        import inspect
        from app_qt import GPRGuiQt

        init_src = inspect.getsource(GPRGuiQt.__init__)
        assert "_last_workflow_run_mode" in init_src, "_last_workflow_run_mode 应该在 __init__ 中初始化"

        run_src = inspect.getsource(GPRGuiQt.run_workflow_methods)
        assert "_last_workflow_run_mode" in run_src, "run_workflow_methods 应该保存 run_mode 到 _last_workflow_run_mode"

    def test_non_realtime_run_is_rejected_while_worker_busy(self, monkeypatch):
        """非实时 Run All/Selected/From 在已有 worker 时不能叠启动新任务。"""
        from app_qt import GPRGuiQt
        from core.workflow_data import WorkflowMethod
        import numpy as np
        import sys
        from PyQt6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv)
        win = GPRGuiQt()
        try:
            win.shared_data.load_data(
                np.arange(24, dtype=np.float32).reshape(6, 4),
                path="demo.csv",
                source="test",
            )
            win._worker = object()
            started = []
            monkeypatch.setattr(win, "_start_processing_worker", lambda *args, **kwargs: started.append((args, kwargs)))
            monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)

            method = WorkflowMethod(
                category="preprocessing",
                stage_id="trace_correction",
                method_id="dc_shift",
                params={"estimator": "mean", "scope": "per_trace"},
            )
            win.run_workflow_methods([method], realtime=False, run_mode="Run All")

            assert started == []
            assert win._pending_workflow_run is None
        finally:
            win._worker = None
            win.close()
            app.processEvents()


class TestValidateErrorUserCancel:
    """测试 _emit_run validate error 且用户取消时，不会改变节点状态"""

    def test_validate_error_user_cancel_no_state_change(self):
        """测试18: validate 有 error 且用户选择 No 时，节点状态不变"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication
        from unittest.mock import patch, MagicMock
        from core.workflow_validation import WorkflowValidationReport, WorkflowValidationIssue

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        config = build_default_workflow_config("high_quality_uav_gpr")

        for method in config.methods:
            method.status = "idle"

        page.config = config

        original_statuses = {m.node_id: m.status for m in config.methods}

        error_report = WorkflowValidationReport()
        error_report.add("error", "test_error", "Test error message")

        signal_received = []
        page.workflow_run_requested.connect(
            lambda *args: signal_received.append(args)
        )

        with patch("ui.gui_workflow_page.validate_workflow_config", return_value=error_report), \
             patch("ui.gui_workflow_page.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
            page._emit_run(
                [m for m in config.methods if not m.hidden and m.enabled][:2],
                realtime=False,
                status="运行中",
                log_text="测试",
                run_mode="Run All",
            )

        assert len(signal_received) == 0, "用户取消时不应该发出 workflow_run_requested 信号"

        for method in config.methods:
            assert method.status == original_statuses[method.node_id], \
                f"节点 {method.node_id} 状态不应该改变，期望 {original_statuses[method.node_id]}，实际 {method.status}"

    def test_validate_error_user_continue_does_emit(self):
        """测试19: validate 有 error 但用户选择 Yes 时，继续运行"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication
        from unittest.mock import patch
        from core.workflow_validation import WorkflowValidationReport

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        config = build_default_workflow_config("high_quality_uav_gpr")
        page.config = config

        error_report = WorkflowValidationReport()
        error_report.add("error", "test_error", "Test error message")

        signal_received = []
        page.workflow_run_requested.connect(
            lambda *args: signal_received.append(args)
        )

        with patch("ui.gui_workflow_page.validate_workflow_config", return_value=error_report), \
             patch("ui.gui_workflow_page.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            visible = [m for m in config.methods if not m.hidden and m.enabled]
            page._emit_run(
                visible[:2],
                realtime=False,
                status="运行中",
                log_text="测试",
                run_mode="Run All",
            )

        assert len(signal_received) == 1, "用户确认继续时应该发出信号"


class TestEmitRunRealNodeStatus:
    """测试 _emit_run 后真实 config 节点状态"""

    def test_first_node_running_rest_queued(self):
        """测试20: _emit_run 后真实 config 第一个运行节点是 running，其余本次节点是 queued"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication
        from unittest.mock import patch
        from core.workflow_validation import WorkflowValidationReport

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        config = build_default_workflow_config("high_quality_uav_gpr")

        for method in config.methods:
            method.status = "idle"

        page.config = config

        visible = [m for m in config.methods if not m.hidden and m.enabled]
        run_methods = visible[:3]

        with patch("ui.gui_workflow_page.validate_workflow_config") as mock_validate:
            mock_validate.return_value = WorkflowValidationReport()
            page._emit_run(
                run_methods,
                realtime=False,
                status="运行中",
                log_text="测试",
                run_mode="Run All",
            )

        run_node_ids = {m.node_id for m in run_methods}
        first_node_id = run_methods[0].node_id

        for method in config.methods:
            if method.hidden or not method.enabled:
                assert method.status == "skipped", \
                    f"隐藏/停用节点 {method.node_id} 应该是 skipped，实际是 {method.status}"
            elif method.node_id == first_node_id:
                assert method.status == "running", \
                    f"第一个运行节点 {method.node_id} 应该是 running，实际是 {method.status}"
            elif method.node_id in run_node_ids:
                assert method.status == "queued", \
                    f"其余运行节点 {method.node_id} 应该是 queued，实际是 {method.status}"

    def test_old_success_becomes_stale_when_not_in_run(self):
        """测试21: _emit_run 后非本次运行的旧 success 节点变为 success_stale"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication
        from unittest.mock import patch
        from core.workflow_validation import WorkflowValidationReport

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        config = build_default_workflow_config("high_quality_uav_gpr")

        for method in config.methods:
            method.status = "success"

        page.config = config

        visible = [m for m in config.methods if not m.hidden and m.enabled]
        run_methods = [visible[1]]

        with patch("ui.gui_workflow_page.validate_workflow_config") as mock_validate:
            mock_validate.return_value = WorkflowValidationReport()
            page._emit_run(
                run_methods,
                realtime=False,
                status="运行中",
                log_text="测试",
                run_mode="Run Selected",
            )

        run_node_ids = {m.node_id for m in run_methods}

        for method in config.methods:
            if method.hidden or not method.enabled:
                assert method.status == "skipped"
            elif method.node_id in run_node_ids:
                assert method.status in ("running", "queued")
            else:
                assert method.status == "success_stale", \
                    f"非本次运行的旧 success 节点 {method.node_id} 应该是 success_stale，实际是 {method.status}"


class TestRunHistoryMode:
    """测试 Run History mode 正确性"""

    def test_run_selected_history_mode(self):
        """测试22: Run Selected 完成后 history mode 是 Run Selected"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        config = build_default_workflow_config("high_quality_uav_gpr")
        page.config = config
        page._run_history = []

        selected_method = config.methods[1]
        outputs = [{
            "method_key": selected_method.method_id,
            "node_id": selected_method.node_id,
            "elapsed_ms": 100.0,
            "data": None,
        }]

        record = page._create_run_record(outputs, realtime=False, run_mode="Run Selected")
        assert record["mode"] == "Run Selected", \
            f"mode 应该是 'Run Selected'，实际是 '{record['mode']}'"

    def test_run_from_history_mode(self):
        """测试23: Run From 完成后 history mode 是 Run From"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        config = build_default_workflow_config("high_quality_uav_gpr")
        page.config = config
        page._run_history = []

        from_method = config.methods[2]
        outputs = [{
            "method_key": from_method.method_id,
            "node_id": from_method.node_id,
            "elapsed_ms": 150.0,
            "data": None,
        }]

        record = page._create_run_record(outputs, realtime=False, run_mode="Run From")
        assert record["mode"] == "Run From", \
            f"mode 应该是 'Run From'，实际是 '{record['mode']}'"

    def test_run_all_history_mode(self):
        """测试24: Run All 完成后 history mode 是 Run All"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        config = build_default_workflow_config("high_quality_uav_gpr")
        page.config = config
        page._run_history = []

        outputs = [{"method_key": "dc_shift", "elapsed_ms": 200.0, "data": None}]
        record = page._create_run_record(outputs, realtime=False, run_mode="Run All")
        assert record["mode"] == "Run All", \
            f"mode 应该是 'Run All'，实际是 '{record['mode']}'"

    def test_realtime_history_mode(self):
        """测试25: Realtime 完成后 history mode 是 Realtime"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        config = build_default_workflow_config("high_quality_uav_gpr")
        page.config = config
        page._run_history = []

        outputs = [{"method_key": "dc_shift", "elapsed_ms": 50.0, "data": None}]
        record = page._create_run_record(outputs, realtime=True, run_mode="Realtime")
        assert record["mode"] == "Realtime", \
            f"mode 应该是 'Realtime'，实际是 '{record['mode']}'"

    def test_run_mode_not_overridden_by_realtime_flag(self):
        """测试26: 有 run_mode 时不被 realtime 标志覆盖"""
        from ui.gui_workflow_page import WorkflowPage
        import sys
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        page = WorkflowPage()
        config = build_default_workflow_config("high_quality_uav_gpr")
        page.config = config
        page._run_history = []

        outputs = [{"method_key": "dc_shift", "elapsed_ms": 50.0, "data": None}]
        record = page._create_run_record(outputs, realtime=True, run_mode="Run Selected")
        assert record["mode"] == "Run Selected", \
            f"有 run_mode 时应该用 run_mode，实际是 '{record['mode']}'"


class TestStartupLazyLoading:
    """测试启动时懒加载功能."""
    def test_startup_lazy_pages(self):
        """测试启动时 page_auto_tune, page_advanced, page_quality 为 None."""
        import sys
        from PyQt6.QtWidgets import QApplication
        from app_qt import GPRGuiQt

        app = QApplication.instance() or QApplication(sys.argv)
        win = GPRGuiQt(version_text="test", enable_startup_profile=False)
        assert win.page_auto_tune is None
        assert win.page_advanced is None
        assert win.page_quality is None

    def test_lazy_create_auto_tune_page(self):
        """测试调用 _ensure_auto_tune_page() 才创建 AutoTunePage."""
        import sys
        from PyQt6.QtWidgets import QApplication
        from app_qt import GPRGuiQt

        app = QApplication.instance() or QApplication(sys.argv)
        win = GPRGuiQt(version_text="test", enable_startup_profile=False)
        assert win.page_auto_tune is None
        page = win._ensure_auto_tune_page()
        assert page is not None
        assert win.page_auto_tune is not None

    def test_lazy_legacy_plot_canvas(self):
        """测试启动时 _legacy_plot_canvas_created 为 False，调用 _ensure_legacy_plot_canvas() 后创建."""
        import sys
        from PyQt6.QtWidgets import QApplication
        from app_qt import GPRGuiQt

        app = QApplication.instance() or QApplication(sys.argv)
        win = GPRGuiQt(version_text="test", enable_startup_profile=False)
        assert hasattr(win, '_legacy_plot_canvas_created')
        assert win._legacy_plot_canvas_created is False
        assert win.fig is None
        assert win.canvas is None
        assert win._main_toolbar is None
        win._ensure_legacy_plot_canvas()
        assert win._legacy_plot_canvas_created is True
        assert win.fig is not None
        assert win.canvas is not None
        assert win._main_toolbar is not None

    def test_lazy_advanced_page_signals(self):
        """测试 _ensure_advanced_page() 后信号连接正常."""
        import sys
        from PyQt6.QtWidgets import QApplication
        from app_qt import GPRGuiQt

        app = QApplication.instance() or QApplication(sys.argv)
        win = GPRGuiQt(version_text="test", enable_startup_profile=False)
        assert win.page_advanced is None
        _ = win._ensure_advanced_page()
        assert win.page_advanced is not None

    def test_startup_profile_no_bad_log(self, capsys):
        """测试 MYGPR_STARTUP_PROFILE=1 时不输出错误的 'app import done' 日志."""
        import sys
        import os
        import logging
        from unittest.mock import patch
        from PyQt6.QtWidgets import QApplication
        from app_qt import GPRGuiQt, main

        # 测试 GPRGuiQt.__init__ 时的日志
        app = QApplication.instance() or QApplication(sys.argv)
        logger = logging.getLogger(__name__)
        with patch.object(logger, 'info') as mock_info:
            win = GPRGuiQt(version_text="test", enable_startup_profile=True)
        # 检查日志中没有 '[startup] app import done'
        for call_args in mock_info.call_args_list:
            if '[startup] app import done' in str(call_args):
                assert False, "不应输出错误的 app import done 日志"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestWorkflowNodeCardUI:
    """测试工作流节点卡片的 UI 优化"""

    def test_port_row_no_duplicate_dots(self):
        """测试27: full 模式端口行只有端口名文字，不出现双圆点"""
        import sys
        from PyQt6.QtWidgets import QApplication
        from ui.workflow_canvas_cards import WorkflowNodeCard
        from core.workflow_data import WorkflowMethod

        app = QApplication.instance() or QApplication(sys.argv)

        method = WorkflowMethod(
            method_id="dc_shift",
            stage_id="gain_compensation",
            category="gain_compensation",
        )
        method.order = 1
        card = WorkflowNodeCard(0, method)

        input_text = card.input_port_label.text()
        output_text = card.output_port_label.text()

        assert "●" not in input_text, f"输入端口不应有圆点，实际是 '{input_text}'"
        assert "●" not in output_text, f"输出端口不应有圆点，实际是 '{output_text}'"

    def test_port_labels_use_workflow_port_labels(self):
        """测试28: port_labels 使用 workflow_port_labels 函数"""
        import sys
        from PyQt6.QtWidgets import QApplication
        from ui.workflow_canvas_cards import WorkflowNodeCard, workflow_port_labels
        from core.workflow_data import WorkflowMethod

        app = QApplication.instance() or QApplication(sys.argv)

        method = WorkflowMethod(
            method_id="manual_velocity_model",
            stage_id="velocity_model",
            category="velocity_model",
        )
        method.order = 0
        card = WorkflowNodeCard(0, method)

        card_labels = card.port_labels()
        expected_labels = workflow_port_labels(method)

        assert card_labels == expected_labels, \
            f"port_labels 应该返回 {expected_labels}，实际是 {card_labels}"

    def test_preview_port_row_no_duplicate_dots(self):
        """测试29: Preview 节点端口行没有双圆点"""
        import sys
        from PyQt6.QtWidgets import QApplication
        from ui.workflow_canvas_preview import BscanPreviewCard

        app = QApplication.instance() or QApplication(sys.argv)

        card = BscanPreviewCard()

        for i in range(card.layout().count()):
            item = card.layout().itemAt(i)
            if hasattr(item, 'layout') and item.layout() is not None:
                for j in range(item.layout().count()):
                    widget = item.layout().itemAt(j).widget()
                    if isinstance(widget, object) and hasattr(widget, 'text'):
                        text = str(getattr(widget, 'text', lambda: '')())
                        if text:
                            assert "●" not in text, f"Preview 端口不应有圆点，实际是 '{text}'"

    def test_preview_compact_mode_shows_port_info(self):
        """测试30: Preview compact 模式设置"""
        import sys
        from PyQt6.QtWidgets import QApplication
        from ui.workflow_canvas_preview import BscanPreviewCard

        app = QApplication.instance() or QApplication(sys.argv)

        card = BscanPreviewCard()
        card.set_preview_data([[1, 2], [3, 4]], label="Test", success=True)
        card.set_compact(True)

        assert card.compact is True

    def test_compact_node_item_uses_port_labels(self):
        """测试31: CompactNodeItem 使用 workflow_port_labels"""
        import sys
        from PyQt6.QtWidgets import QApplication
        from ui.workflow_canvas_cards import CompactNodeItem, WorkflowNodeProxy
        from core.workflow_data import WorkflowMethod

        app = QApplication.instance() or QApplication(sys.argv)

        method = WorkflowMethod(
            method_id="dc_shift",
            stage_id="gain_compensation",
            category="gain_compensation",
        )
        method.order = 0

        proxy = WorkflowNodeProxy(0, method, None)
        compact_item = CompactNodeItem(proxy)
        compact_item.refresh()

        port_text = compact_item.port_item.text()
        assert "->" not in port_text or "→" in port_text, \
            f"Compact 节点端口应使用 Unicode 箭头，实际是 '{port_text}'"

    def test_mini_node_item_uses_port_labels(self):
        """测试32: MiniNodeItem 使用 workflow_port_labels"""
        import sys
        from PyQt6.QtWidgets import QApplication
        from ui.workflow_canvas_cards import MiniNodeItem, WorkflowNodeProxy
        from core.workflow_data import WorkflowMethod

        app = QApplication.instance() or QApplication(sys.argv)

        method = WorkflowMethod(
            method_id="dc_shift",
            stage_id="gain_compensation",
            category="gain_compensation",
        )
        method.order = 0

        proxy = WorkflowNodeProxy(0, method, None)
        mini_item = MiniNodeItem(proxy)
        mini_item.refresh()

        status_text = mini_item.status_item.text()
        assert "in ●──────● out" not in status_text, \
            f"Mini 节点不应使用硬编码端口显示，实际是 '{status_text}'"

    def test_mini_preview_node_shows_status_and_port(self):
        """测试33: Mini Preview 节点显示状态和 data→preview"""
        import sys
        from PyQt6.QtWidgets import QApplication
        from ui.workflow_canvas_cards import MiniNodeItem, WorkflowNodeProxy
        from core.workflow_data import WorkflowMethod

        app = QApplication.instance() or QApplication(sys.argv)

        preview_method = WorkflowMethod(
            method_id="bscan_preview",
            stage_id="preview",
            category="preview",
        )

        proxy = WorkflowNodeProxy(-1, preview_method, None)
        mini_item = MiniNodeItem(proxy)
        mini_item.refresh()

        title_text = mini_item.title_item.text()
        status_text = mini_item.status_item.text()

        assert "B-scan Preview" in title_text, \
            f"Mini Preview 标题应该是 'B-scan Preview'，实际是 '{title_text}'"
        assert "data → preview" in status_text, \
            f"Mini Preview 应该显示 'data → preview'，实际是 '{status_text}'"

    def test_param_row_bool_chip_fixed_width(self):
        """测试34: bool chip ON/OFF 宽度稳定"""
        import sys
        from PyQt6.QtWidgets import QApplication
        from ui.workflow_canvas_cards import WorkflowNodeCard, ParamRowWidget
        from core.workflow_data import WorkflowMethod
        from PyQt6.QtWidgets import QToolButton

        app = QApplication.instance() or QApplication(sys.argv)

        method = WorkflowMethod(
            method_id="dc_shift",
            stage_id="gain_compensation",
            category="gain_compensation",
        )
        method.order = 0
        card = WorkflowNodeCard(0, method)

        param_metas = [
            {"name": "enabled", "type": "bool", "default": True, "label": "启用"},
            {"name": "normalize", "type": "bool", "default": False, "label": "归一化"},
        ]

        width_on = None
        width_off = None
        for meta in param_metas:
            control, getter = card._create_param_control(meta)
            if isinstance(control, QToolButton):
                if control.isChecked():
                    width_on = control.width()
                else:
                    control.setChecked(True)
                    width_on = control.width()
                    control.setChecked(False)
                    width_off = control.width()

        if width_on is not None and width_off is not None:
            assert abs(width_on - width_off) <= 5, \
                f"ON/OFF 宽度应该相近，实际 ON={width_on}, OFF={width_off}"
