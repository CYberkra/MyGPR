# -*- coding: utf-8 -*-
"""任务 F 候选 1 迁移遗留断链的回归测试（不构造完整窗口）。

两次线上症状同源：状态/槽位迁移到 PageCoordinator 后，窗口侧仍按旧位置
引用；Qt 信号槽会静默吞掉槽内异常，表现为"点了没反应"而非报错。

1. `_on_backend_ready` 用 hasattr 探测 JobBridge 三个槽——若被改名/移走，
   探测失败但方法继续走完，`load_methods` 不再执行 → 方法库为空、
   "应用到选中步骤"永久禁用。
2. `_require_line` 必须委托 coordinator 的 `_current_line_id`——读窗口自身
   属性会 AttributeError 并被槽吞掉 → "运行处理链"无反应。
"""
from __future__ import annotations

import types

import pytest

# PageCoordinator 本身无 Qt 依赖（后端 CI 可跑本文件的 coordinator 侧断言）；
# 窗口侧断言需要 PyQt6，无 Qt 环境自动跳过（CONTRIBUTING 规则 11）。
from ui.page_coordinator import PageCoordinator

pytest.importorskip("PyQt6", reason="窗口侧断言需要 PyQt6")
from ui.main_window import MyGPRMainWindow  # noqa: E402


def test_job_bridge_slot_targets_exist_on_coordinator() -> None:
    """_on_backend_ready 依赖的三个槽位必须存在于 PageCoordinator。"""
    for slot in ('_on_job_progress', '_on_job_status', '_on_job_completed'):
        assert hasattr(PageCoordinator, slot), (
            f'PageCoordinator 缺少 {slot}：显式接线会立刻 AttributeError 暴露'
        )


def test_connect_job_bridge_pins_all_three_signals() -> None:
    """钉住接线类（变异测试曾证明改名槽位后旧 hasattr 写法全绿）：三次 connect 必须发生。"""
    class _FakeSignal:
        def __init__(self, log, name: str) -> None:
            self._log, self._name = log, name
        def connect(self, slot) -> None:
            self._log.append(self._name)

    class _FakeBridge:
        def __init__(self) -> None:
            self.connected: list[str] = []
        def __getattr__(self, name: str):
            return _FakeSignal(self.connected, name)

    bridge = _FakeBridge()
    pc = PageCoordinator.__new__(PageCoordinator)   # 无需 __init__（纯接线方法）
    pc.connect_job_bridge(bridge)
    assert bridge.connected == ['progress_changed', 'status_changed', 'job_completed'], (
        'JobBridge 接线不完整：任务进度/状态/完成信号漏接会导致任务页无反应'
    )


def test_coordinator_line_state_attribute_exists() -> None:
    """coordinator 必须保留 _current_line_id 状态并经 current_line_id() 暴露。"""
    pc = PageCoordinator.__new__(PageCoordinator)
    pc._current_line_id = 'L03'
    assert pc.current_line_id() == 'L03'


def test_window_require_line_delegates_to_coordinator() -> None:
    """_require_line 必须读 coordinator 的当前测线，而非窗口自身属性。"""
    calls: list[str] = []

    class _FakeCoordinator:
        _current_line_id = 'L07'
        def current_line_id(self) -> str:
            return self._current_line_id

    window = MyGPRMainWindow.__new__(MyGPRMainWindow)
    window.page_coordinator = _FakeCoordinator()

    def _fake_require_project(self) -> bool:
        calls.append('require_project')
        return True

    window._require_project = types.MethodType(_fake_require_project, window)  # type: ignore[method-assign]

    assert MyGPRMainWindow._require_line(window) == 'L07'
    assert calls == ['require_project']


def test_window_require_line_empty_without_line_and_no_attribute_error() -> None:
    """无测线时返回 '' 并提示，不得抛 AttributeError。"""
    class _FakeCoordinator:
        _current_line_id = ''
        def current_line_id(self) -> str:
            return self._current_line_id

    notices: list[tuple] = []

    window = MyGPRMainWindow.__new__(MyGPRMainWindow)
    window.page_coordinator = _FakeCoordinator()
    window._require_project = lambda: True  # type: ignore[assignment]
    window._infobar = lambda *a: notices.append(a)  # type: ignore[assignment]

    assert MyGPRMainWindow._require_line(window) == ''
    assert notices, '无测线时应给出提示而非静默返回'
