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

from ui.main_window import MyGPRMainWindow
from ui.page_coordinator import PageCoordinator


def test_job_bridge_slot_targets_exist_on_coordinator() -> None:
    """_on_backend_ready 依赖的三个槽位必须存在于 PageCoordinator。"""
    for slot in ('_on_job_progress', '_on_job_status', '_on_job_completed'):
        assert hasattr(PageCoordinator, slot), (
            f'PageCoordinator 缺少 {slot}：_on_backend_ready 的 hasattr 探测会静默'
            '跳过连接，导致 load_methods 不执行（方法库为空）'
        )


def test_coordinator_line_state_attribute_exists() -> None:
    """_require_line 委托读取的状态名必须与 coordinator 一致。"""
    assert hasattr(PageCoordinator, '__init__')
    # 通过源码级检查：coordinator 初始化里声明了该属性
    import inspect
    src = inspect.getsource(PageCoordinator.__init__)
    assert '_current_line_id' in src, (
        'PageCoordinator.__init__ 不再持有 _current_line_id：'
        '窗口 _require_line 的委托读取会失效'
    )


def test_window_require_line_delegates_to_coordinator() -> None:
    """_require_line 必须读 coordinator 的当前测线，而非窗口自身属性。"""
    calls: list[str] = []

    class _FakeCoordinator:
        _current_line_id = 'L07'

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

    notices: list[tuple] = []

    window = MyGPRMainWindow.__new__(MyGPRMainWindow)
    window.page_coordinator = _FakeCoordinator()
    window._require_project = lambda: True  # type: ignore[assignment]
    window._infobar = lambda *a: notices.append(a)  # type: ignore[assignment]

    assert MyGPRMainWindow._require_line(window) == ''
    assert notices, '无测线时应给出提示而非静默返回'
