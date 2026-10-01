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
from PyQt6.QtCore import QObject, pyqtSignal  # noqa: E402
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
    class _StubWindow:
        _backend_ready = True
        backend_controller = None
        project_controller = None
        processing_controller = None
        interpretation_controller = None
        delivery_controller = None
        settings = None
        output_panel = None

        def _page(self, name):
            return None

        def _infobar(self, *args):
            return None

        def log_message(self, msg):
            return None

        def _goto_page(self, name):
            return None

        def _show_new_project_dialog(self):
            return None

        def _open_project_dialog(self):
            return None

        def _current_project_id(self):
            return None

        def _require_project(self):
            return False

        def _require_line(self):
            return ''

        def _job_bridge(self):
            return None

    pc = PageCoordinator(_StubWindow())
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


# ---------------------------------------------------------------- 方法库加载时序竞态
class _FakeProcessingController(QObject):
    """处理控制器桩：记录 load_methods 调用次数并即时回吐方法列表。

    用真 pyqtSignal（而非 list 记录）才能让「信号发出时有/无接收者」
    这个竞态本质被测到：无接收者时 emit 静默丢弃，方法库保持空。
    """

    methods_loaded = pyqtSignal(list)
    # 接线器 connect_all 会一次性接这五个信号；补全以免 AttributeError
    # 掩盖真正的竞态断言（本测试只关心 methods_loaded 有没有接收者）
    run_finished = pyqtSignal(object)
    autotune_finished = pyqtSignal(object)
    autotune_failed = pyqtSignal(str)
    velocity_finished = pyqtSignal(object)
    velocity_failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.methods = [{'method_id': 'agc', 'display_name': 'AGC 增益'}]

    def load_methods(self) -> None:
        self.load_calls += 1
        self.methods_loaded.emit(self.methods)


def _build_race_window(monkeypatch):
    """构造真主窗口（控制器置 None 不启后端线程）+ 假处理控制器。"""
    import ui.main_window as mw
    for name in ('BackendController', 'ProjectController',
                 'ProcessingController', 'InterpretationController',
                 'DeliveryController'):
        monkeypatch.setattr(mw, name, None)
    # 文件树打桩：同 tests/test_main_window_titlebar.py 注释——offscreen 下
    # qfluentwidgets TreeWidget 反复构造会触发库级 access violation。
    from PyQt6.QtWidgets import QWidget

    class _FileTreeStub(QWidget):
        line_selected = pyqtSignal(str)
        line_process_requested = pyqtSignal(str)
        line_delete_requested = pyqtSignal(list)
        delivery_focus_requested = pyqtSignal(str)
        artifact_focus_requested = pyqtSignal(str, str)
        artifact_delete_requested = pyqtSignal(str, str)

        def set_settings_manager(self, _settings):
            pass
        def apply_page(self, _name):
            pass
        def set_artifacts(self, _artifacts):
            pass
        def set_spatial_results(self, _results):
            pass
        def set_reports(self, _packages):
            pass

    monkeypatch.setattr(mw, 'FileTreePanel', _FileTreeStub)
    window = mw.MyGPRMainWindow()
    window.splashScreen.close()
    fake = _FakeProcessingController()
    # 接线器经 window.processing_controller 取控制器（property 透传窗口属性）
    window.processing_controller = fake
    return window, fake


class TestMethodsLoadedWarmupRace:
    """方法库空白回归（线上症状：方法库什么都不显示）。

    根因：`methods_loaded → 页面` 的接线在预热收尾 `_finish_warmup()`
    （需 8 页齐）才建立，而 `_on_backend_ready()` 无条件调
    `load_methods()`。后端若先于预热收尾就绪，信号发出时没有接收者，
    Qt 静默丢弃 → 方法库永久空白（且不会有任何报错）。
    """

    def test_backend_ready_before_warmup_still_fills_methods(
            self, monkeypatch, qapp):
        """后端先就绪：预热收尾必须补加载，方法库最终有内容。"""
        window, fake = _build_race_window(monkeypatch)
        try:
            window._on_backend_ready()      # 后端先到（预热未收尾）
            assert fake.load_calls == 0, (
                '未接线时若加载，emit 会发进空气：方法库将永久空白')

            window.ensure_pages_ready()     # 预热收尾 → 接线完成 → 补加载
            assert fake.load_calls == 1, (
                f'后端先就绪场景未补加载（方法库将空白）: {fake.load_calls}')
            page = window._page('processingInterface')
            assert page._methods == fake.methods, (
                '补加载后处理页方法库必须被填充')
        finally:
            window.close()
            window.deleteLater()
            qapp.processEvents()

    def test_backend_ready_after_warmup_loads_exactly_once(
            self, monkeypatch, qapp):
        """后端后就绪：走 _on_backend_ready 正常加载，不得与补加载重复。"""
        window, fake = _build_race_window(monkeypatch)
        try:
            window.ensure_pages_ready()
            assert fake.load_calls == 0, '接线完成但未就绪时不应提前加载'
            window._on_backend_ready()
            assert fake.load_calls == 1, (
                f'后端后就绪应恰好加载一次，实际 {fake.load_calls}')
            assert window._page('processingInterface')._methods == fake.methods
        finally:
            window.close()
            window.deleteLater()
            qapp.processEvents()
