# -*- coding: utf-8 -*-
"""PageCoordinator 单链单元测试（任务 F 候选 1 的直接收益证明）。

信号链收敛到 PageCoordinator 后，每条链可在无 Qt 环境下用鸭子类型的
假窗口独立验证——这在接线分散在 window_mixins/main_window 时做不到。
本文件不导入 PyQt6，后端 CI 也能运行。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.page_coordinator import PageCoordinator  # noqa: E402


class _StubPage:
    """记录方法调用的假页面（鸭子类型满足 hasattr 探测）。

    注入的具名实现也会先记录调用再执行；未注入的方法由 __getattr__
    返回纯记录器。
    """

    def __init__(self, **methods):
        self.calls: list[tuple] = []
        for name, impl in methods.items():
            def _wrapped(*args, _impl=impl, _name=name, **kwargs):
                self.calls.append((_name, args))
                return _impl(*args, **kwargs)
            setattr(self, name, _wrapped)

    def __getattr__(self, name):  # 未注入的方法返回可调用记录器
        def _record(*args, **kwargs):
            self.calls.append((name, args))
        return _record


def _make_window(*, project_id="P-001", line_id="L02", has_project=True):
    pages = {
        "homeInterface": _StubPage(),
        "projectInterface": _StubPage(),
        "processingInterface": _StubPage(),
        "interpretationInterface": _StubPage(),
        "deliveryInterface": _StubPage(),
        "spatialInterface": _StubPage(),
        "jobsInterface": _StubPage(),
    }

    class _StubWindow:
        _backend_ready = True

        def __init__(self):
            self.infobars: list[tuple] = []
            self.logs: list[str] = []
            self.project_controller = _StubPage(
                preview_line=lambda *_: None,
                refresh_artifacts=lambda *_: None,
            ) if has_project else None
            self.processing_controller = _StubPage(
                run_pipeline=lambda *a, **kw: "JOB-1")
            self.interpretation_controller = None
            self.delivery_controller = None
            self.backend_controller = None
            self.log_panel = _StubPage()

            self._pages = pages

        def _page(self, name):
            return self._pages.get(name)

        def _infobar(self, level, title, content, duration=None):
            self.infobars.append((level, title, content))

        def log_message(self, msg):
            self.logs.append(msg)

        def _goto_page(self, name):
            self._pages.setdefault("__nav__", _StubPage())

        def _current_project_id(self):
            return project_id if has_project else None

        def _require_project(self):
            return has_project

        def _require_line(self):
            return line_id if (has_project and line_id) else ""

        def _job_bridge(self):
            return None

    return _StubWindow(), pages


def test_line_selected_updates_state_and_triggers_preview():
    window, _ = _make_window()
    coordinator = PageCoordinator(window)

    coordinator._on_line_selected("L03")

    assert coordinator._current_line_id == "L03"
    assert window.project_controller.calls == [
        ("preview_line", ("L03",)),
        ("refresh_artifacts", ("L03",)),
    ]
    # 处理页/解释页标签同步刷新
    processing_calls = [c for c in window._pages["processingInterface"].calls]
    assert ("set_line_label", ("L03",)) in processing_calls


def test_empty_line_id_is_ignored():
    window, _ = _make_window()
    coordinator = PageCoordinator(window)
    coordinator._on_line_selected("")
    assert coordinator._current_line_id == ""
    assert window.project_controller.calls == []


def test_run_requested_rejects_empty_pipeline_without_submit():
    window, _ = _make_window()
    coordinator = PageCoordinator(window)

    coordinator._on_run_requested({"steps": []})

    assert window.infobars[0][0] == "warning"
    # 未向 controller 提交任何任务
    assert window.processing_controller.calls == []


def test_run_requested_snapshots_line_and_marks_running():
    window, pages = _make_window()
    coordinator = PageCoordinator(window)

    coordinator._on_run_requested({"steps": [{"method_id": "dewow"}]})

    assert coordinator._processing_job_id == "JOB-1"
    assert coordinator._processing_line_id == "L02"
    assert coordinator._show_run_completion_notice is True
    assert ("set_running", (True, "JOB-1")) in pages["processingInterface"].calls


def test_run_finished_returns_to_submitted_line_not_current():
    window, _ = _make_window()
    coordinator = PageCoordinator(window)
    coordinator._on_run_requested({"steps": [{"method_id": "dewow"}]})
    # 运行期间用户切到别的测线
    coordinator._current_line_id = "L09"

    coordinator._on_run_finished(True, "")

    # 成果刷新回到提交时测线，且因不等于当前测线而"不"自动预览
    refresh_calls = [c for c in window.project_controller.calls
                     if c[0] == "refresh_artifacts"]
    assert refresh_calls == [("refresh_artifacts", ("L02",))]
    assert coordinator._preview_newest_artifact is False
    assert coordinator._processing_job_id == ""
