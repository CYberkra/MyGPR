# -*- coding: utf-8 -*-
"""阶段 3 子接线器拆分后的针对性测试。

- JobHub：任务事件对 JobTable / MiniJobList×2 三视图的同构扇出
  （upsert/set_status/update_progress/remove_inactive）、取消语义、
  终态清理（prune 委派 BackendController，不旁路 backend 句柄）；
- ProjectChain：成果删除链（后代闭包 → ui.dialogs 确认框 → 提交删除）。

JobHub 部分无 Qt 依赖（后端 CI 可跑）；删除链因 ui.dialogs
依赖 qfluentwidgets，无 Qt 环境自动跳过。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.page_coordinator import PageCoordinator  # noqa: E402

# ui.dialogs 依赖 PyQt6/qfluentwidgets/pyqtgraph；缺失时仅跳过删除链用例
_HAS_QT = all(
    importlib.util.find_spec(module) is not None
    for module in ('PyQt6', 'qfluentwidgets', 'pyqtgraph')
)
needs_qt = pytest.mark.skipif(not _HAS_QT, reason='ui.dialogs 需要 Qt 环境')


class _FakeSignal:
    """记录并手动触发的假信号。"""

    def __init__(self) -> None:
        self._slots: list = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def emit(self, *args) -> None:
        for slot in list(self._slots):
            slot(*args)


class _StubView:
    """三视图同构协议（upsert_job/set_status/update_progress/remove_inactive）。"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def upsert_job(self, job_id: str, title: str) -> None:
        self.calls.append(('upsert_job', job_id, title))

    def set_status(self, job_id: str, status: str) -> None:
        self.calls.append(('set_status', job_id, status))

    def update_progress(self, job_id: str, completed: int, total: int,
                        message: str) -> None:
        self.calls.append(('update_progress', job_id, completed, total, message))

    def remove_inactive(self) -> None:
        self.calls.append(('remove_inactive',))


class _StubPage:
    """记录方法调用的假页面（鸭子类型，未注入方法返回记录器）。"""

    def __init__(self, **methods) -> None:
        self.calls: list[tuple] = []
        for name, impl in methods.items():
            def _wrapped(*args, _impl=impl, _name=name, **kwargs):
                self.calls.append((_name, args))
                return _impl(*args, **kwargs)
            setattr(self, name, _wrapped)

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args))
        return _record


class _StubBridge:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self._titles: dict[str, str] = {}

    def titles(self) -> dict[str, str]:
        return dict(self._titles)

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(str(job_id))


def _make_window(*, bridge=None, with_views=True):
    jobs_view = _StubView()
    log_view = _StubView()
    home_view = _StubView()

    jobs_page = _StubPage(
        job_table=lambda: jobs_view if with_views else None,
    )
    home_page = _StubPage(
        mini_jobs=lambda: home_view if with_views else None,
    )
    log_panel = _StubPage(
        mini_jobs=lambda: log_view if with_views else None,
        cancel_job_requested=_FakeSignal(),
    )
    pages = {
        'homeInterface': home_page,
        'projectInterface': _StubPage(),
        'processingInterface': _StubPage(),
        'interpretationInterface': _StubPage(),
        'deliveryInterface': _StubPage(),
        'spatialInterface': _StubPage(),
        'jobsInterface': jobs_page,
    }

    class _StubWindow:
        _backend_ready = True

        def __init__(self) -> None:
            self.infobars: list[tuple] = []
            self.logs: list[str] = []
            self.project_controller = _StubPage()
            self.processing_controller = _StubPage()
            self.interpretation_controller = None
            self.delivery_controller = None
            self.backend_controller = None
            self.log_panel = log_panel
            self.settings = None
            self._pages = pages
            self._bridge = bridge

        def _page(self, name):
            return self._pages.get(name)

        def _infobar(self, level, title, content, duration=None):
            self.infobars.append((level, title, content))

        def log_message(self, msg):
            self.logs.append(msg)

        def _goto_page(self, name):
            return None

        def _show_new_project_dialog(self):
            return None

        def _open_project_dialog(self):
            return None

        def _current_project_id(self):
            return 'P-001'

        def _require_project(self):
            return True

        def _require_line(self):
            return 'L01'

        def _job_bridge(self):
            return self._bridge

    window = _StubWindow()
    return window, pages, (jobs_view, log_view, home_view)


# ============================================================ JobHub 同构扇出
class TestJobHubFanOut:
    def test_status_upserts_and_fans_out_to_three_views(self):
        bridge = _StubBridge()
        bridge._titles['JOB-9'] = '导入测线 L01'
        window, _pages, views = _make_window(bridge=bridge)
        hub = PageCoordinator(window).jobs

        hub.on_status('JOB-9', 'running')

        for view in views:
            assert ('upsert_job', 'JOB-9', '导入测线 L01') in view.calls
            assert ('set_status', 'JOB-9', 'running') in view.calls

    def test_terminal_status_prunes_inactive_on_all_views(self):
        window, _pages, views = _make_window()
        hub = PageCoordinator(window).jobs

        hub.on_status('JOB-9', 'completed')

        for view in views:
            assert ('remove_inactive',) in view.calls

    def test_non_terminal_status_does_not_prune(self):
        window, _pages, views = _make_window()
        hub = PageCoordinator(window).jobs

        hub.on_status('JOB-9', 'queued')

        for view in views:
            assert ('remove_inactive',) not in view.calls

    def test_progress_fans_out_and_forwards_processing_page(self):
        window, pages, views = _make_window()
        hub = PageCoordinator(window).jobs
        hub._co.processing.processing_job_id = 'JOB-1'

        hub.on_progress('JOB-1', 3, 10, '处理中')

        for view in views:
            assert ('update_progress', 'JOB-1', 3, 10, '处理中') in view.calls
        processing_calls = [c for c in pages['processingInterface'].calls]
        assert ('set_progress', (3, 10, '处理中')) in processing_calls

    def test_completed_import_refreshes_lines(self):
        window, _pages, views = _make_window()
        hub = PageCoordinator(window).jobs
        hub.import_job_ids.add('JOB-IMP')

        hub.on_completed('JOB-IMP', True, '完成', None)

        pc_calls = window.project_controller.calls
        assert ('refresh_lines', ()) in pc_calls
        hub.import_job_ids.discard('JOB-IMP')  # 不残留
        assert 'JOB-IMP' not in hub.import_job_ids
        for view in views:
            assert ('set_status', 'JOB-IMP', 'completed') in view.calls

    def test_completed_spatial_refreshes_delivery(self):
        window, _pages, _views = _make_window()
        coordinator = PageCoordinator(window)
        hub = coordinator.jobs
        hub.spatial_job_ids.add('JOB-SPA')
        coordinator._win.delivery_controller = _StubPage()

        hub.on_completed('JOB-SPA', True, '完成', None)

        dc_calls = window.delivery_controller.calls
        assert ('refresh_spatial', ('P-001',)) in dc_calls
        assert 'JOB-SPA' not in hub.spatial_job_ids

    def test_failed_processing_job_does_not_double_infobar(self):
        """处理任务失败由 run_finished 单独弹窗：JobHub 不再重复 error。"""
        window, _pages, _views = _make_window()
        coordinator = PageCoordinator(window)
        coordinator.processing.processing_job_id = 'JOB-RUN'

        coordinator.jobs.on_completed('JOB-RUN', False, '爆炸', None)

        assert window.infobars == []
        assert any('JOB-RUN' in log for log in window.logs)

    def test_failed_other_job_shows_error_infobar(self):
        window, _pages, _views = _make_window()
        hub = PageCoordinator(window).jobs

        hub.on_completed('JOB-X', False, '失败', None)

        assert window.infobars[0][0] == 'error'

    def test_cancel_marks_processing_request(self):
        bridge = _StubBridge()
        window, _pages, _views = _make_window(bridge=bridge)
        coordinator = PageCoordinator(window)
        coordinator.processing.processing_job_id = 'JOB-RUN'

        coordinator.jobs.on_cancel('JOB-RUN')

        assert coordinator.processing.processing_cancel_requested is True
        assert bridge.cancelled == ['JOB-RUN']

    def test_views_filter_none_widgets(self):
        window, _pages, _views = _make_window(with_views=False)
        hub = PageCoordinator(window).jobs
        # None 视图被过滤：不抛异常、扇出为空
        assert hub._views() == ()
        hub.on_status('JOB-9', 'running')


# ============================================================ JobHub prune 委派
class TestJobHubPrune:
    def test_prune_without_backend_warns(self):
        window, _pages, _views = _make_window()
        hub = PageCoordinator(window).jobs

        hub.on_prune_jobs()

        assert window.infobars[0][0] == 'warning'

    def test_prune_delegates_backend_controller(self):
        class _BackendController:
            def __init__(self) -> None:
                self.pruned = False

            def prune_jobs(self) -> bool:
                self.pruned = True
                return True

        window, _pages, _views = _make_window()
        backend_controller = _BackendController()
        window.backend_controller = backend_controller
        hub = PageCoordinator(window).jobs

        hub.on_prune_jobs()

        assert backend_controller.pruned is True
        assert window.infobars == []


# ============================================================ 成果删除链
@needs_qt
class TestArtifactDeleteChain:
    def test_delete_requested_queries_descendants(self):
        window, _pages, _views = _make_window()
        project = PageCoordinator(window).project
        window.project_controller = _StubPage()

        project.on_artifact_delete_requested('L01', ['a1', 'a2'])

        calls = window.project_controller.calls
        assert ('get_artifact_descendants', ('L01', ['a1'])) in calls

    def test_descendants_confirmed_submits_delete(self, monkeypatch):
        from ui import dialogs
        window, _pages, _views = _make_window()
        deleted: list[tuple] = []
        window.project_controller = _StubPage(
            delete_artifacts=lambda line_id, ids: deleted.append((line_id, list(ids))))
        monkeypatch.setattr(dialogs, 'ask_artifact_delete', lambda parent, names: True)
        project = PageCoordinator(window).project

        project.on_artifact_descendants_ready(
            'L01', ['a1', 'child-1'], {'a1': '成果A', 'child-1': '子成果'})

        assert deleted == [('L01', ['a1', 'child-1'])]

    def test_descendants_cancelled_skips_delete(self, monkeypatch):
        from ui import dialogs
        window, _pages, _views = _make_window()
        deleted: list[tuple] = []
        window.project_controller = _StubPage(
            delete_artifacts=lambda line_id, ids: deleted.append((line_id, list(ids))))
        monkeypatch.setattr(dialogs, 'ask_artifact_delete', lambda parent, names: False)
        project = PageCoordinator(window).project

        project.on_artifact_descendants_ready('L01', ['a1'], {})

        assert deleted == []

    def test_empty_descendants_noop(self, monkeypatch):
        from ui import dialogs
        asked: list[bool] = []
        window, _pages, _views = _make_window()
        monkeypatch.setattr(
            dialogs, 'ask_artifact_delete',
            lambda parent, names: asked.append(True) or True)
        project = PageCoordinator(window).project

        project.on_artifact_descendants_ready('L01', [], {})

        assert asked == []  # 空闭包不弹确认框
