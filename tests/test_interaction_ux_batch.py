# -*- coding: utf-8 -*-
"""交互体验批次（A/B/C）回归测试（offscreen QApplication）。

覆盖：
- A1 JobTable：apply_theme 空态标签崩溃修复、标题/消息 tooltip、
  右键菜单动作（复制标题/消息、取消、清理）、focus_job 定位；
- A2 MiniJobList：任务行点击发 job_clicked；JobHub 跳页定位接线；
- A3 DeliveryPage：报告产物路径单开文件按钮；
- B2 InterpretationPage：标注点表 Delete 快捷键；
- C1 DeliveryPage 空间成果表：双击/右键「设为当前成果」+ 复制；
- DeliveryController.set_current_spatial 命令成功/失败路径。
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py

from PyQt6.QtCore import QCoreApplication, QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


def _await(emitted: list, timeout_s: float = 5.0) -> None:
    """等异步 worker 的信号回包（无 pytest-qt，轮询 processEvents）。"""
    deadline = time.monotonic() + timeout_s
    while not emitted and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def _menu_actions(menu) -> dict:
    """RoundMenu 动作按文本索引（separator 无文本，跳过）。"""
    return {a.text(): a for a in menu.actions() if a.text()}


# ============================================================ A1 JobTable
class TestJobTableInteraction:
    def _make_table(self, qapp):
        from ui.widgets.job_widgets import JobTable
        table = JobTable()
        table.upsert_job('j1', '导入测线 L01')
        return table

    def test_apply_theme_restyles_without_crash(self, qapp):
        """回归：空态标签曾是局部变量，apply_theme 引用 self._empty_label 崩。"""
        table = self._make_table(qapp)
        table.apply_theme(False)
        table.apply_theme(True)
        assert table._empty_label.text() == '暂无任务'

    def test_title_and_message_tooltips(self, qapp):
        table = self._make_table(qapp)
        table.update_progress('j1', 3, 10, '正在读取 3/10')
        title_item = table._table.item(0, table._COL_TITLE)
        message_item = table._table.item(0, table._COL_MESSAGE)
        assert title_item.toolTip() == '导入测线 L01'
        assert message_item.toolTip() == '正在读取 3/10'

    def test_context_menu_copy_error_message(self, qapp):
        """失败任务：右键「复制消息」把错误信息放进剪贴板。"""
        table = self._make_table(qapp)
        table.update_progress('j1', 0, 0, '后端爆炸：disk full')
        table.set_status('j1', 'failed')
        menu = table._build_context_menu('j1', 0)
        actions = _menu_actions(menu)
        assert actions['复制消息'].isEnabled()
        actions['复制消息'].trigger()
        assert QApplication.clipboard().text() == '后端爆炸：disk full'
        # 终态任务不可再取消
        assert not actions['取消任务'].isEnabled()

    def test_context_menu_cancel_active_job(self, qapp):
        table = self._make_table(qapp)
        table.set_status('j1', 'running')
        cancelled = []
        table.cancel_requested.connect(cancelled.append)
        menu = table._build_context_menu('j1', 0)
        actions = _menu_actions(menu)
        assert actions['取消任务'].isEnabled()
        actions['取消任务'].trigger()
        assert cancelled == ['j1']

    def test_context_menu_blank_area_only_cleanup(self, qapp):
        table = self._make_table(qapp)
        menu = table._build_context_menu(None, -1)
        assert list(_menu_actions(menu)) == ['清理已完成']

    def test_focus_job_selects_and_scrolls(self, qapp):
        table = self._make_table(qapp)
        table.upsert_job('j2', '生成报告')
        table.focus_job('j2')
        assert table._table.currentRow() == 1
        table.focus_job('missing')   # 不存在的 id：静默不崩
        assert table._table.currentRow() == 1


# ============================================================ A2 MiniJobList 点击跳任务页
class TestMiniJobListClick:
    def _click(self, widget):
        event = QMouseEvent(
            QEvent.Type.MouseButtonRelease, QPointF(2, 2),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier)
        return event

    def test_row_click_emits_job_clicked(self, qapp):
        from ui.widgets.job_widgets import MiniJobList
        mini = MiniJobList()
        mini.upsert_job('j1', '任务一')
        clicked = []
        mini.job_clicked.connect(clicked.append)
        row_widget = mini._jobs['j1']['widget']
        mini.eventFilter(row_widget, self._click(row_widget))
        assert clicked == ['j1']

    def test_row_has_pointer_cursor_and_hint(self, qapp):
        from ui.widgets.job_widgets import MiniJobList
        mini = MiniJobList()
        mini.upsert_job('j1', '任务一')
        row_widget = mini._jobs['j1']['widget']
        assert row_widget.cursor().shape() == Qt.CursorShape.PointingHandCursor
        assert '任务中心' in row_widget.toolTip()

    def test_unrelated_widget_click_emits_nothing(self, qapp):
        from ui.widgets.job_widgets import MiniJobList
        mini = MiniJobList()
        mini.upsert_job('j1', '任务一')
        clicked = []
        mini.job_clicked.connect(clicked.append)
        mini.eventFilter(mini, self._click(mini))   # 非任务行
        assert clicked == []


class TestJobHubMiniClickWiring:
    def test_click_goes_to_jobs_page_and_focuses(self, qapp):
        from ui.coordinator_jobs import JobHub
        calls = []
        table = SimpleNamespace(
            focus_job=lambda jid: calls.append(('focus', jid)))
        co = SimpleNamespace(
            goto_page=lambda name: calls.append(('goto', name)),
            page=lambda name: SimpleNamespace(job_table=lambda: table))
        JobHub(co).on_mini_job_clicked('j1')
        assert calls == [('goto', 'jobsInterface'), ('focus', 'j1')]


# ============================================================ A3/C1 DeliveryPage
class TestDeliveryReportOpenButtons:
    def test_open_buttons_follow_paths(self, qapp):
        from ui.pages.delivery_page import DeliveryPage
        page = DeliveryPage()
        result = {'pdf_path': '/tmp/r.pdf', 'html_path': '',
                  'xlsx_path': '/tmp/r.xlsx', 'delivery_zip_path': ''}
        page.set_report_result(result)
        assert page._report_open_btns['pdf_path'].isEnabled()
        assert page._report_open_btns['xlsx_path'].isEnabled()
        assert not page._report_open_btns['html_path'].isEnabled()
        assert not page._report_open_btns['delivery_zip_path'].isEnabled()

    def test_open_file_opens_url(self, qapp, monkeypatch):
        from ui.pages import delivery_page
        from ui.pages.delivery_page import DeliveryPage
        opened = []
        monkeypatch.setattr(
            delivery_page.QDesktopServices, 'openUrl',
            staticmethod(lambda url: opened.append(url)))
        page = DeliveryPage()
        page.set_report_result({'pdf_path': '/tmp/r.pdf'})
        page._report_open_btns['pdf_path'].click()
        assert len(opened) == 1
        assert opened[0].toLocalFile() == '/tmp/r.pdf'


class TestDeliverySpatialTable:
    def _page(self, qapp):
        from ui.pages.delivery_page import DeliveryPage
        page = DeliveryPage()
        page.set_spatial_results([
            {'result_id': 'R1', 'name': '全场拼接', 'line_ids': ['L01'],
             'created_at': '2026-01-01'},
            {'name': '无ID成果', 'line_count': 2},
        ])
        return page

    def test_results_retained_for_row_ops(self, qapp):
        page = self._page(qapp)
        assert len(page._spatial_results) == 2
        assert page._spatial_table.rowCount() == 2

    def test_activate_emits_result_id(self, qapp):
        page = self._page(qapp)
        emitted = []
        page.set_current_spatial_requested.connect(emitted.append)
        page._activate_spatial_row(0)
        assert emitted == ['R1']

    def test_activate_without_id_is_noop(self, qapp):
        page = self._page(qapp)
        emitted = []
        page.set_current_spatial_requested.connect(emitted.append)
        page._activate_spatial_row(1)
        page._activate_spatial_row(99)
        assert emitted == []

    def test_activate_blocked_when_busy(self, qapp):
        page = self._page(qapp)
        page.set_busy(True)
        emitted = []
        page.set_current_spatial_requested.connect(emitted.append)
        page._activate_spatial_row(0)
        assert emitted == []

    def test_double_click_and_enter_wired(self, qapp):
        """双击与 Enter(activated) 都指向行激活（全站约定）。"""
        page = self._page(qapp)
        table = page._spatial_table
        assert table.receivers(table.itemDoubleClicked) >= 1
        assert table.receivers(table.activated) >= 1

    def test_context_menu_actions(self, qapp):
        page = self._page(qapp)
        menu = page._build_spatial_menu(page._spatial_results[0], 0)
        actions = _menu_actions(menu)
        assert actions['设为当前成果'].isEnabled()
        assert actions['复制名称'].isEnabled()
        assert actions['复制成果 ID'].isEnabled()
        actions['复制成果 ID'].trigger()
        assert QApplication.clipboard().text() == 'R1'

    def test_context_menu_no_id_disables_id_actions(self, qapp):
        page = self._page(qapp)
        menu = page._build_spatial_menu(page._spatial_results[1], 1)
        actions = _menu_actions(menu)
        assert not actions['设为当前成果'].isEnabled()
        assert not actions['复制成果 ID'].isEnabled()
        assert actions['复制名称'].isEnabled()


# ============================================================ C1 DeliveryController 命令
class TestSetCurrentSpatialCommand:
    def _controller(self, backend):
        from ui.controllers.delivery_controller import DeliveryController
        controller = DeliveryController()
        controller.set_backend(SimpleNamespace(backend=backend,
                                               job_bridge=None))
        return controller

    def test_success_emits_changed(self, qapp):
        calls = []
        backend = SimpleNamespace(spatial=SimpleNamespace(
            set_current=lambda pid, rid: calls.append((pid, rid))))
        controller = self._controller(backend)
        changed = []
        controller.spatial_current_changed.connect(changed.append)
        controller.set_current_spatial('P1', 'R1')
        _await(changed)
        assert calls == [('P1', 'R1')]
        assert changed == ['R1']

    def test_failure_logs_and_emits_nothing(self, qapp):
        def _boom(pid, rid):
            raise RuntimeError('store locked')

        backend = SimpleNamespace(
            spatial=SimpleNamespace(set_current=_boom))
        controller = self._controller(backend)
        logs = []
        controller.log_message.connect(logs.append)
        changed = []
        controller.spatial_current_changed.connect(changed.append)
        controller.set_current_spatial('P1', 'R1')
        _await(logs)
        assert changed == []
        assert any('设置当前空间成果失败' in m for m in logs)


# ============================================================ B2 解译页标注点 Delete
class TestInterpretationPointsDelete:
    def _page_with_points(self, qapp):
        from ui.pages.interpretation_page import InterpretationPage
        page = InterpretationPage()
        page._points = [(0, 0), (5, 5)]
        page._refresh_points_table()
        return page

    def test_delete_shortcut_wired(self, qapp):
        page = self._page_with_points(qapp)
        shortcut = page._delete_point_shortcut
        assert shortcut.receivers(shortcut.activated) >= 1

    def test_delete_selected_point(self, qapp):
        page = self._page_with_points(qapp)
        page._points_table.selectRow(0)
        page._on_remove_selected_point()
        assert page._points == [(5, 5)]
        # 行号重排：幸存行的 # 列从 1 重新编号
        assert page._points_table.item(0, 0).text() == '1'
