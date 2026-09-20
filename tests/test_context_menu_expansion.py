# -*- coding: utf-8 -*-
"""右键交互扩充批次回归测试（offscreen QApplication）。

覆盖：
- A InterpretationPage 标注点表右键：复制该点信息 / 删除选中点 / 清空全部
  （编辑动作与按钮同门控：会话已打开且不在忙态；右击行先选中）；
- B DeliveryPage 报告产物路径标签右键：打开文件 / 打开所在目录 / 复制路径
  （无产物不构造菜单）；
- C OutputPanel 日志区右键：复制（无选中禁用）/ 全选 / 清空 / 导出；
- D MiniJobList 右键：行上（打开任务中心/复制标题/取消）+ 空白区
  （打开任务中心 → open_job_center_requested）与 JobHub 接线。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py

from PyQt6.QtCore import QObject, QPoint, Qt, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


def _menu_actions(menu) -> dict:
    """RoundMenu 动作按文本索引（separator 无文本，跳过）。"""
    return {a.text(): a for a in menu.actions() if a.text()}


# ============================================================ JobHub 接线存根
class _MiniStub(QObject):
    """MiniJobList 信号面（真实 pyqtSignal 供 connect_all 接线）。"""
    cancel_requested = pyqtSignal(str)
    job_clicked = pyqtSignal(str)
    open_job_center_requested = pyqtSignal()


class _JobsPageStub(QObject):
    cancel_requested = pyqtSignal(str)
    prune_requested = pyqtSignal()

    def __init__(self, mini):
        super().__init__()
        self._mini = mini

    def job_table(self):
        return SimpleNamespace(focus_job=lambda jid: None)

    def mini_jobs(self):
        return self._mini


class _PanelStub(QObject):
    cancel_job_requested = pyqtSignal(str)

    def __init__(self, mini):
        super().__init__()
        self._mini = mini

    def mini_jobs(self):
        return self._mini


# ============================================================ A 标注点表右键
class TestPointsTableContextMenu:
    def _page(self, qapp):
        from ui.pages.interpretation_page import InterpretationPage
        page = InterpretationPage()
        page._points = [(0, 0), (5, 5)]
        page._refresh_points_table()
        return page

    def test_copy_point_info_without_bundle(self, qapp):
        """无 bundle（无时间轴）：只复制道号与采样点。"""
        page = self._page(qapp)
        actions = _menu_actions(page._build_points_menu(1))
        actions['复制该点信息'].trigger()
        assert QApplication.clipboard().text() == '道 6, 采样点 6'

    def test_copy_point_info_with_time_axis(self, qapp):
        """有 bundle 时间轴：附带双程走时。"""
        page = self._page(qapp)
        page._bundle = SimpleNamespace(sample_axis=[0.0, 100.0],
                                       sample_count=101)
        actions = _menu_actions(page._build_points_menu(1))
        actions['复制该点信息'].trigger()
        assert QApplication.clipboard().text() == '道 6, 采样点 6, 5.00 ns'

    def test_blank_row_omits_point_actions(self, qapp):
        page = self._page(qapp)
        page.set_session_active(True)   # 隔离门控因素（门控由专门用例覆盖）
        actions = _menu_actions(page._build_points_menu(-1))
        assert '复制该点信息' not in actions
        # 空白右击未选中行：删除动作禁用（无目标）
        assert not actions['删除选中点'].isEnabled()
        assert actions['清空全部'].isEnabled()

    def test_edit_actions_gated_by_session_and_busy(self, qapp):
        page = self._page(qapp)
        # 未开会话：编辑动作禁用（复制不受限）
        actions = _menu_actions(page._build_points_menu(0))
        assert not actions['删除选中点'].isEnabled()
        assert not actions['清空全部'].isEnabled()
        assert actions['复制该点信息'].isEnabled()
        # 开会话后启用；忙态再次禁用
        page.set_session_active(True)
        actions = _menu_actions(page._build_points_menu(0))
        assert actions['删除选中点'].isEnabled()
        page.set_busy(True)
        actions = _menu_actions(page._build_points_menu(0))
        assert not actions['删除选中点'].isEnabled()
        assert not actions['清空全部'].isEnabled()

    def test_delete_action_targets_selected_row(self, qapp):
        """真实路径：右击行先 selectRow，删除动作作用于该行。"""
        page = self._page(qapp)
        page.set_session_active(True)
        page._points_table.selectRow(0)   # _on_points_context_menu 的选中步骤
        emitted = []
        page.points_changed.connect(emitted.append)
        _menu_actions(page._build_points_menu(0))['删除选中点'].trigger()
        assert page._points == [(5, 5)]
        assert emitted == [[(5, 5)]]

    def test_clear_action_emits_points_changed(self, qapp):
        page = self._page(qapp)
        page.set_session_active(True)
        emitted = []
        page.points_changed.connect(emitted.append)
        _menu_actions(page._build_points_menu(-1))['清空全部'].trigger()
        assert page._points == []
        assert emitted == [[]]

    def test_context_menu_policy_wired(self, qapp):
        page = self._page(qapp)
        table = page._points_table
        assert (table.contextMenuPolicy()
                == Qt.ContextMenuPolicy.CustomContextMenu)
        assert table.receivers(table.customContextMenuRequested) >= 1


# ============================================================ B 报告路径标签右键
class TestReportLabelContextMenu:
    def _page(self, qapp):
        from ui.pages.delivery_page import DeliveryPage
        return DeliveryPage()

    def test_labels_have_context_menu_policy(self, qapp):
        page = self._page(qapp)
        for label in page._report_path_labels.values():
            assert (label.contextMenuPolicy()
                    == Qt.ContextMenuPolicy.CustomContextMenu)
            assert label.receivers(label.customContextMenuRequested) >= 1

    def test_menu_actions_with_path(self, qapp):
        page = self._page(qapp)
        page.set_report_result({'pdf_path': '/tmp/r.pdf'})
        actions = _menu_actions(page._build_report_menu('pdf_path'))
        assert actions['打开文件'].isEnabled()
        assert actions['打开所在目录'].isEnabled()
        actions['复制路径'].trigger()
        assert QApplication.clipboard().text() == '/tmp/r.pdf'

    def test_open_containing_dir(self, qapp, monkeypatch):
        from ui.pages import delivery_page
        opened = []
        monkeypatch.setattr(
            delivery_page.QDesktopServices, 'openUrl',
            staticmethod(lambda url: opened.append(url)))
        page = self._page(qapp)
        page.set_report_result({'pdf_path': '/tmp/sub/r.pdf'})
        page._open_containing_dir('pdf_path')
        assert len(opened) == 1
        assert opened[0].toLocalFile() == '/tmp/sub'

    def test_no_path_skips_menu(self, qapp, monkeypatch):
        """无产物：右键不构造菜单（避免弹全禁用菜单）。"""
        from ui.pages import delivery_page
        built = []
        real_make_menu = delivery_page.make_menu
        monkeypatch.setattr(
            delivery_page, 'make_menu',
            lambda parent=None: (built.append(1), real_make_menu(parent))[1])
        page = self._page(qapp)
        page._on_report_label_context_menu('pdf_path', QPoint(0, 0))
        assert built == []


# ============================================================ C 日志区右键
class TestLogContextMenu:
    def _panel(self, qapp):
        from ui.widgets.output_panel import OutputPanel
        return OutputPanel()

    def test_menu_actions(self, qapp):
        panel = self._panel(qapp)
        panel.append_log('INFO hello')
        actions = _menu_actions(panel._build_log_menu())
        assert not actions['复制'].isEnabled()   # 无选中
        assert actions['全选'].isEnabled()
        assert actions['清空日志'].isEnabled()
        assert actions['导出日志到文件'].isEnabled()

    def test_copy_after_select_all(self, qapp):
        panel = self._panel(qapp)
        panel.append_log('INFO hello world')
        actions = _menu_actions(panel._build_log_menu())
        actions['全选'].trigger()
        actions2 = _menu_actions(panel._build_log_menu())
        assert actions2['复制'].isEnabled()   # 全选后可复制
        actions2['复制'].trigger()
        assert 'hello world' in QApplication.clipboard().text()

    def test_clear_log_action(self, qapp):
        panel = self._panel(qapp)
        panel.append_log('INFO x')
        _menu_actions(panel._build_log_menu())['清空日志'].trigger()
        assert panel._log_edit.toPlainText() == ''
        assert len(panel._entries) == 0


# ============================================================ D MiniJobList 右键
class TestMiniJobListContextMenu:
    def _mini(self, qapp):
        from ui.widgets.job_widgets import MiniJobList
        mini = MiniJobList()
        mini.upsert_job('j1', '导入测线')
        return mini

    def test_row_menu_actions(self, qapp):
        mini = self._mini(qapp)
        mini.set_status('j1', 'running')
        clicked, cancelled = [], []
        mini.job_clicked.connect(clicked.append)
        mini.cancel_requested.connect(cancelled.append)
        actions = _menu_actions(mini._build_context_menu('j1'))
        assert actions['打开任务中心'].isEnabled()
        assert actions['取消任务'].isEnabled()
        actions['复制任务标题'].trigger()
        assert QApplication.clipboard().text() == '导入测线'
        actions['打开任务中心'].trigger()
        assert clicked == ['j1']
        actions['取消任务'].trigger()
        assert cancelled == ['j1']

    def test_row_menu_cancel_disabled_when_finished(self, qapp):
        mini = self._mini(qapp)
        mini.set_status('j1', 'completed')
        actions = _menu_actions(mini._build_context_menu('j1'))
        assert not actions['取消任务'].isEnabled()

    def test_blank_menu_only_open_job_center(self, qapp):
        mini = self._mini(qapp)
        actions = _menu_actions(mini._build_context_menu(None))
        assert list(actions) == ['打开任务中心']
        opened = []
        mini.open_job_center_requested.connect(lambda: opened.append(True))
        actions['打开任务中心'].trigger()
        assert opened == [True]

    def test_job_id_at_miss(self, qapp):
        """空白坐标（越界）不命中任何任务行。"""
        mini = self._mini(qapp)
        assert mini._job_id_at(QPoint(5000, 5000)) is None

    def test_context_menu_policy_wired(self, qapp):
        mini = self._mini(qapp)
        assert (mini.contextMenuPolicy()
                == Qt.ContextMenuPolicy.CustomContextMenu)
        assert mini.receivers(mini.customContextMenuRequested) >= 1


class TestJobHubOpenJobCenter:
    def test_open_job_center_goto_only(self, qapp):
        from ui.coordinator_jobs import JobHub
        gotos = []
        co = SimpleNamespace(goto_page=gotos.append)
        JobHub(co).on_open_job_center()
        assert gotos == ['jobsInterface']   # 仅跳页，不 focus 定位

    def test_connect_all_wires_open_job_center(self, qapp):
        """两个迷你视图（输出面板 + 主页）的 open_job_center 都接到 JobHub。"""
        from ui.coordinator_jobs import JobHub
        mini_out, mini_home = _MiniStub(), _MiniStub()
        gotos = []
        co = SimpleNamespace(
            page=lambda name: jobs_page,
            output_panel=_PanelStub(mini_out),
            goto_page=gotos.append)
        jobs_page = _JobsPageStub(mini_home)
        # hub 须持有引用：PyQt 信号连接不阻止接收方被 GC，临时对象会被静默断连
        hub = JobHub(co)
        hub.connect_all()
        mini_out.open_job_center_requested.emit()
        mini_home.open_job_center_requested.emit()
        assert gotos == ['jobsInterface', 'jobsInterface']
