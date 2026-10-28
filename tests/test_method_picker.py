# -*- coding: utf-8 -*-
"""MethodPickerDialog 契约（P0-2）：非模态 Tool 窗 / 双击连续添加 /
搜索复位 / Esc / 外部点击关闭；处理页添加按钮与替换方法共用单实例。

offscreen 平台下真实焦点移交不可靠（架构共享知识 5.2）：外部点击关闭
走内部复核函数 ``_close_if_focus_left`` 契约，真机走查补外部点击。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QKeyEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ui.widgets.method_picker import MethodPickerDialog  # noqa: E402


def _methods() -> list:
    return [
        {'method_id': 'dewow', 'name': 'dewow', 'display_name': '去直流滤波',
         'category': 'filter', 'category_label': '滤波', 'tags': [],
         'parameter_schema': []},
        {'method_id': 'agc', 'name': 'agc', 'display_name': '自动增益控制',
         'category': 'gain', 'category_label': '增益', 'tags': [],
         'parameter_schema': []},
    ]


@pytest.fixture
def picker(qapp):
    w = MethodPickerDialog()
    w.set_methods(_methods())
    yield w
    w.close()
    w.deleteLater()


def _press_escape(widget) -> None:
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape,
                      Qt.KeyboardModifier.NoModifier)
    widget.keyPressEvent(event)


class TestPickerWindowContract:
    def test_non_modal_tool_window(self, picker):
        assert picker.isModal() is False
        assert picker.windowFlags() & Qt.WindowType.Tool

    def test_set_methods_forwards_to_browser(self, picker):
        assert len(picker._browser._methods) == 2
        assert picker._count_label.text() == '2 个方法'
        assert picker._title_label.text() == '添加处理步骤'
        assert '双击添加' in picker._hint_label.text()
        assert picker.minimumWidth() >= 520

    def test_reset_state_clears_search(self, picker):
        picker._browser._search.setText('agc')
        assert picker._browser._search.text() == 'agc'
        picker.reset_state()
        assert picker._browser._search.text() == ''
        # 过滤复位：全部分类重新可见
        assert picker._browser._tree.topLevelItemCount() == 2

    def test_search_filters_tree_in_realtime(self, picker, qapp):
        """搜索框输入实时过滤分类树（PRD A2③）：method_id 命中 + 显示名
        命中 + 清空恢复全量。"""
        tree = picker._browser._tree
        top_filter = tree.topLevelItem(0)     # 滤波（dewow）
        top_gain = tree.topLevelItem(1)       # 增益（agc）
        picker._browser._search.setText('agc')
        qapp.processEvents()
        assert top_gain.isHidden() is False
        assert top_gain.child(0).isHidden() is False
        assert top_filter.isHidden() is True    # dewow 不命中 → 整类隐藏
        # 显示名（中文）命中路径
        picker._browser._search.setText('去直')
        qapp.processEvents()
        assert top_filter.isHidden() is False
        assert top_filter.child(0).isHidden() is False
        assert top_gain.isHidden() is True
        # 清空 → 全量恢复
        picker._browser._search.setText('')
        qapp.processEvents()
        assert top_filter.isHidden() is False
        assert top_gain.isHidden() is False

    def test_show_event_resets_search(self, picker, qapp):
        picker.show()
        qapp.processEvents()
        picker._browser._search.setText('agc')
        picker.hide()
        picker.show()          # 再开 = 状态复位
        qapp.processEvents()
        assert picker._browser._search.text() == ''

    def test_escape_closes(self, picker, qapp):
        picker.show()
        qapp.processEvents()
        assert picker.isVisible()
        _press_escape(picker)
        assert not picker.isVisible()

    def test_focus_recheck_closes_when_focus_left(self, picker, qapp,
                                                  monkeypatch):
        """offscreen 下 focusWidget 非本窗后代 → 复核关闭（真机补点外部）。"""
        picker.show()
        qapp.processEvents()
        assert picker.isVisible()
        # offscreen 的 show 可能把焦点指派给窗内控件——这里钉的是
        # 「焦点在本窗之外必关」的契约，与平台焦点策略无关
        monkeypatch.setattr(QApplication, 'focusWidget',
                            classmethod(lambda cls: None))
        picker._close_if_focus_left()
        assert not picker.isVisible()

    def test_focus_recheck_keeps_open_when_focus_inside(self, picker, qapp):
        picker.show()
        qapp.processEvents()
        picker._browser._search.setFocus()
        qapp.processEvents()
        assert QApplication.focusWidget() is picker._browser._search
        picker._close_if_focus_left()
        assert picker.isVisible()


class TestPickerChooseFlow:
    def test_double_click_emits_and_stays_open(self, picker, qapp):
        """双击 → sig_method_chosen；不关窗，可连续添加（PRD Q1 定案）。"""
        got = []
        picker.sig_method_chosen.connect(got.append)
        picker.show()
        qapp.processEvents()
        # 等价双击路径：浏览器双击发 sig_add_requested → 弹窗转发
        picker._browser.sig_add_requested.emit('dewow')
        assert got == ['dewow']
        assert picker.isVisible()
        picker._browser.sig_add_requested.emit('agc')
        assert got == ['dewow', 'agc']
        assert picker.isVisible()

    def test_empty_method_id_ignored(self, picker):
        got = []
        picker.sig_method_chosen.connect(got.append)
        picker._browser.sig_add_requested.emit('')
        assert got == []

    def test_selection_updates_details_and_primary_action(self, picker, qapp):
        got = []
        picker.sig_method_chosen.connect(got.append)
        assert picker._add_selected_btn.isEnabled() is False
        picker.show()
        qapp.processEvents()
        item = picker._browser._tree.topLevelItem(0).child(0)
        picker._browser._tree.setCurrentItem(item)
        qapp.processEvents()
        assert picker._selected_name_label.text() == '去直流滤波'
        assert picker._add_selected_btn.isEnabled() is True
        picker._add_selected_btn.click()
        assert got == ['dewow']
        assert picker.isVisible()

    def test_enter_motion_uses_small_offset(self, picker, qapp, monkeypatch):
        from ui import motion

        monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
        picker.show()
        qapp.processEvents()
        animation = picker._geometry_animation
        assert animation.duration() == 180
        assert abs(animation.endValue().y() - animation.startValue().y()) == 8

    def test_apply_theme_is_idempotent(self, picker):
        """主窗 findChildren 遍历可能命中隐藏弹窗：纯令牌重刷无副作用。"""
        picker.apply_theme(True)
        picker.apply_theme(False)
        picker.apply_theme(True)


# ============================================================ 页面接线
class TestPagePickerWiring:
    @staticmethod
    def _page(qapp):
        from ui.pages.processing_page import ProcessingPage

        page = ProcessingPage()
        page.set_methods(_methods())
        return page

    def test_plus_opens_picker_singleton(self, qapp):
        """「+」→ 选择器；再点「+」= 唤起同一实例（防堆叠）；关窗后引用释放。"""
        page = self._page(qapp)
        try:
            assert page._method_picker is None
            page._open_method_picker()
            first = page._method_picker
            assert first is not None
            assert first.isVisible()
            first._browser._search.setText('agc')
            page._open_method_picker()          # 再点：同一实例 + 状态复位
            assert page._method_picker is first
            assert first._browser._search.text() == ''
            first.close()                        # 任何路径关闭 → 引用置 None
            assert page._method_picker is None
        finally:
            page.close()

    def test_add_button_reopens_fresh_picker_after_escape(self, qapp):
        """Retired chain-strip + is replaced by the workbench add button."""
        page = self._page(qapp)
        try:
            page._add_method_btn.click()
            first = page._method_picker
            first._browser.sig_add_requested.emit('dewow')
            _press_escape(first)
            assert page._method_picker is None
            page._add_method_btn.click()
            assert page._method_picker is not first
            assert page._method_picker._browser._search.text() == ''
            page._method_picker._browser.sig_add_requested.emit('agc')
            assert [s['method_id'] for s in page._pipeline_list.steps()] == ['dewow', 'agc']
        finally:
            page.close()

    def test_add_method_button_opens_same_picker(self, qapp):
        page = self._page(qapp)
        try:
            page._add_method_btn.click()
            assert page._method_picker is not None
            page._method_picker.close()
            assert page._method_picker is None
        finally:
            page.close()

    def test_replace_picker_reused_by_add_button_returns_to_append(self, qapp):
        """Button reuses the live picker but switches its intent to append."""
        page = self._page(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            page._open_method_picker(purpose='replace', index=0)
            first = page._method_picker
            page._add_method_btn.click()
            assert page._method_picker is first
            first._browser.sig_add_requested.emit('agc')
            assert [s['method_id'] for s in page._pipeline_list.steps()] == ['dewow', 'agc']
            assert first.isVisible()
        finally:
            page.close()

    def test_double_click_appends_to_chain(self, qapp):
        page = self._page(qapp)
        try:
            page._open_method_picker()
            page._method_picker._browser.sig_add_requested.emit('dewow')
            page._method_picker._browser.sig_add_requested.emit('agc')
            steps = page._pipeline_list.steps()
            assert [s['method_id'] for s in steps] == ['dewow', 'agc']
            # 选择器保持打开，可继续添加
            assert page._method_picker.isVisible()
        finally:
            page.close()

    def test_no_double_click_no_step(self, qapp):
        """未双击任何条目前不向链尾添加（G1：消除「复制上一个」错觉）。"""
        page = self._page(qapp)
        try:
            page._open_method_picker()
            assert page._pipeline_list.steps() == []
        finally:
            page.close()

    def test_set_methods_syncs_live_picker(self, qapp):
        page = self._page(qapp)
        try:
            page._open_method_picker()
            page.set_methods(_methods() + [{
                'method_id': 'bp', 'name': 'bp', 'display_name': '带通滤波',
                'category': 'filter', 'category_label': '滤波', 'tags': [],
                'parameter_schema': []}])
            assert len(page._method_picker._browser._methods) == 3
        finally:
            page.close()

    def test_replace_purpose_dispatch(self, qapp):
        """replace 分流：双击选择器 → _replace_step_method（T03 消费端）。"""
        page = self._page(qapp)
        try:
            page._add_method_to_pipeline('dewow')
            done: list = []
            page._replace_step_method = lambda i, m: done.append((i, m))
            page._open_method_picker(purpose='replace', index=0)
            page._method_picker._browser.sig_add_requested.emit('agc')
            assert done == [(0, 'agc')]
        finally:
            page.close()
