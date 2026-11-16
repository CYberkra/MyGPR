# -*- coding: utf-8 -*-
"""处理页 v2 组件契约：ChainStrip 手势 + 处理页工作区行为。

点卡与链条 chip 双向同步高亮；处理页各 tab 的来源与管道状态以
ProcessingPage 的公开 API 为准。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from PyQt6.QtCore import QPointF  # noqa: E402
from ui.pages.processing_page import ProcessingPage  # noqa: E402
from ui.widgets.chain_strip import ChainStrip  # noqa: E402


class _FakeDropEvent:
    def __init__(self, x: float, y: float):
        self._pos = QPointF(x, y)
        self.accepted = False
        self.action = None

    def position(self):
        return self._pos

    def setDropAction(self, action):
        self.action = action

    def accept(self):
        self.accepted = True


class TestChainStrip:
    @pytest.fixture
    def strip(self, qapp):
        w = ChainStrip()
        w.set_steps([{'label': '去直达波', 'enabled': True},
                     {'label': 'SEC 增益', 'enabled': True},
                     {'label': '带通', 'enabled': False}])
        return w

    def test_chips_match_steps(self, strip):
        """v2.2：无「输入」chip——每个步骤一个 chip（测线=首 chip 由
        宿主注入，不在 list 内）。"""
        assert strip._list.count() == 3

    def test_selection_maps_directly(self, strip):
        """v2.2：row 即步骤索引（无输入 chip 偏移）。"""
        got = []
        strip.sig_step_selected.connect(got.append)
        strip._list.setCurrentRow(1)
        assert got == [1]

    def test_toggle_and_remove_signals(self, strip):
        toggled, removed = [], []
        strip.sig_step_toggled.connect(lambda i, e: toggled.append((i, e)))
        strip.sig_step_removed.connect(removed.append)
        strip._on_dot_clicked(1)              # 第 2 步：True → False
        strip._on_chip_deleted(0)
        assert toggled == [(1, False)]
        assert removed == [0]

    def test_drop_event_maps_to_step_indices(self, strip, qapp):
        got = []
        strip.show()                          # 几何需真实布局
        qapp.processEvents()
        strip.sig_step_moved.connect(lambda s, t: got.append((s, t)))
        strip._list.setCurrentRow(0)          # 步骤 0
        item = strip._list.item(2)            # 步骤 2
        rect = strip._list.visualItemRect(item)
        strip._list_drop_event(_FakeDropEvent(rect.left() + 2, rect.center().y()))
        assert got and got[0][0] == 0         # 源 = 步骤 0

    def test_select_step_moves_highlight(self, strip):
        strip.select_step(2)
        assert strip._list.currentRow() == 2


def _bundle(tag: float, samples: int = 8, traces: int = 6):
    """鸭子类型 PreviewBundle（矩阵全为 tag）。"""
    import numpy as np
    from types import SimpleNamespace
    return SimpleNamespace(
        matrix=np.full((samples, traces), float(tag), dtype=np.float32),
        vmin=0.0, vmax=float(tag), title=f'b{tag}', x_label='道数',
        y_label='采样点', trace_axis_m=None, sample_axis=None,
        sample_axis_label='', trace_count=traces, sample_count=samples,
        trace_elevation_m=None, depth_axis_m=None)


def _saved_artifact(aid, method, group, step):
    from types import SimpleNamespace
    return SimpleNamespace(
        artifact_id=aid, line_id='L01', name=f'{method} {aid}', method_id=method,
        created_at='2026-10-16T10:00:00',
        manifest={'params': {'artifact_kind': 'processing', 'run_group_id': group,
                             'run_step_index': step, 'lineage_version': 2,
                             'run_input_artifact_id': '', 'step_input_artifact_id': ''}})


class TestWorkspaceLazyPrinciple:
    """有数据才建画布：运行前不摆空 B-Scan 窗口。"""

    def test_page_before_run_opens_only_raw_in_one_group(self, qapp):
        page = ProcessingPage()
        try:
            assert page._workspace.tab_keys() == []
            assert len(page._workspace.views()) == 1
            page.set_original_bundle(_bundle(1))
            assert page._workspace.tab_keys() == ['original']
            assert page._workspace.primary_view()._matrix is page._sources['original']['bundle'].matrix
            assert page.current_pipeline()['steps'] == []
        finally:
            page.close()

    def test_catalog_results_stay_closed_until_explicitly_selected(self, qapp):
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle(1))
            page.set_artifacts([_saved_artifact('S1', 'dewow', 'G1', 1),
                                _saved_artifact('F1', 'sec_gain', 'G1', 2)])
            assert page._workspace.tab_keys() == ['original']
            page.select_artifact('S1')
            page.select_artifact('F1')
            assert page._workspace.tab_keys() == ['original', 'artifact:S1', 'artifact:F1']
            assert page.current_pipeline()['input_artifact_id'] == 'F1'
            assert len(page._workspace.views()) == 1
        finally:
            page.close()


class TestRunButtonMotion:
    """运行钮动效（移植自 Transitions.dev 的 spinner → ✓ 形变思路）。"""

    def test_running_flips_button_to_cancel(self, qapp):
        """运行钮就近翻转（2026-10-09 深查 P2）：运行中同位置 = 取消可点，
        红描边走 error 语义令牌；结束恢复「运行」与 primary 默认样式。
        """
        from ui.design_tokens import color

        strip = ChainStrip()
        strip.set_running(True)
        assert strip.run_button().text() == '取消'
        assert strip.run_button().isEnabled() is True
        assert color('error') in strip.run_button().styleSheet()
        strip.set_running(False)
        assert strip.run_button().text() == '运行'
        assert strip.run_button().styleSheet() == ''

    def test_flash_success_then_restores(self, qapp):
        """成功反馈为「完成」文字——全应用不出现勾形元素（用户定案）。"""
        strip = ChainStrip()
        strip.flash_success()
        assert strip.run_button().text() == '完成'
        assert '✓' not in strip.run_button().text()
        strip.run_button().setText('运行')      # 1.2s 后由定时器复位
        assert strip.run_button().text() == '运行'

    def test_dot_toggle_is_round_not_check(self, qapp):
        """启用开关 = 圆点（非勾形图标）：点击发 toggle 信号。"""
        strip = ChainStrip()
        strip.set_steps([{'label': 'a', 'enabled': True}])
        toggled = []
        strip.sig_step_toggled.connect(lambda i, e: toggled.append((i, e)))
        chip = strip._list.itemWidget(strip._list.item(0))
        chip.dot_btn.click()
        assert toggled == [(0, False)]


class TestChainSlidingPill:
    """链条选中滑动胶囊（动效移植②，来自 BeUI Tabs 思路）。"""

    def test_pill_appears_on_selection(self, qapp):
        strip = ChainStrip()
        strip.set_steps([{'label': '去直达波', 'enabled': True},
                         {'label': 'SEC 增益', 'enabled': True}])
        assert strip._pill.isHidden()
        strip.select_step(0)
        assert strip._pill.isVisibleTo(strip)

    def test_pill_slides_to_selected_chip(self, qapp, monkeypatch):
        from ui import motion
        monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
        strip = ChainStrip()
        strip.set_steps([{'label': '去直达波', 'enabled': True},
                         {'label': 'SEC 增益', 'enabled': True}])
        strip.show()
        qapp.processEvents()
        strip.select_step(0)
        strip.select_step(1)                      # 第二次选中 → 走动画
        target = strip._list.visualItemRect(strip._list.item(1))
        assert strip._pill_anim.endValue().x() == target.x()
        strip._pill_anim.setCurrentTime(strip._pill_anim.duration())
        assert strip._pill.geometry() == target
        strip.close()
        strip.deleteLater()

    def test_pill_hidden_when_no_row(self, qapp):
        strip = ChainStrip()
        strip._move_pill(-1)
        assert strip._pill.isHidden()


class TestStatusSemantics:
    """状态与反馈：最终结果 ✓ / 选中卡幅值范围 / 改动提示。"""

    def test_saved_result_tab_uses_resource_identity_without_final_gimmick(self, qapp):
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle(1))
            page.set_artifacts([_saved_artifact('F1', 'agc', 'G1', 1)])
            page.select_artifact('F1')
            bar = page._workspace.tab_bar()
            assert 'F1' in bar.tabText(bar.currentIndex())
            assert '✓' not in bar.tabText(bar.currentIndex())
            assert page.current_pipeline()['steps'] == []
        finally:
            page.close()


    def test_chain_dirty_label_toggles(self, qapp):
        strip = ChainStrip()
        assert not strip._dirty_label.isVisibleTo(strip)
        strip.set_dirty(True)
        assert strip._dirty_label.isVisibleTo(strip)
        strip.set_dirty(False)
        assert not strip._dirty_label.isVisibleTo(strip)

    def test_draft_edits_preserve_saved_result_and_visible_matrix(self, qapp):
        page = ProcessingPage()
        try:
            result = _saved_artifact('F1', 'agc', 'G1', 1)
            page.set_original_bundle(_bundle(1))
            page.set_artifacts([result])
            preview = _bundle(4)
            page.set_artifact_bundle('F1', preview)
            page.select_artifact('F1')
            page._pipeline_list.add_step('dewow', '去直流', {'win': 11})
            assert page._workspace.primary_view()._matrix is preview.matrix
            assert page._artifacts_by_id['F1'] is result
            assert page.current_pipeline()['steps'][0]['method_id'] == 'dewow'
            assert page.current_pipeline()['input_artifact_id'] == 'F1'
        finally:
            page.close()


class TestDraftSelectionIsolation:
    """结果过期（链已改未重算）时解除 chip↔卡片索引联动。

    旧 run 的卡片索引对不上新链（删/移步骤后整体移位），继续按索引
    联动会高亮错位的卡（真机走查：chip 亮 bandpass 卡亮 sec_gain）。
    """

    @staticmethod
    def _page_with_run(qapp):
        page = ProcessingPage()
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([_saved_artifact('F1', 'agc', 'G1', 1)])
        page.select_artifact('F1')
        return page

    def test_selected_saved_result_starts_with_empty_new_steps(self, qapp):
        page = self._page_with_run(qapp)
        try:
            assert page.current_pipeline()['steps'] == []
            assert page.current_pipeline()['input_artifact_id'] == 'F1'
            assert page._workspace.active_key() == 'artifact:F1'
        finally:
            page.close()

    def test_selecting_draft_step_does_not_change_input(self, qapp):
        page = self._page_with_run(qapp)
        try:
            page._pipeline_list.add_step('dewow', '去直流', {})
            page._pipeline_list.add_step('agc', 'AGC', {})
            page._pipeline_list.select_step(0)
            assert page._workspace.active_key() == 'artifact:F1'
            assert page.current_pipeline()['input_artifact_id'] == 'F1'
            page._pipeline_list.select_step(1)
            assert page.current_pipeline()['input_artifact_id'] == 'F1'
        finally:
            page.close()

    def test_switching_saved_result_restores_its_own_draft(self, qapp):
        page = self._page_with_run(qapp)
        try:
            page._pipeline_list.add_step('dewow', '去直流', {'win': 11})
            page.select_original()
            assert page.current_pipeline()['steps'] == []
            page._pipeline_list.add_step('agc', 'AGC', {'window': 7})
            page.select_artifact('F1')
            assert page.current_pipeline()['steps'][0]['params'] == {'win': 11}
            page.select_original()
            assert page.current_pipeline()['steps'][0]['params'] == {'window': 7}
        finally:
            page.close()

    def test_display_change_never_appends_processing_steps(self, qapp):
        page = self._page_with_run(qapp)
        try:
            before = page.current_pipeline()
            page._workspace.primary_view().set_display_mode('envelope')
            assert page.current_pipeline() == before
            assert page._workspace.active_key() == 'artifact:F1'
        finally:
            page.close()


class TestCardClose:
    """关闭仅影响视图；原始输入与成果采用相同生命周期。"""

    @staticmethod
    def _page_with_run(qapp):
        from types import SimpleNamespace
        page = ProcessingPage()
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([
            SimpleNamespace(
                artifact_id='F1', line_id='L01', name='run_agc',
                method_id='agc', created_at='2026-09-28T10:02:00',
                manifest={'params': {'artifact_kind': 'processing',
                                     'run_group_id': 'G1'}}),
        ])
        return page


    def test_closed_result_stays_closed_after_catalog_and_new_run_refresh(self, qapp):
        page = self._page_with_run(qapp)
        try:
            page.select_artifact('F1')
            page.close_artifact_tab('F1')
            old = _saved_artifact('F1', 'agc', 'G1', 1)
            page.set_artifacts([old])
            assert page._workspace.tab_keys() == ['original']
            page.set_artifacts([old, _saved_artifact('F2', 'agc', 'G2', 1)])
            assert page._workspace.tab_keys() == ['original']
            assert {'artifact:F1', 'artifact:F2'} <= set(page._sources)
            page.select_artifact('F1')
            assert page.current_pipeline()['input_artifact_id'] == 'F1'
        finally:
            page.close()

    def test_raw_tab_closes_and_reopens_without_implicit_replacement(self, qapp):
        page = self._page_with_run(qapp)
        try:
            page._workspace.close_source('original')
            assert page._workspace.tab_keys() == []
            assert page._workspace.active_key() == ''
            assert not page._run_btn.isEnabled()
            page.select_original()
            assert page._workspace.tab_keys() == ['original']
            assert page.current_pipeline()['input_artifact_id'] == ''
        finally:
            page.close()


class TestParamEmptyHint:
    """The parameter pane only reports missing fields for a selected step."""

    def test_hint_toggles_with_form_schema(self, qapp):
        page = ProcessingPage()
        try:
            parent = page._param_empty_hint.parentWidget()
            assert not page._param_empty_hint.isVisibleTo(parent)
            page._methods_by_id['agc'] = {'parameter_schema': [
                {'name': 'window', 'label': '窗口', 'type': 'int',
                 'default': 5, 'min': 1, 'max': 100}]}
            page._pipeline_list.add_step('agc', 'AGC', {})
            assert not page._param_empty_hint.isVisibleTo(parent)
            page._pipeline_list._list.setCurrentRow(-1)
            assert not page._param_empty_hint.isVisibleTo(parent)
            page._pipeline_list.add_step('without_params', '无参数算法', {})
            assert page._param_empty_hint.isVisibleTo(parent)
        finally:
            page.close()


class TestChainAlternativePaths:
    """拖拽的替代路径（guidelines：拖拽需有点击/键盘替代）。"""

    def test_delete_key_removes_selected_step(self, qapp):
        """Delete 键发信号；数据由宿主页改（本测试模拟宿主回写）。"""
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        strip = ChainStrip()
        steps = [{'label': 'a', 'enabled': True},
                 {'label': 'b', 'enabled': True},
                 {'label': 'c', 'enabled': True}]
        strip.set_steps(steps)
        strip.sig_step_removed.connect(
            lambda i: strip.set_steps(steps[:i] + steps[i + 1:]))
        strip.show()                       # 快捷键需活动窗口才触发
        strip.select_step(1)
        strip._list.setFocus()
        qapp.processEvents()
        QTest.keyClick(strip._list, Qt.Key.Key_Delete)
        assert strip._list.count() == 2            # 剩 2 步
        assert [s['label'] for s in strip._steps] == ['a', 'c']

    def test_context_menu_requires_explicit_step_editor_action(self, qapp):
        strip = ChainStrip()
        strip.set_steps([{'label': '一'}, {'label': '二'}])
        strip.resize(700, 50)
        strip.show()
        qapp.processEvents()
        strip.select_step(0)
        got = []
        strip.sig_step_editor_requested.connect(got.append)
        rect = strip._list.visualItemRect(strip._list.item(1))
        menu = strip._context_menu(rect.center())
        assert not got and strip._list.currentRow() == 0
        next(action for action in menu.actions() if action.data() == 'edit-step').trigger()
        assert got == [1] and strip._list.currentRow() == 0
        menu.deleteLater()
        strip.close()

    def test_fill_step_menu_removed(self, qapp):
        """旧四动作右键菜单已删除（动作收敛进合一弹窗，不维护两套入口）。"""
        strip = ChainStrip()
        assert not hasattr(strip, '_fill_step_menu')


class TestExplicitSourceOpening:
    """「全部步骤」开关：关=只铺输入 + 最终结果（P3 就地版）。"""

    @staticmethod
    def _page_with_run(qapp):
        from types import SimpleNamespace
        page = ProcessingPage()
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([
            SimpleNamespace(
                artifact_id='S1', line_id='L01', name='run 步骤1_dewow',
                method_id='dewow', created_at='2026-09-27T10:01:00',
                manifest={'params': {'artifact_kind': 'intermediate',
                                     'run_group_id': 'G1'}}),
            SimpleNamespace(
                artifact_id='S2', line_id='L01', name='run 步骤2_agc',
                method_id='agc', created_at='2026-09-27T10:02:00',
                manifest={'params': {'artifact_kind': 'intermediate',
                                     'run_group_id': 'G1'}}),
            SimpleNamespace(
                artifact_id='F1', line_id='L01', name='run_bandpass',
                method_id='bandpass', created_at='2026-09-27T10:03:00',
                manifest={'params': {'artifact_kind': 'processing',
                                     'run_group_id': 'G1'}}),
        ])
        return page

    def test_catalog_never_implicitly_expands_all_steps(self, qapp):
        page = self._page_with_run(qapp)
        try:
            assert page._workspace.tab_keys() == ['original']
            assert {'artifact:S1', 'artifact:S2', 'artifact:F1'} <= set(page._sources)
            assert len(page._workspace.views()) == 1
        finally:
            page.close()

    def test_explicit_open_keeps_every_selected_step_in_single_group(self, qapp):
        page = self._page_with_run(qapp)
        try:
            for artifact in ('S1', 'S2', 'F1'):
                page.select_artifact(artifact)
            assert page._workspace.tab_keys() == ['original', 'artifact:S1', 'artifact:S2', 'artifact:F1']
            assert len(page._workspace.views()) == 1
            assert page.current_pipeline()['input_artifact_id'] == 'F1'
        finally:
            page.close()


class TestHistoricalTabsPreserved:
    """总览墙只显示当前上下文（真机反馈：跑一步蹦出一屏几天前的成果）。"""

    @staticmethod
    def _artifacts():
        from types import SimpleNamespace
        return [
            SimpleNamespace(
                artifact_id='OLD1', line_id='L09', name='run_old1',
                method_id='dewow', created_at='2026-09-23T10:07:12',
                manifest={'params': {'artifact_kind': 'intermediate',
                                     'run_group_id': 'G_OLD'}}),
            SimpleNamespace(
                artifact_id='NEW1', line_id='L09', name='run_new1',
                method_id='set_zero_time', created_at='2026-09-27T15:20:06',
                manifest={'params': {'artifact_kind': 'processing',
                                     'run_group_id': 'G_NEW'}}),
        ]

    def test_new_run_catalog_keeps_explicitly_opened_historical_result(self, qapp):
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle(1))
            old, new = self._artifacts()
            page.set_artifacts([old])
            page.select_artifact('OLD1')
            page.set_artifacts([old, new])
            assert page._workspace.tab_keys() == ['original', 'artifact:OLD1']
            assert page.current_pipeline()['input_artifact_id'] == 'OLD1'
            page.select_artifact('NEW1')
            assert page._workspace.tab_keys() == ['original', 'artifact:OLD1', 'artifact:NEW1']
        finally:
            page.close()


class TestColorbarPrefForwarding:
    """设置页「显示色标」对 v2 处理页生效（真机反馈开关失效）。"""

    def test_page_forwards_pref_to_all_workspace_groups(self, qapp):
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle(1))
            page._workspace.split_source('original', 'right', copy=True)
            page.set_colorbar_pref(False)
            assert all(not view.colorbar_visible() for view in page._workspace.all_views())
            page.set_colorbar_pref(True)
            assert all(view.colorbar_visible() for view in page._workspace.all_views())
        finally:
            page.close()


class TestAuditFixes20261009:
    """处理页审计修复回归（2026-10-09，commit 范围：开关改名/卡序/总览墙
    入口/AutoTune 展开反馈/主题令牌/搜索无匹配浮层）。"""

    # ---------------- P1-A：「中间步骤」开关改名 ----------------

    # ---------------- P1-B：左栏卡序（执行卡上移） ----------------
    @staticmethod
    def _direct_child(widget, layout):
        while widget is not None:
            parent = widget.parentWidget()
            if parent is not None and parent.layout() is layout:
                return widget
            widget = parent
        return None

    def test_next_input_and_run_control_remain_above_parameter_tools(self, qapp):
        from PyQt6.QtCore import QPoint
        page = ProcessingPage()
        try:
            page.resize(1100, 700)
            page.set_original_bundle(_bundle(1))
            page.show()
            qapp.processEvents()
            assert '原始输入' in page._input_label.text()
            assert page._input_label.mapTo(page, QPoint()).y() < page._param_form.mapTo(page, QPoint()).y()
            assert page._run_btn.isVisibleTo(page)
        finally:
            page.close()

    # ---------------- P2-C：总览墙手动入口（挂结果网格头部） ----------------
    def test_manual_layout_changes_only_views_and_preserves_draft(self, qapp):
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle(1))
            page._pipeline_list.add_step('dewow', '去直流', {'win': 11})
            before = page.current_pipeline()
            page._workspace.split_source('original', 'right', copy=True)
            assert len(page._workspace.views()) == 2
            assert page.current_pipeline() == before
            page._workspace.reset_layout()
            assert len(page._workspace.views()) == 1
            assert page.current_pipeline() == before
        finally:
            page.close()

    # ---------------- P2-E：AutoTune 结果到达自动展开 ----------------
    def test_autotune_result_expands_card(self, qapp):
        """调参是异步的：结果写进默认收起的卡等于没有反馈——到达时展开。"""
        page = ProcessingPage()
        try:
            assert page._autotune_card.is_collapsed() is True
            page.set_line_label('L01')
            page.set_original_bundle(_bundle(1))
            page._pipeline_list.add_step('dewow', '去直流', {'win': 3})
            page._on_autotune_clicked()
            page.set_autotune_result('dewow', {'best_params': {'win': 5}})
            assert page._autotune_card.is_collapsed() is False
        finally:
            page.close()

    # ---------------- P2-D：chip 状态点走令牌 ----------------
    def test_chip_dot_uses_theme_tokens(self, qapp):
        """原硬编码 #7CC464（绿）/ #5A5A56（灰）不随主题且违反跳色纪律。"""
        from ui.design_tokens import color

        strip = ChainStrip()
        strip.set_steps([{'label': '去直流', 'enabled': True},
                         {'label': 'AGC', 'enabled': False}])
        try:
            chip_on = strip._list.itemWidget(strip._list.item(0))
            chip_off = strip._list.itemWidget(strip._list.item(1))
            assert color('success') in chip_on.dot_btn.styleSheet()
            assert color('disabled') in chip_off.dot_btn.styleSheet()
            assert '#7CC464' not in chip_on.dot_btn.styleSheet()
        finally:
            strip.deleteLater()

    def test_page_apply_theme_restyles_hint(self, qapp):
        """参数空态提示原硬编码 'color: gray'；apply_theme 后须为深色
        主题的 text_muted 令牌值。"""
        from ui.design_tokens import color

        page = ProcessingPage()
        try:
            page.apply_theme(dark=True)
            assert color('text_muted', True) in (
                page._param_empty_hint.styleSheet())
        finally:
            page.close()

    def test_chain_strip_apply_theme_restrokes_chips(self, qapp):
        """主题切换后逐 chip 重刷（set_enabled_visual 按当时主题取令牌）。"""
        from ui.design_tokens import color

        strip = ChainStrip()
        strip.set_steps([{'label': 'AGC', 'enabled': False}])
        try:
            strip.apply_theme(dark=True)
            chip = strip._list.itemWidget(strip._list.item(0))
            assert color('disabled', True) in chip.dot_btn.styleSheet()
        finally:
            strip.deleteLater()

    # ---------------- P2-F：搜索无匹配浮层 + 清除按钮 ----------------
    def test_search_nomatch_overlay(self, qapp):
        from ui.widgets.method_browser import MethodBrowser

        browser = MethodBrowser()
        try:
            browser.set_methods([{
                'method_id': 'dewow', 'name': 'dewow',
                'display_name': '去直流滤波', 'category': 'filter',
                'category_label': '滤波', 'tags': [],
                'parameter_schema': []}])
            assert browser._nomatch.isVisibleTo(browser) is False

            browser._apply_filter('不存在的关键词')
            assert browser._nomatch.isVisibleTo(browser) is True
            browser._apply_filter('去直流')
            assert browser._nomatch.isVisibleTo(browser) is False
            browser._apply_filter('')
            assert browser._nomatch.isVisibleTo(browser) is False
            # 空库不叠浮层：「暂无方法」引导已覆盖该态
            browser.set_methods([])
            browser._apply_filter('任意词')
            assert browser._nomatch.isVisibleTo(browser) is False
            assert browser._search.isClearButtonEnabled() is True
        finally:
            browser.deleteLater()


class TestDeepAudit20261009:
    """第二轮深查（工作流走查 + 状态矩阵探针）落地项。"""

    def _strip(self):
        from ui.widgets.chain_strip import ChainStrip
        return ChainStrip()

    def _page_with_run(self, display_name=None):
        """输入 + 单步 run_group（与既有测试同构的鸭子类型成果）。"""
        from types import SimpleNamespace

        from ui.pages.processing_page import ProcessingPage
        page = ProcessingPage()
        if display_name:
            page.set_methods([{
                'method_id': 'dewow', 'name': 'dewow',
                'display_name': display_name, 'category': 'filter',
                'category_label': '滤波', 'tags': [],
                'parameter_schema': []}])
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([SimpleNamespace(
            artifact_id='S1', line_id='L01', name='run 步骤1',
            method_id='dewow', created_at='2026-09-25T10:01:00',
            manifest={'params': {'artifact_kind': 'intermediate',
                                 'run_group_id': 'G1'}})])
        return page

    # ---------------- P1：链 chip 溢出可达（+N 胶囊） ----------------
    def test_overflow_pill_shown_when_chips_clipped(self, qapp):
        strip = self._strip()
        try:
            strip.resize(360, 40)
            strip.show()
            strip.set_steps([{'label': f'm{i}', 'enabled': True}
                             for i in range(8)])
            qapp.processEvents()
            strip._update_overlays()
            assert strip._overflow_btn.isVisibleTo(strip) is True
            text = strip._overflow_btn.text()
            assert text.startswith('+') and text[1:].isdigit()
            assert 1 <= int(text[1:]) <= 8
        finally:
            strip.deleteLater()

    def test_overflow_pill_hidden_when_fits(self, qapp):
        strip = self._strip()
        try:
            strip.resize(900, 40)
            strip.show()
            strip.set_steps([{'label': 'a', 'enabled': True}])
            qapp.processEvents()
            strip._update_overlays()
            assert strip._overflow_btn.isVisibleTo(strip) is False
        finally:
            strip.deleteLater()

    def test_overflow_menu_builds_one_submenu_per_step(self, qapp):
        strip = self._strip()
        try:
            strip.set_steps([{'label': f'm{i}', 'enabled': i != 1}
                             for i in range(3)])
            menu = strip._build_overflow_menu()
            # 每步骤一个子菜单（builder 显式收藏，qfw 未暴露清单）
            assert len(menu._step_submenus) == 3
        finally:
            strip.deleteLater()

    def test_overflow_menu_select_reaches_hidden_step(self, qapp):
        """选中动作走 setCurrentRow → sig_step_selected（真实信号路径）。"""
        strip = self._strip()
        try:
            strip.set_steps([{'label': f'm{i}', 'enabled': True}
                             for i in range(3)])
            got = []
            strip.sig_step_selected.connect(got.append)
            menu = strip._build_overflow_menu()
            sub = menu._step_submenus[0]
            sub.actions()[0].trigger()           # 第一项 = 「选中」
            assert got == [0]
        finally:
            strip.deleteLater()

    def test_overflow_menu_select_nonzero_index_not_overridden(self, qapp):
        """潜伏 bug 钉扎：triggered 信号带 checked=False 首参，lambda 必须
        显式吞掉——否则 ix 被 False 覆盖，非 0 步的「选中」实际选中第 0 行。"""
        strip = self._strip()
        try:
            strip.set_steps([{'label': f'm{i}', 'enabled': True}
                             for i in range(3)])
            got = []
            strip.sig_step_selected.connect(got.append)
            menu = strip._build_overflow_menu()
            sub = menu._step_submenus[2]
            sub.actions()[0].trigger()           # 第 3 步的「选中」
            assert strip._list.currentRow() == 2
            assert got == [2]
        finally:
            strip.deleteLater()

    def test_overflow_menu_edit_requests_step_editor(self, qapp):
        """An overflow step exposes the same commands as its visible chip."""
        strip = self._strip()
        try:
            strip.set_steps([{'label': f'm{i}', 'enabled': True}
                             for i in range(3)])
            got = []
            strip.sig_step_editor_requested.connect(got.append)
            menu = strip._build_overflow_menu()
            sub = menu._step_submenus[1]
            commands = {a.data(): a for a in sub.actions() if a.data()}
            assert set(commands) == {'edit-step', 'toggle-step', 'copy-step', 'move-up',
                                     'move-down', 'replace-step', 'remove-step'}
            commands['edit-step'].trigger()
            assert got == [1]
        finally:
            strip.deleteLater()

    # ---------------- P2：链空态引导 ----------------
    def test_empty_chain_shows_add_action(self, qapp):
        strip = self._strip()
        try:
            strip.show()
            strip.set_steps([])
            qapp.processEvents()
            assert strip._empty_add.isVisibleTo(strip)
            assert not strip._add_btn.isVisibleTo(strip)
            strip.set_steps([{'label': 'a', 'enabled': True}])
            qapp.processEvents()
            assert not strip._empty_add.isVisibleTo(strip)
            assert strip._add_btn.isVisibleTo(strip)
        finally:
            strip.deleteLater()

    # ---------------- P2：运行中清脏标记 / 取消恢复 ----------------
    def test_running_locks_submitted_draft_and_cancel_retains_it(self, qapp):
        page = self._page_with_run()
        try:
            page._pipeline_list.add_step('dewow', '去直流', {'win': 11})
            before = page.current_pipeline()
            page.set_running(True)
            assert not page._pipeline_list.isEnabled()
            assert not page._param_form.isEnabled()
            page.set_running(False)
            assert page._pipeline_list.isEnabled()
            assert page._param_form.isEnabled()
            assert page.current_pipeline() == before
        finally:
            page.close()

    def test_running_task_requires_explicit_cancel(self, qapp):
        page = self._page_with_run()
        try:
            cancelled, submitted = [], []
            page.cancel_requested.connect(lambda: cancelled.append(True))
            page.run_requested.connect(submitted.append)
            page.set_running(True)
            page._on_run_clicked()
            assert not submitted and not cancelled
            page._cancel_btn.click()
            assert cancelled == [True]
        finally:
            page.close()

    # ---------------- P2：chip / 卡标题中文显示名 ----------------
    def test_browsing_saved_result_does_not_replay_registered_method(self, qapp):
        page = self._page_with_run(display_name='去直流滤波')
        try:
            page.select_artifact('S1')
            assert page.current_pipeline()['steps'] == []
            assert page.current_pipeline()['input_artifact_id'] == 'S1'
            page._add_method_to_pipeline('dewow')
            assert page.current_pipeline()['steps'][0]['label'] == '去直流滤波'
        finally:
            page.close()

    def test_new_draft_method_falls_back_to_id_without_registry(self, qapp):
        page = self._page_with_run()
        try:
            page.select_artifact('S1')
            assert page.current_pipeline()['steps'] == []
            page._add_method_to_pipeline('dewow')
            assert page.current_pipeline()['steps'][0]['label'] == 'dewow'
            assert page.current_pipeline()['input_artifact_id'] == 'S1'
        finally:
            page.close()

    def test_long_method_name_elided_in_chip(self, qapp):
        strip = self._strip()
        try:
            long_id = 'zero_offset_correction_v2'
            strip.set_steps([{'label': long_id, 'enabled': True}])
            chip = strip._list.itemWidget(strip._list.item(0))
            assert '…' in chip.name.text()
            assert chip.name.text() != long_id   # 截断有省略号，非硬裁
        finally:
            strip.deleteLater()

    # ---------------- P3：flash 竞态 + a11y + 空态开关 ----------------
    def test_flash_reset_cancelled_by_rerun(self, qapp):
        strip = self._strip()
        try:
            strip.flash_success()
            assert strip.run_button().text() == '完成'
            assert strip._flash_timer.isActive() is True
            strip.set_running(True)      # 1.2s 内重跑 → 复位定时器被取消
            assert strip._flash_timer.isActive() is False
            assert strip.run_button().text() == '取消'
        finally:
            strip.deleteLater()

    def test_icon_only_buttons_have_accessible_names(self, qapp):
        strip = self._strip()
        try:
            strip.set_steps([{'label': 'a', 'enabled': True}])
            assert strip._add_btn.accessibleName()
            chip = strip._list.itemWidget(strip._list.item(0))
            assert chip.dot_btn.accessibleName()
            assert chip.del_btn.accessibleName()
        finally:
            strip.deleteLater()


    def test_line_combo_placeholder(self, qapp):
        page = self._page_with_run()
        try:
            assert page._line_combo._placeholderText == '选择测线'
        finally:
            page.close()

    # ---------------- 第三轮调研（✕ 对称 / 总览墙占位 / +N 渐隐） --------

    def test_gallery_placeholder_for_missing_bundle(self, qapp):
        """bundle 未回填的源显示 muted 占位（原直接跳过 → pyqtgraph 空画布）。"""
        from PyQt6.QtWidgets import QLabel

        from ui.widgets.bscan_gallery import BScanGallery
        page = self._page_with_run()
        try:
            gallery = BScanGallery(page)
            gallery.set_sources(
                [{'key': 'input', 'title': '输入', 'bundle': None}], 'input')
            try:
                hints = [w for w in gallery.findChildren(QLabel)
                         if w.text() == '预览尚未生成']
                assert hints, '空 bundle 源应显示占位而非空画布'
            finally:
                gallery.deleteLater()
        finally:
            page.close()

    def test_overflow_pill_gradient_fade(self, qapp):
        """+N 胶囊左缘渐隐（qlineargradient）——半遮 chip 文字不再硬切。"""
        strip = self._strip()
        try:
            assert 'qlineargradient' in strip._overflow_btn.styleSheet()
        finally:
            strip.deleteLater()
