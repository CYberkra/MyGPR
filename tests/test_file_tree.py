# -*- coding: utf-8 -*-
"""文件树：分组纯函数 + 面板行为（叶子可选、分组行不可选、同步防回环）。"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402

from ui.file_tree import (  # noqa: E402
    build_tree_model, group_lines, group_stats, line_suffix,
)


def _line(line_id, updated_at='', length_m=0.0, status='', processed_result='',
          target_count=0, interface_keypoint_count=0):
    return types.SimpleNamespace(
        line_id=line_id, name=line_id, updated_at=updated_at,
        length_m=length_m, processing_status=status,
        processed_result=processed_result, target_count=target_count,
        interface_keypoint_count=interface_keypoint_count)


def _spatial(result_id='SR1', name='剖面图', revision=2, status='ok',
             line_ids=('L01',), created_at='2026-09-16T10:00:00', stale=False):
    return types.SimpleNamespace(
        result_id=result_id, name=name, revision=revision, status=status,
        line_ids=list(line_ids), created_at=created_at, stale=stale)


def _report(package_dir='proj/reports/r20260916',
            generated_at='2026-09-16T12:00:00', file_count=3):
    return types.SimpleNamespace(
        package_dir=package_dir, generated_at=generated_at,
        file_count=file_count)


# ---------------------------------------------------------------- 纯函数
def test_group_by_date_desc_and_lines_asc():
    lines = [
        _line('L09', '2026-09-16T01:20:58'),
        _line('L01', '2026-09-16T01:00:00'),
        _line('L05', '2026-09-15T23:10:00'),
    ]
    groups = dict(group_lines(lines))
    assert list(groups) == ['2026-09-16', '2026-09-15']  # 日期倒序
    assert [ln.line_id for ln in groups['2026-09-16']] == ['L01', 'L09']
    assert [ln.line_id for ln in groups['2026-09-15']] == ['L05']


def test_all_undated_falls_back_to_flat():
    lines = [_line('L01'), _line('L02')]
    groups = group_lines(lines)
    assert groups == [('', lines)]  # 平铺：不产生分组节点


def test_partial_undated_joins_ungrouped():
    lines = [_line('L01', '2026-09-16T01:00:00'), _line('L02')]
    groups = dict(group_lines(lines))
    assert set(groups) == {'2026-09-16', '未分组'}
    assert [ln.line_id for ln in groups['未分组']] == ['L02']


def test_group_stats_shows_count_and_length():
    lines = [_line('L01', length_m=98.4), _line('L02', length_m=50.0)]
    assert group_stats(lines) == '2 条 · 148 m'
    assert group_stats([]) == '0 条'


# ------------------------------------------------ 节点模型（Provider 纯函数）
def test_line_suffix_derives_from_line_fields():
    assert line_suffix(_line('L01')) == ''
    assert line_suffix(_line('L01', processed_result='res.dat')) == '成果✓'
    assert line_suffix(_line('L01', target_count=3)) == '标3'
    assert line_suffix(_line('L01', processed_result='r', target_count=2,
                             interface_keypoint_count=5)) == '成果✓ 标2 界面✓'


def test_build_tree_model_groups_lines_and_suffixes():
    nodes = build_tree_model([
        _line('L01', '2026-09-16T01:00:00', processed_result='r'),
        _line('L02', '2026-09-16T02:00:00'),
    ])
    assert len(nodes) == 1
    group = nodes[0]
    assert group.kind == 'group' and group.text == '2026-09-16'
    assert [c.payload for c in group.children] == ['L01', 'L02']
    assert group.children[0].suffix == '成果✓'
    assert group.children[1].suffix == ''


def test_build_tree_model_omits_empty_sections():
    # 无空间成果/报告时不产生对应分组节点
    nodes = build_tree_model([_line('L01', '2026-09-16T01:00:00')])
    assert all(n.text not in ('空间成果', '项目报告') for n in nodes)
    # 只有成果无测线：只有成果组
    nodes = build_tree_model([], spatial_results=[_spatial()])
    assert [n.text for n in nodes] == ['空间成果']
    assert nodes[0].children[0].kind == 'spatial'
    assert nodes[0].children[0].payload == 'SR1'
    assert nodes[0].children[0].text == '剖面图 v2'


def test_build_tree_model_spatial_sorted_desc_and_reports():
    nodes = build_tree_model(
        [],
        spatial_results=[
            _spatial('SR1', created_at='2026-09-15T10:00:00'),
            _spatial('SR2', created_at='2026-09-16T10:00:00'),
        ],
        reports=[_report()],
    )
    spatial_group = next(n for n in nodes if n.text == '空间成果')
    assert [c.payload for c in spatial_group.children] == ['SR2', 'SR1']
    report_group = next(n for n in nodes if n.text == '项目报告')
    assert report_group.children[0].kind == 'report'
    assert report_group.children[0].payload == 'proj/reports/r20260916'
    assert report_group.children[0].text == 'r20260916'
    assert report_group.children[0].suffix == '2026-09-16'


# ---------------------------------------------------------------- 面板行为
_PANEL_HOLDER: dict = {}


@pytest.fixture
def panel(qapp):
    from PyQt6.QtWidgets import QWidget
    from ui.widgets.file_tree_panel import _DEFAULT_PAGE_COLLAPSED, FileTreePanel

    # 全模块共享一个面板实例、逐用例复位，而不是每用例新建/销毁：
    # offscreen 平台下 qfluentwidgets TreeWidget 反复走原生窗口生命周期
    # 会触发库级 access violation（该缺陷同样被 CI 以"retry once on
    # Qt-offscreen exit crash (139)"容忍）。挂隐藏宿主避免 setVisible(True)
    # 创建原生窗口；唯一实例在进程退出时随 QApplication 一起销毁。
    if 'panel' not in _PANEL_HOLDER:
        host = QWidget()
        _PANEL_HOLDER['host'] = host
        _PANEL_HOLDER['panel'] = FileTreePanel(host)
    p = _PANEL_HOLDER['panel']
    # 复位到初始态（等价于新建面板）
    p.set_busy(False)
    p.set_settings_manager(None)
    p._page_states = dict(_DEFAULT_PAGE_COLLAPSED)
    p._current_page = ''
    p.set_project_info(None)
    p.set_lines([])
    p._current_line_id = ''
    p._update_strip_text()
    p.set_collapsed(False, animate=False)
    yield p
    qapp.sendPostedEvents()
    qapp.processEvents()


def test_panel_visible_empty_state_without_project(panel):
    """面板常驻（入口通用性）：无项目时显示空态而非整树隐藏。"""
    panel.set_project_info(None)
    assert not panel.isHidden()
    assert panel._project_label.text() == '未打开项目'
    assert panel._tree.isHidden()


def test_panel_builds_leaves_and_selects(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00', 98.0, '已处理'),
                     _line('L09', '2026-09-16T02:00:00')])
    tree = panel._tree
    assert tree.topLevelItemCount() == 1  # 单一分组
    group = tree.topLevelItem(0)
    assert group.childCount() == 2
    # 分组行不可选（防预览代数被无关点击推进）
    assert not (group.flags() & Qt.ItemFlag.ItemIsSelectable)
    # 当前测线高亮
    panel.set_current_line('L09')
    current = tree.currentItem()
    assert current is not None
    assert current.data(0, Qt.ItemDataRole.UserRole) == 'L09'


def test_leaf_click_emits_line_selected(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00')])
    got = []
    panel.line_selected.connect(got.append)
    leaf = panel._line_id_by_item['L01']
    panel._on_item_clicked(leaf, 0)
    assert got == ['L01']


def test_set_current_line_does_not_reemit(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00')])
    got = []
    panel.line_selected.connect(got.append)
    panel.set_current_line('L01')  # 外部换线同步，不得回发
    assert got == []


def test_busy_blocks_click_emission(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00')])
    got = []
    panel.line_selected.connect(got.append)
    panel.set_busy(True)
    assert not panel._tree.isEnabled()
    panel._on_item_clicked(panel._line_id_by_item['L01'], 0)
    assert got == []


# ------------------------------------------------ 空间成果 / 项目报告节点
_KIND_ROLE = Qt.ItemDataRole.UserRole + 1


def _find_by_kind(tree, kind):
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        if top.data(0, _KIND_ROLE) == kind:
            return top
        for j in range(top.childCount()):
            child = top.child(j)
            if child.data(0, _KIND_ROLE) == kind:
                return child
    return None


def test_spatial_and_report_leaves_render_and_click_goto_delivery(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00')])
    panel.set_spatial_results([_spatial()])
    panel.set_reports([_report()])
    tree = panel._tree
    spatial = _find_by_kind(tree, 'spatial')
    report = _find_by_kind(tree, 'report')
    assert spatial is not None and report is not None
    assert spatial.text(0) == '剖面图 v2'
    assert spatial.text(1) == '2026-09-16'  # 行尾角标列
    lines_got, delivery_got = [], []
    panel.line_selected.connect(lines_got.append)
    panel.delivery_focus_requested.connect(delivery_got.append)
    panel._on_item_clicked(spatial, 0)
    panel._on_item_clicked(report, 0)
    assert delivery_got == ['spatial', 'report']
    assert lines_got == []  # 成果/报告点击不得触发换线


def test_tree_shown_with_only_spatial_results(qapp, panel):
    # 无测线但有成果：树可见（空态只看三类数据全空）
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([])
    panel.set_spatial_results([_spatial()])
    assert not panel._tree.isHidden()


def test_line_leaf_shows_suffix_in_second_column(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00',
                           processed_result='r', target_count=2)])
    leaf = panel._line_id_by_item['L01']
    assert leaf.text(0) == 'L01'
    assert leaf.text(1) == '成果✓ 标2'


def test_context_menu_suppressed_for_spatial_leaf(qapp, panel, monkeypatch):
    from PyQt6.QtCore import QPoint
    from qfluentwidgets import RoundMenu
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_spatial_results([_spatial()])
    shown = []
    monkeypatch.setattr(RoundMenu, 'exec',
                        lambda self, *a, **k: shown.append(1))
    spatial = _find_by_kind(panel._tree, 'spatial')
    monkeypatch.setattr(panel._tree, 'itemAt', lambda p: spatial)
    panel._on_context_menu(QPoint(5, 5))
    assert shown == []


# ------------------------------------------------ 细条态与页面记忆
class _FakeSettings:
    def __init__(self):
        self._d = {}
    def get(self, key, default=None):
        return self._d.get(key, default)
    def set(self, key, value):
        self._d[key] = value
    def save(self):
        return True


def test_collapse_shows_strip_with_current_line(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00')])
    panel.set_current_line('L01')
    panel.set_collapsed(True, animate=False)
    assert panel.width() <= 20  # 细条
    assert not panel._expanded_view.isVisibleTo(panel)
    assert panel._strip_view.isVisibleTo(panel)
    assert 'L\n0\n1' in panel._strip_line_label.text()
    panel.set_collapsed(False, animate=False)
    assert panel._expanded_view.isVisibleTo(panel)
    assert panel.width() >= 200


def test_apply_page_defaults_project_open_others_collapsed(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.apply_page('projectInterface')
    assert not panel._collapsed
    panel.apply_page('processingInterface')
    assert panel._collapsed


def test_manual_toggle_is_remembered_per_page_and_persisted(qapp, panel):
    settings = _FakeSettings()
    panel.set_settings_manager(settings)
    panel.apply_page('projectInterface')
    panel._remember_current(True)  # 模拟用户在项目页手动收起
    panel.apply_page('processingInterface')
    assert panel._collapsed
    panel.apply_page('projectInterface')
    assert panel._collapsed  # 项目页记忆被手动覆盖
    assert settings.get('file_tree_page_states')['projectInterface'] is True


def test_strip_text_updates_on_line_switch_while_collapsed(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L09')])
    panel.set_collapsed(True, animate=False)
    panel.set_current_line('L09')
    assert 'L\n0\n9' in panel._strip_line_label.text()


# ------------------------------------------------ 状态圆点与右键菜单
def test_status_key_mapping(qapp):
    from ui.widgets.file_tree_panel import _status_key
    assert _status_key('已完成') == 'success'
    assert _status_key('处理完成') == 'success'
    assert _status_key('已导入') == 'info'
    assert _status_key('未处理') == 'disabled'
    assert _status_key('') == 'disabled'


def test_leaf_has_status_icon_tooltip_and_plain_text(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00', 98.4, '已完成')])
    leaf = panel._line_id_by_item['L01']
    assert leaf.text(0) == 'L01'  # 不再拼 [状态] 尾巴
    assert not leaf.icon(0).isNull()  # 状态色圆点
    tip = leaf.toolTip(0)
    assert '已完成' in tip and '98.4' in tip and '2026-09-16' in tip


def test_apply_theme_rebuilds_keeps_selection(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00', status='已导入')])
    panel.set_current_line('L01')
    panel.apply_theme(True)
    leaf = panel._line_id_by_item['L01']
    assert panel._tree.currentItem() is leaf
    assert not leaf.icon(0).isNull()


def test_context_menu_suppressed_for_blank_group_and_busy(qapp, panel,
                                                          monkeypatch):
    from PyQt6.QtCore import QPoint
    from qfluentwidgets import RoundMenu
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00')])
    shown = []
    monkeypatch.setattr(RoundMenu, 'exec',
                        lambda self, *a, **k: shown.append(1))
    pos = QPoint(5, 5)
    monkeypatch.setattr(panel._tree, 'itemAt', lambda p: None)  # 空白
    panel._on_context_menu(pos)
    monkeypatch.setattr(panel._tree, 'itemAt',
                        lambda p: panel._tree.topLevelItem(0))  # 分组行
    panel._on_context_menu(pos)
    monkeypatch.setattr(panel._tree, 'itemAt',
                        lambda p: panel._line_id_by_item['L01'])
    panel.set_busy(True)  # busy
    panel._on_context_menu(pos)
    assert shown == []


def test_context_menu_on_leaf_selects_shows_and_emits(qapp, panel,
                                                      monkeypatch):
    from PyQt6.QtCore import QPoint
    from qfluentwidgets import RoundMenu
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00'),
                     _line('L02', '2026-09-16T02:00:00')])
    panel.set_current_line('L01')
    got = []
    panel.line_selected.connect(got.append)
    shown = []
    monkeypatch.setattr(RoundMenu, 'exec',
                        lambda self, *a, **k: shown.append(1))
    leaf = panel._line_id_by_item['L02']
    monkeypatch.setattr(panel._tree, 'itemAt', lambda p: leaf)
    panel._on_context_menu(QPoint(5, 5))
    assert len(shown) == 1
    assert got == ['L02']  # 右键即选中并换线
    assert panel._tree.currentItem() is leaf


def test_confirm_delete_emits_line_ids_on_accept(qapp, panel, monkeypatch):
    from PyQt6.QtWidgets import QDialog
    from qfluentwidgets import MessageBox
    monkeypatch.setattr(MessageBox, 'exec',
                        lambda self: QDialog.DialogCode.Accepted)
    got = []
    panel.line_delete_requested.connect(got.append)
    panel._confirm_delete('L01')
    assert got == [['L01']]
