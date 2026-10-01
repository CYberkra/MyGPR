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
    build_artifacts_model, build_files_model, build_tree_model, group_lines,
    group_stats, line_suffix,
)
from ui.widgets.file_tree_panel import (  # noqa: E402
    _SUFFIX_MAX_RATIO, _SUFFIX_PAD, suffix_column_width,
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


def _artifact(artifact_id='A1', line_id='L01', name='去直流',
              method_name='dewow', created_at='2026-09-16T10:00:00',
              shape=(100, 500), dtype='float32'):
    return types.SimpleNamespace(
        artifact_id=artifact_id, line_id=line_id, name=name,
        method_name=method_name, method_id=method_name,
        created_at=created_at, shape=shape, dtype=dtype)


def test_build_artifacts_model_groups_by_line_desc():
    nodes = build_artifacts_model([
        _artifact('A1', 'L02', created_at='2026-09-15T10:00:00'),
        _artifact('A2', 'L01'),
        _artifact('A3', 'L02', created_at='2026-09-16T11:00:00'),
    ])
    assert [n.text for n in nodes] == ['L01', 'L02']  # 测线组升序
    assert nodes[0].suffix == '1 项'
    assert nodes[1].suffix == '2 项'
    # 组内按创建时间倒序
    assert [c.payload for c in nodes[1].children] == ['A3', 'A1']
    leaf = nodes[1].children[0]
    assert leaf.kind == 'artifact' and leaf.aux == 'L02'
    assert leaf.suffix == 'dewow'  # 行尾角标=方法名


def test_build_artifacts_model_includes_spatial_and_reports():
    nodes = build_artifacts_model(
        [_artifact()],
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


def test_build_artifacts_model_omits_empty_sections():
    assert build_artifacts_model([]) == []
    nodes = build_artifacts_model([], spatial_results=[_spatial()])
    assert [n.text for n in nodes] == ['空间成果']
    assert nodes[0].children[0].kind == 'spatial'
    assert nodes[0].children[0].payload == 'SR1'
    assert nodes[0].children[0].text == '剖面图 v2'


# ------------------------------------------------ 文件视图（纯函数单层扫描）
def test_build_files_model_filters_internal_and_sorts(tmp_path):
    for name in ('raw', 'cache', 'metadata', '.trash', '.transactions'):
        (tmp_path / name).mkdir()
    (tmp_path / 'catalog.sqlite').write_text('x', encoding='utf-8')
    (tmp_path / 'catalog.sqlite-wal').write_text('x', encoding='utf-8')
    (tmp_path / 'project.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'a.dat').write_bytes(b'x' * 2048)

    nodes = build_files_model(tmp_path)
    names = [n.text for n in nodes]
    assert names == ['raw', 'a.dat', 'project.json']  # 目录在前 + 名称排序
    assert nodes[0].kind == 'dir' and nodes[0].children == ()
    assert nodes[1].kind == 'file' and nodes[1].suffix == '2.0 KB'
    assert nodes[1].payload == str(tmp_path / 'a.dat')


def test_build_files_model_bad_path_returns_empty(tmp_path):
    assert build_files_model(tmp_path / 'nonexistent') == []
    assert build_files_model('') == []


# ---------------------------------------------------------------- 面板行为
_PANEL_HOLDER: dict = {}


@pytest.fixture
def panel(qapp):
    from PyQt6.QtWidgets import QWidget
    from ui.widgets.file_tree_panel import FileTreePanel

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
    p._page_states = {}
    p._current_page = ''
    # 视图分段复位到「测线」（阻断信号避免触发持久化路径）
    p._view_segment.blockSignals(True)
    p._view_segment.setCurrentItem('lines')
    p._view_segment.blockSignals(False)
    p._current_view = 'lines'
    p._empty_label.setText('尚未导入测线')
    p.set_project_info(None)   # 同时清空测线/成果/空间/报告与签名
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
    panel._set_view('artifacts', remember=False)  # 空间/报告组在「成果」视图
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


def test_artifact_leaf_click_emits_focus_with_line(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel._set_view('artifacts', remember=False)
    panel.set_artifacts([_artifact('A9', 'L02')])
    leaf = _find_by_kind(panel._tree, 'artifact')
    assert leaf is not None
    assert leaf.text(0) == '去直流' and leaf.text(1) == 'dewow'
    got, lines_got = [], []
    panel.artifact_focus_requested.connect(lambda *a: got.append(a))
    panel.line_selected.connect(lines_got.append)
    panel._on_item_clicked(leaf, 0)
    assert got == [('L02', 'A9')]
    assert lines_got == []  # 换线由链路决定，面板不直接发


def test_set_artifacts_signature_dedup_skips_rebuild(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel._set_view('artifacts', remember=False)
    panel.set_artifacts([_artifact('A1')])
    leaf = _find_by_kind(panel._tree, 'artifact')
    panel.set_artifacts([_artifact('A1')])  # 同签名：不重建
    assert _find_by_kind(panel._tree, 'artifact') is leaf
    panel.set_artifacts([_artifact('A1'), _artifact('A2')])  # 变了才重建
    assert _find_by_kind(panel._tree, 'artifact') is not leaf


def test_files_view_shows_project_root(qapp, panel, tmp_path):
    """文件视图：有项目根则显示文件浏览树，无项目显示空态文案。"""
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'note.txt').write_text('x', encoding='utf-8')
    panel.set_project_info(
        types.SimpleNamespace(name='测试1', root_path=str(tmp_path)))
    panel._set_view('files', remember=False)
    assert not panel._tree.isHidden()
    names = [panel._tree.topLevelItem(i).text(0)
             for i in range(panel._tree.topLevelItemCount())]
    assert names == ['raw', 'note.txt']  # 目录在前
    # 无项目：空态文案
    panel.set_project_info(None)
    assert panel._tree.isHidden()
    assert panel._empty_label.text() == '打开项目后在此浏览项目文件'


def test_files_view_lazy_expands_directory(qapp, panel, tmp_path):
    """目录带占位行，首次展开时替换为真实子层（懒加载）。"""
    sub = tmp_path / 'data'
    sub.mkdir()
    (sub / 'a.dat').write_text('x', encoding='utf-8')
    (sub / 'cache').mkdir()  # 内部目录即使嵌套也过滤
    panel.set_project_info(
        types.SimpleNamespace(name='测试1', root_path=str(tmp_path)))
    panel._set_view('files', remember=False)
    dir_item = panel._tree.topLevelItem(0)
    assert dir_item.text(0) == 'data'
    assert dir_item.childCount() == 1  # 占位行
    assert dir_item.child(0).data(0, _KIND_ROLE) == 'placeholder'
    panel._on_item_expanded(dir_item)
    assert dir_item.childCount() == 1
    assert dir_item.child(0).text(0) == 'a.dat'
    assert dir_item.child(0).data(0, _KIND_ROLE) == 'file'


def test_files_view_double_click_file_opens(qapp, panel, tmp_path,
                                            monkeypatch):
    f = tmp_path / 'a.dat'
    f.write_text('x', encoding='utf-8')
    panel.set_project_info(
        types.SimpleNamespace(name='测试1', root_path=str(tmp_path)))
    panel._set_view('files', remember=False)
    opened = []
    monkeypatch.setattr(
        'ui.widgets.file_tree_panel.QDesktopServices.openUrl',
        lambda url: opened.append(url))
    panel._on_item_double_clicked(panel._tree.topLevelItem(0), 0)
    assert len(opened) == 1


def test_files_view_context_menu_on_file(qapp, panel, tmp_path, monkeypatch):
    from PyQt6.QtCore import QPoint
    from qfluentwidgets import RoundMenu
    f = tmp_path / 'a.dat'
    f.write_text('x', encoding='utf-8')
    panel.set_project_info(
        types.SimpleNamespace(name='测试1', root_path=str(tmp_path)))
    panel._set_view('files', remember=False)
    shown = []
    monkeypatch.setattr(RoundMenu, 'exec',
                        lambda self, *a, **k: shown.append(1))
    monkeypatch.setattr(panel._tree, 'itemAt',
                        lambda p: panel._tree.topLevelItem(0))
    panel._on_context_menu(QPoint(5, 5))
    assert len(shown) == 1


def test_tree_shown_with_only_spatial_results(qapp, panel):
    # 成果视图：无处理成果但有空间成果时树可见
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel._set_view('artifacts', remember=False)
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
    panel._set_view('artifacts', remember=False)
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
    saved = settings.get('file_tree_page_states')
    assert saved['projectInterface']['lines'] is True  # 二维格式


def test_collapse_memory_is_per_page_and_view(qapp, panel):
    """同一页面不同视图各自记忆收起态（按页×按视图二维）。"""
    settings = _FakeSettings()
    panel.set_settings_manager(settings)
    panel.apply_page('projectInterface')
    panel._set_view('artifacts', remember=False)
    panel._remember_current(True)   # 项目页×成果视图：收起
    panel._set_view('lines', remember=False)
    assert not panel._collapsed     # 项目页×测线视图：默认展开，不受影响
    panel._set_view('artifacts', remember=False)
    assert panel._collapsed         # 切回成果视图恢复收起
    saved = settings.get('file_tree_page_states')
    assert saved['projectInterface']['artifacts'] is True


def test_settings_legacy_flat_format_broadcasts_to_all_views(qapp, panel):
    """旧版一维设置 {page: bool} 读取时广播到全部视图。"""
    settings = _FakeSettings()
    settings.set('file_tree_page_states', {'projectInterface': True})
    panel.set_settings_manager(settings)
    assert panel._collapsed_for('projectInterface', 'lines') is True
    assert panel._collapsed_for('projectInterface', 'artifacts') is True


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


def test_artifact_context_menu_shows_and_selects_without_emitting(
        qapp, panel, monkeypatch):
    """成果叶子右键：弹菜单 + 右键即选中；菜单项点击前不发任何信号。"""
    from PyQt6.QtCore import QPoint
    from qfluentwidgets import RoundMenu
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel._set_view('artifacts', remember=False)
    panel.set_artifacts([_artifact('A9', 'L02')])
    leaf = _find_by_kind(panel._tree, 'artifact')
    assert leaf is not None
    shown = []
    monkeypatch.setattr(RoundMenu, 'exec',
                        lambda self, *a, **k: shown.append(1))
    monkeypatch.setattr(panel._tree, 'itemAt', lambda p: leaf)
    focus_got, delete_got = [], []
    panel.artifact_focus_requested.connect(lambda *a: focus_got.append(a))
    panel.artifact_delete_requested.connect(lambda *a: delete_got.append(a))
    panel._on_context_menu(QPoint(5, 5))
    assert len(shown) == 1
    assert panel._tree.currentItem() is leaf  # 右键即选中（与测线同语义）
    assert focus_got == [] and delete_got == []  # 信号只在点菜单项时发


def test_artifact_context_menu_actions_emit_focus_delete_copy(
        qapp, panel, monkeypatch):
    """菜单项：添加到显示 / 删除成果… / 复制成果号 各自发信号或写剪贴板。"""
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QApplication
    from qfluentwidgets import RoundMenu
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel._set_view('artifacts', remember=False)
    panel.set_artifacts([_artifact('A9', 'L02')])
    leaf = _find_by_kind(panel._tree, 'artifact')
    monkeypatch.setattr(RoundMenu, 'exec', lambda self, *a, **k: None)
    monkeypatch.setattr(panel._tree, 'itemAt', lambda p: leaf)
    # 捕获 add_action 挂进菜单的 slot，逐个触发模拟点击菜单项
    actions: list = []
    monkeypatch.setattr(
        'ui.widgets.file_tree_panel.add_action',
        lambda menu, icon, text, slot, **k: actions.append((str(text), slot)))
    panel._on_context_menu(QPoint(5, 5))
    assert [t for t, _ in actions] == ['添加到显示', '删除成果…', '复制成果号']
    focus_got, delete_got = [], []
    panel.artifact_focus_requested.connect(lambda *a: focus_got.append(a))
    panel.artifact_delete_requested.connect(lambda *a: delete_got.append(a))
    for _text, slot in actions:
        slot()
    assert focus_got == [('L02', 'A9')]   # 与单击同语义（换线如需+跳预览）
    assert delete_got == [('L02', 'A9')]  # 面板不弹确认框，直接交删除链路
    assert QApplication.clipboard().text() == 'A9'


def test_confirm_delete_emits_line_ids_on_accept(qapp, panel, monkeypatch):
    from PyQt6.QtWidgets import QDialog
    from qfluentwidgets import MessageBox
    monkeypatch.setattr(MessageBox, 'exec',
                        lambda self: QDialog.DialogCode.Accepted)
    got = []
    panel.line_delete_requested.connect(got.append)
    panel._confirm_delete('L01')
    assert got == [['L01']]


# ---------------------------------------------------------------- 名称列宽度
class TestNameColumnWidth:
    """文件树「只能显示两个字」回归（名称列被角标列挤没）。

    根因（offscreen 实测，面板 232px / 视口 215px）：Qt6 的 QHeaderView
    默认 ``stretchLastSection=True``，即便第二列（角标）设为
    ResizeToContents 也会被拉伸到与名称列平分（108 / 107）；再叠加每级
    缩进 + 状态圆点，深层节点实际只剩约两个汉字的绘制宽度。
    修复：关末列拉伸 + 角标列按内容定宽并封顶（名称列 Stretch 吃余量）。

    注：这里刻意不把面板 show 出来取真实布局——实测那样做会让进程退出时
    触发本模块 fixture 注释记载的 offscreen access violation（6 次约 1 次），
    改为「纯函数定策略 + setColumnWidth 打桩验证调用值」。
    """

    # ---------------------------------------------------------- 结构不变量
    def test_last_section_stretch_disabled(self, panel):
        """末列拉伸必须关闭——开启时角标列会被拉到与名称列平分。"""
        assert panel._tree.header().stretchLastSection() is False

    def test_suffix_column_is_fixed_mode(self, panel):
        """角标列必须是定宽：ResizeToContents 会把名称列反向压没。"""
        from PyQt6.QtWidgets import QHeaderView
        assert panel._tree.header().sectionResizeMode(1) ==             QHeaderView.ResizeMode.Fixed

    # ---------------------------------------------------------- 定宽策略（纯函数）
    def test_width_is_zero_without_viewport_or_suffix(self):
        assert suffix_column_width(0, 100) == 0
        assert suffix_column_width(215, 0) == 0

    def test_short_suffix_uses_content_width(self):
        assert suffix_column_width(215, 72) == 72 + _SUFFIX_PAD

    def test_long_suffix_is_capped(self):
        """长角标（如 16 字符时间戳 192px）封顶，不吞掉名称列。"""
        assert suffix_column_width(215, 192) == int(215 * _SUFFIX_MAX_RATIO)
        assert suffix_column_width(215, 192) < 192, '封顶未生效'

    def test_cap_scales_with_viewport(self):
        assert suffix_column_width(160, 999) < suffix_column_width(215, 999)

    def test_name_column_keeps_majority(self):
        """名称列 = 视口余量，任何角标下都拿大头（旧实现仅约 50%）。"""
        for widest in (0, 40, 72, 132, 192, 999):
            col1 = suffix_column_width(215, widest)
            assert 215 - col1 >= int(215 * 0.6), (
                f'角标 {widest}px 时名称列只剩 {215 - col1}/215')

    # ---------------------------------------------------------- 面板接线
    def test_apply_suffix_width_sets_capped_width(self, panel, monkeypatch):
        """_apply_suffix_width 必须按视口宽封顶后写给第二列。"""
        panel.set_lines([_line('L01', '2026-09-16T01:00:00', 98.0, '已处理'),
                         _line('L02', '2026-09-16T02:00:00', 98.0, '已处理')])
        calls = []
        monkeypatch.setattr(panel._tree, 'setColumnWidth',
                            lambda col, w: calls.append((col, w)))
        # 隐藏态视口宽恒为默认的 95px，打桩成真实布局下的值
        monkeypatch.setattr(panel._tree.viewport(), 'width', lambda: 215)
        panel._apply_suffix_width()
        assert calls, '未设置角标列宽'
        col, width = calls[-1]
        assert col == 1
        assert width == suffix_column_width(215, max(
            panel._tree.fontMetrics().horizontalAdvance(s)
            for s in panel._suffixes))

    def test_long_suffix_does_not_squeeze_name_column(self, panel,
                                                      monkeypatch):
        """16 字符时间戳角标下仍封顶（ResizeToContents 会把名称列压到 20px）。"""
        panel.set_artifacts([
            types.SimpleNamespace(
                artifact_id=f'A{i:03d}', line_id=f'L{i:02d}', name=f'A{i:03d}',
                method_name='', created_at='2026-09-16T12:00:00', shape=(1, 2))
            for i in range(1, 4)])
        calls = []
        monkeypatch.setattr(panel._tree, 'setColumnWidth',
                            lambda col, w: calls.append((col, w)))
        monkeypatch.setattr(panel._tree.viewport(), 'width', lambda: 215)
        panel._apply_suffix_width()
        assert calls[-1][1] <= int(215 * _SUFFIX_MAX_RATIO), (
            f'长角标未封顶：{calls[-1][1]}')
