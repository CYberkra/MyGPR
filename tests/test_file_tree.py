# -*- coding: utf-8 -*-
"""文件树：装配纯函数 + 面板行为（叶子可选、分类行不可选、同步防回环）。"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402

from ui.file_tree import (  # noqa: E402
    build_artifacts_model, build_files_model, build_project_model,
    build_tree_model, line_suffix,
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


# ------------------------------------------------ 节点模型（Provider 纯函数）
def test_line_suffix_derives_from_line_fields():
    assert line_suffix(_line('L01')) == ''
    assert line_suffix(_line('L01', processed_result='res.dat')) == '成果✓'
    assert line_suffix(_line('L01', target_count=3)) == '标3'
    assert line_suffix(_line('L01', processed_result='r', target_count=2,
                             interface_keypoint_count=5)) == '成果✓ 标2 界面✓'


def test_build_tree_model_flat_lines_sorted_by_id():
    """测线平铺：无分组层，line_id 升序（排序与 updated_at 日期无关）。"""
    nodes = build_tree_model([
        _line('L09', '2026-09-16T01:20:58', processed_result='r'),
        _line('L01', '2026-09-15T23:10:00'),
    ])
    assert [n.kind for n in nodes] == ['line', 'line']
    assert [n.payload for n in nodes] == ['L01', 'L09']
    assert nodes[1].suffix == '成果✓'
    assert nodes[0].suffix == ''


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


# ------------------------------------------------ 项目总树（测线来源树与独立交付分类）
def test_build_project_model_four_categories_with_counts():
    lines = [_line('L01'), _line('L02')]
    nodes = build_project_model(lines, [_artifact('A1', 'L01')])
    assert [n.kind for n in nodes] == ['category'] * 4
    assert [n.text for n in nodes] == ['测线', '空间成果', '交付成果', '来源与附件']
    assert [n.suffix for n in nodes] == ['2', '0', '0', '0']
    assert nodes[0].children[0].kind == 'line'
    sources = nodes[0].children[0].children
    assert sources[0].kind == 'raw'
    assert sources[1].text == '历史成果 · 来源待核对'
    assert sources[1].children[0].kind == 'artifact'
    assert all(node.children == () for node in nodes[1:])


def test_build_project_model_counts_spatial_reports_and_files(tmp_path):
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'a.dat').write_text('x', encoding='utf-8')
    nodes = build_project_model(
        [], [], spatial_results=[_spatial()], reports=[_report()],
        files_root=tmp_path)
    assert nodes[1].suffix == nodes[2].suffix == '1'
    assert nodes[3].suffix == '2'   # 1 目录 + 1 文件
    assert nodes[3].children[0].kind == 'dir'


def test_build_project_model_without_root_omits_files(tmp_path):
    nodes = build_project_model([], [], files_root=None)
    assert nodes[3].suffix == '0'
    assert nodes[3].children == ()


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
    assert tree.topLevelItemCount() == 4
    lines_cat = tree.topLevelItem(0)
    assert lines_cat.text(0) == '测线'
    assert lines_cat.text(1) == '2'      # 计数角标
    leaf0 = lines_cat.child(0)           # 平铺：叶子直接挂分类下
    assert leaf0.text(0).startswith('L01')
    assert leaf0.childCount() == 1
    assert leaf0.child(0).data(0, Qt.ItemDataRole.UserRole + 1) == 'raw'
    assert lines_cat.child(1).text(0).startswith('L09')
    # 分类行不可选（防预览代数被无关点击推进）
    assert not (lines_cat.flags() & Qt.ItemFlag.ItemIsSelectable)
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
    """深度优先找第一个指定 kind 的节点（分类→分组→叶子 任意深度）。"""
    def _walk(item):
        if item.data(0, _KIND_ROLE) == kind:
            return item
        for j in range(item.childCount()):
            found = _walk(item.child(j))
            if found is not None:
                return found
        return None

    for i in range(tree.topLevelItemCount()):
        found = _walk(tree.topLevelItem(i))
        if found is not None:
            return found
    return None


def _category(tree, name):
    """按文字取顶层分类行（测线/成果/文件）。"""
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        if top.text(0) == name:
            return top
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


def test_artifact_leaf_click_emits_focus_with_line(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L02')])
    panel.set_artifacts([_artifact('A9', 'L02')])
    leaf = _find_by_kind(panel._tree, 'artifact')
    assert leaf is not None
    assert leaf.text(0).startswith('去直流') and leaf.text(1) == 'dewow'
    assert '记录不完整' in leaf.text(0)
    got, lines_got = [], []
    panel.artifact_focus_requested.connect(lambda *a: got.append(a))
    panel.line_selected.connect(lines_got.append)
    panel._on_item_clicked(leaf, 0)
    assert got == [('L02', 'A9')]
    assert lines_got == []  # 换线由链路决定，面板不直接发


def test_set_artifacts_signature_dedup_skips_rebuild(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01')])
    panel.set_artifacts([_artifact('A1')])
    leaf = _find_by_kind(panel._tree, 'artifact')
    panel.set_artifacts([_artifact('A1')])  # 同签名：不重建
    assert _find_by_kind(panel._tree, 'artifact') is leaf
    panel.set_artifacts([_artifact('A1'), _artifact('A2')])  # 变了才重建
    assert _find_by_kind(panel._tree, 'artifact') is not leaf


def test_files_view_shows_project_root(qapp, panel, tmp_path):
    """文件分类：有项目根则挂目录树；无项目时整树隐藏（分类计数自明）。"""
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'note.txt').write_text('x', encoding='utf-8')
    panel.set_project_info(
        types.SimpleNamespace(name='测试1', root_path=str(tmp_path)))
    files_cat = _category(panel._tree, '来源与附件')
    assert files_cat is not None
    assert files_cat.text(1) == '2'   # 1 目录 + 1 文件
    names = [files_cat.child(i).text(0) for i in range(files_cat.childCount())]
    assert names == ['raw', 'note.txt']  # 目录在前
    # 无项目：整树隐藏
    panel.set_project_info(None)
    assert panel._tree.isHidden()


def test_files_view_lazy_expands_directory(qapp, panel, tmp_path):
    """目录带占位行，首次展开时替换为真实子层（懒加载）。"""
    sub = tmp_path / 'data'
    sub.mkdir()
    (sub / 'a.dat').write_text('x', encoding='utf-8')
    (sub / 'cache').mkdir()  # 内部目录即使嵌套也过滤
    panel.set_project_info(
        types.SimpleNamespace(name='测试1', root_path=str(tmp_path)))
    files_cat = _category(panel._tree, '来源与附件')
    dir_item = files_cat.child(0)
    assert dir_item.text(0) == 'data'
    assert dir_item.childCount() == 1  # 占位行
    assert dir_item.child(0).data(0, _KIND_ROLE) == 'placeholder'
    panel._on_item_expanded(dir_item)
    assert dir_item.childCount() == 1
    assert dir_item.child(0).text(0) == 'a.dat'
    assert dir_item.child(0).data(0, _KIND_ROLE) == 'file'


def test_attachments_start_collapsed_and_preserve_explicit_expansion(panel, tmp_path):
    """Physical files stay out of the source tree until explicitly opened."""
    sub = tmp_path / 'data'
    sub.mkdir()
    (sub / 'a.dat').write_text('x', encoding='utf-8')
    panel.set_project_info(types.SimpleNamespace(name='测试', root_path=str(tmp_path)))
    files = _category(panel._tree, '来源与附件')
    assert not files.isExpanded()
    assert not files.child(0).isExpanded()
    files.setExpanded(True)
    files.child(0).setExpanded(True)
    panel.apply_theme(True)
    files = _category(panel._tree, '来源与附件')
    assert files.isExpanded()
    assert files.child(0).isExpanded()
    assert files.child(0).child(0).text(0) == 'a.dat'
    files.setExpanded(False)
    panel.set_lines([_line('L01')])
    assert not _category(panel._tree, '来源与附件').isExpanded()


def test_first_attachment_does_not_expand_physical_files(panel, tmp_path):
    panel.set_project_info(types.SimpleNamespace(name='测试', root_path=str(tmp_path)))
    (tmp_path / 'notes.txt').write_text('notes', encoding='utf-8')
    panel.set_lines([_line('L01')])
    files = _category(panel._tree, '来源与附件')
    assert files.childCount() == 1
    assert not files.isExpanded()
    assert _category(panel._tree, '测线').isExpanded()


def test_expanded_empty_directory_survives_theme_rebuild(panel, tmp_path):
    (tmp_path / 'empty').mkdir()
    panel.set_project_info(types.SimpleNamespace(name='测试', root_path=str(tmp_path)))
    files = _category(panel._tree, '来源与附件')
    files.setExpanded(True)
    directory = files.child(0)
    directory.setExpanded(True)
    assert directory.isExpanded()
    assert directory.childCount() == 0
    panel.apply_theme(True)
    directory = _category(panel._tree, '来源与附件').child(0)
    assert directory.isExpanded()
    assert directory.childCount() == 0


def test_files_view_double_click_file_opens(qapp, panel, tmp_path,
                                            monkeypatch):
    f = tmp_path / 'a.dat'
    f.write_text('x', encoding='utf-8')
    panel.set_project_info(
        types.SimpleNamespace(name='测试1', root_path=str(tmp_path)))
    files_cat = _category(panel._tree, '来源与附件')
    opened = []
    monkeypatch.setattr(
        'ui.widgets.file_tree_panel.QDesktopServices.openUrl',
        lambda url: opened.append(url))
    panel._on_item_double_clicked(files_cat.child(0), 0)
    assert len(opened) == 1


def test_files_view_context_menu_on_file(qapp, panel, tmp_path, monkeypatch):
    from PyQt6.QtCore import QPoint
    from qfluentwidgets import RoundMenu
    f = tmp_path / 'a.dat'
    f.write_text('x', encoding='utf-8')
    panel.set_project_info(
        types.SimpleNamespace(name='测试1', root_path=str(tmp_path)))
    files_cat = _category(panel._tree, '来源与附件')
    shown = []
    monkeypatch.setattr(RoundMenu, 'exec',
                        lambda self, *a, **k: shown.append(1))
    monkeypatch.setattr(panel._tree, 'itemAt', lambda p: files_cat.child(0))
    panel._on_context_menu(QPoint(5, 5))
    assert len(shown) == 1


def test_tree_shown_with_only_spatial_results(qapp, panel):
    # 无处理成果但有空间成果：成果分类有子节点、树可见
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([])
    panel.set_spatial_results([_spatial()])
    assert not panel._tree.isHidden()
    artifacts_cat = _category(panel._tree, '空间成果')
    assert artifacts_cat.text(1) == '1'
    assert artifacts_cat.childCount() >= 1


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


def test_collapse_shows_blank_strip(qapp, panel):
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00')])
    panel.set_current_line('L01')
    panel.set_collapsed(True, animate=False)
    assert panel.width() <= 20  # 细条
    assert not panel._expanded_view.isVisibleTo(panel)
    assert panel._strip_view.isVisibleTo(panel)
    assert panel._strip_line_label.isHidden()  # 竖排文字已删，细条留白
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
    assert saved['projectInterface'] is True  # 一维格式 {page: collapsed}


def test_collapse_memory_is_per_page(qapp, panel):
    """收起态按页独立记忆（一维 {page: collapsed}）。"""
    settings = _FakeSettings()
    panel.set_settings_manager(settings)
    panel.apply_page('projectInterface')
    panel._remember_current(True)   # 项目页：手动收起
    panel.apply_page('processingInterface')
    assert panel._collapsed
    panel.apply_page('projectInterface')
    assert panel._collapsed         # 项目页记忆被手动覆盖
    panel._remember_current(False)  # 再手动展开
    panel.apply_page('processingInterface')
    panel.apply_page('projectInterface')
    assert not panel._collapsed
    saved = settings.get('file_tree_page_states')
    assert saved['projectInterface'] is False


def test_settings_legacy_2d_format_reads_main_view_value(qapp, panel):
    """旧版二维设置 {page: {view: collapsed}} 读取时取测线视图的值。"""
    settings = _FakeSettings()
    settings.set('file_tree_page_states',
                 {'projectInterface': {'lines': False, 'artifacts': True},
                  'homeInterface': {'artifacts': True}})
    panel.set_settings_manager(settings)
    assert panel._collapsed_for('projectInterface') is False
    assert panel._collapsed_for('homeInterface') is True


def test_strip_text_blank_while_collapsed(qapp, panel):
    """收起细条无竖排文字：左坞仅此一面板，线号/面板名无辨识价值。"""
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L09')])
    panel.set_collapsed(True, animate=False)
    panel.set_current_line('L09')
    assert panel._strip_line_label.text() == ''
    assert panel._strip_line_label.isHidden()


def _strip_press_event() -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(9, 100), QPointF(9, 100),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier)


def test_strip_click_anywhere_expands(qapp, panel):
    """收起细条整条可点（兑现收起钮 tooltip 的「点击细条可展开」）。"""
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_collapsed(True, animate=False)
    assert panel.is_collapsed()
    assert panel.eventFilter(panel._strip_view, _strip_press_event()) is True
    assert not panel.is_collapsed()
    # 展开态下细条隐藏，过滤器不响应
    assert panel.eventFilter(panel._strip_view, _strip_press_event()) is False


def test_rebuild_preserves_user_collapsed_category(qapp, panel):
    """数据刷新不冲掉用户手动折叠的分类（按 key 快照恢复）。"""
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00')])
    panel.set_artifacts([_artifact('A1', 'L01')])
    tree = panel._tree
    assert tree.topLevelItem(0).isExpanded()
    tree.topLevelItem(0).setExpanded(False)    # 用户手动折叠「测线」
    panel.set_lines([_line('L01', '2026-09-16T01:00:00')])
    assert not tree.topLevelItem(0).isExpanded()   # 刷新不冲掉折叠态
    panel.set_artifacts([_artifact('A1', 'L01'), _artifact('A2', 'L01')])
    assert not tree.topLevelItem(0).isExpanded()   # 新成果到达也不强制展开


def test_rebuild_first_build_defaults(qapp, panel):
    """首次构建（树为空，无快照）：有内容的分类默认展开，空分类折叠。"""
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L01', '2026-09-16T01:00:00')])
    tree = panel._tree
    assert tree.topLevelItem(0).isExpanded()      # 测线（有内容）
    assert all(not tree.topLevelItem(index).isExpanded() for index in (1, 2, 3))
    assert tree.topLevelItem(0).child(0).isExpanded()  # 首次显示原始输入


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


def test_context_menu_readonly_navigation_for_blank_group_and_suppressed_when_busy(qapp, panel,
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
                        lambda p: panel._tree.topLevelItem(0))  # 分类行
    panel._on_context_menu(pos)
    monkeypatch.setattr(panel._tree, 'itemAt',
                        lambda p: panel._line_id_by_item['L01'])
    panel.set_busy(True)  # busy
    panel._on_context_menu(pos)
    assert len(shown) == 2


def test_context_menu_on_line_only_selects_tree_without_switching_input(qapp, panel,
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
    assert got == []
    assert panel._current_line_id == 'L01'
    assert panel._tree.currentItem() is leaf


def test_artifact_context_menu_shows_and_selects_without_emitting(
        qapp, panel, monkeypatch):
    """成果叶子右键：弹菜单 + 右键即选中；菜单项点击前不发任何信号。"""
    from PyQt6.QtCore import QPoint
    from qfluentwidgets import RoundMenu
    panel.set_project_info(types.SimpleNamespace(name='测试1'))
    panel.set_lines([_line('L02')])
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
    panel.set_lines([_line('L02')])
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
    assert [t for t, _ in actions] == [
        '打开结果标签', '从此结果继续处理', '修改此步参数并另建分支',
        '参数与来源', '打包到此步的结果…', '删除此结果及依赖成果…', '复制成果号',
    ]
    focus_got, delete_got = [], []
    panel.artifact_focus_requested.connect(lambda *a: focus_got.append(a))
    panel.artifact_delete_requested.connect(lambda *a: delete_got.append(a))
    for text, slot in actions:
        if text in ('打开结果标签', '删除此结果及依赖成果…', '复制成果号'):
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


def _v2_artifact(aid, parent='', *, run='run001', index=1, line_id='L01'):
    artifact = _artifact(aid, line_id)
    artifact.manifest = {'params': {
        'lineage_version': 2, 'run_group_id': run, 'run_step_index': index,
        'run_input_artifact_id': '', 'step_input_artifact_id': parent,
        'lineage': [{'method_id': 'dewow', 'params': {'window': 11},
                     'implementation_version': 'native-1.0'}] * index,
    }}
    return artifact


def _source_item(panel, aid):
    stack = panel._top_items()
    while stack:
        item = stack.pop()
        if item.data(0, _KIND_ROLE) == 'artifact' and item.data(0, Qt.ItemDataRole.UserRole) == aid:
            return item
        stack.extend(item.child(index) for index in range(item.childCount()))
    raise AssertionError(f'artifact {aid} is absent')


def _capture_source_menu(panel, monkeypatch, item):
    from PyQt6.QtCore import QPoint
    from qfluentwidgets import RoundMenu
    from ui.widgets import file_tree_panel as module
    actions = {}
    original = module.add_action

    def capture(menu, icon, text, slot, **kwargs):
        action = original(menu, icon, text, slot, **kwargs)
        actions[str(text)] = action
        return action

    monkeypatch.setattr(module, 'add_action', capture)
    monkeypatch.setattr(RoundMenu, 'exec', lambda *args, **kwargs: None)
    monkeypatch.setattr(panel._tree, 'itemAt', lambda point: item)
    panel._on_context_menu(QPoint(5, 5))
    return actions


def test_old_project_menu_cannot_delete_reused_ids(panel, monkeypatch):
    panel.set_project_info(types.SimpleNamespace(project_id='first', name='first'))
    panel.set_lines([_line('L01')])
    panel.set_artifacts([_artifact('A1', 'L01')])
    leaf = _find_by_kind(panel._tree, 'artifact')
    actions = _capture_source_menu(panel, monkeypatch, leaf)
    deleted = []
    panel.artifact_delete_requested.connect(lambda *args: deleted.append(args))
    panel.set_project_info(types.SimpleNamespace(project_id='second', name='second'))
    panel.set_lines([_line('L01')])
    panel.set_artifacts([_artifact('A1', 'L01')])
    actions['删除此结果及依赖成果…'].trigger()
    assert deleted == []


def test_deep_result_chain_survives_theme_rebuild(panel):
    panel.set_project_info(types.SimpleNamespace(project_id='deep', name='deep'))
    panel.set_lines([_line('L01')])
    artifacts = []
    for index in range(1050):
        artifact = _artifact(f'A{index:04d}', 'L01')
        artifact.manifest = {'params': {'lineage_version': 2,
                             'step_input_artifact_id': f'A{index - 1:04d}' if index else ''}}
        artifacts.append(artifact)
    panel.set_artifacts(artifacts)
    panel.apply_theme(True)
    assert sum(item.data(0, Qt.ItemDataRole.UserRole + 1) == 'artifact'
               for item in panel._items()) == 1050


def test_raw_click_activates_raw_source_without_line_selection_signal(panel):
    panel.set_project_info(types.SimpleNamespace(name='raw'))
    panel.set_lines([_line('L01')])
    raw = _find_by_kind(panel._tree, 'raw')
    activated, lines = [], []
    panel.raw_focus_requested.connect(activated.append)
    panel.line_selected.connect(lines.append)
    panel._on_item_clicked(raw, 0)
    assert activated == ['L01']
    assert lines == []


def test_raw_context_open_preserves_input_and_action_uses_captured_source(panel, monkeypatch):
    panel.set_project_info(types.SimpleNamespace(name='raw menu'))
    panel.set_lines([_line('L01'), _line('L02')])
    panel.set_current_line('L02')
    panel.set_active_source('L02', '')
    activated = []
    panel.raw_focus_requested.connect(activated.append)
    actions = _capture_source_menu(panel, monkeypatch, _find_by_kind(panel._tree, 'raw'))
    assert activated == []
    assert panel._current_line_id == 'L02'
    actions['打开原始输入'].trigger()
    assert activated == ['L01']


def test_active_source_marker_preserves_suffix_and_is_distinct_from_tree_selection(panel):
    panel.set_project_info(types.SimpleNamespace(name='marker'))
    panel.set_lines([_line('L01'), _line('L02', processed_result='result')])
    panel.set_artifacts([_v2_artifact('A')])
    artifact = _source_item(panel, 'A')
    base = artifact.text(1)
    panel.set_active_source('L01', 'A')
    assert artifact.text(1) == f'{base} 输入'
    assert artifact.font(0).bold()
    panel.set_current_line('L02')
    assert panel._tree.currentItem() is panel._line_id_by_item['L02']
    assert artifact.text(1) == f'{base} 输入'
    panel.apply_theme(True)
    assert _source_item(panel, 'A').text(1) == f'{base} 输入'
    panel.set_active_source('L02', '')
    assert _source_item(panel, 'A').text(1) == base
    assert not _source_item(panel, 'A').font(0).bold()
    assert panel._line_id_by_item['L02'].text(1) == '成果✓'
    assert panel._line_id_by_item['L02'].child(0).text(1) == '输入'


@pytest.mark.parametrize('aid,blocked', [('A', True), ('B', True), ('S', False)])
def test_running_input_blocks_ancestor_delete_but_allows_sibling_view(panel, monkeypatch, aid, blocked):
    panel.set_project_info(types.SimpleNamespace(name='running'))
    panel.set_lines([_line('L01')])
    panel.set_artifacts([_v2_artifact('A'), _v2_artifact('B', 'A', index=2),
                         _v2_artifact('S', run='sibling')])
    panel.set_processing_input('L01', 'B', True)
    assert panel._tree.isEnabled()
    actions = _capture_source_menu(panel, monkeypatch, _source_item(panel, aid))
    assert actions['打开结果标签'].isEnabled()
    assert actions['删除此结果及依赖成果…'].isEnabled() is not blocked
    if blocked:
        assert '运行' in actions['删除此结果及依赖成果…'].toolTip()


def test_running_line_delete_is_disabled_without_blocking_raw_view(panel, monkeypatch):
    panel.set_project_info(types.SimpleNamespace(name='line running'))
    panel.set_lines([_line('L01')])
    panel.set_processing_input('L01', '', True)
    actions = _capture_source_menu(panel, monkeypatch, panel._line_id_by_item['L01'])
    assert not actions['删除测线…'].isEnabled()
    assert not actions['处理该测线'].isEnabled()


@pytest.mark.parametrize('case,enabled', [('saved', True), ('legacy', False), ('unsaved', False)])
def test_reconfigure_requires_actual_single_step_source_and_info_does_not_activate(
        panel, monkeypatch, case, enabled):
    from PyQt6.QtWidgets import QDialog, QPlainTextEdit
    panel.set_project_info(types.SimpleNamespace(name='reconfigure'))
    panel.set_lines([_line('L01')])
    artifact = {'saved': _v2_artifact('B', 'A', index=2),
                'legacy': _artifact('B'),
                'unsaved': _v2_artifact('B', index=2)}[case]
    panel.set_artifacts([_v2_artifact('A'), artifact])
    focused, reconfigured, info = [], [], []
    panel.artifact_focus_requested.connect(lambda *args: focused.append(args))
    panel.artifact_reconfigure_requested.connect(lambda *args: reconfigured.append(args))
    panel.source_info_requested.connect(lambda *args: info.append(args))
    displayed = []
    monkeypatch.setattr(QDialog, 'exec', lambda dialog: displayed.append(
        dialog.findChild(QPlainTextEdit).toPlainText()))
    actions = _capture_source_menu(panel, monkeypatch, _source_item(panel, 'B'))
    assert actions['修改此步参数并另建分支'].isEnabled() is enabled
    assert focused == reconfigured == info == []
    actions['参数与来源'].trigger()
    assert info == [('L01', 'B')]
    assert '数据：B' in displayed[0]
    assert ('来源待核对' in displayed[0]) is (case == 'legacy')
    assert focused == []
    actions['修改此步参数并另建分支'].trigger()
    assert reconfigured == ([('L01', 'B')] if enabled else [])


def test_delete_rechecks_task_dependency_when_menu_was_opened_before_task(panel, monkeypatch):
    panel.set_project_info(types.SimpleNamespace(name='menu race'))
    panel.set_lines([_line('L01')])
    panel.set_artifacts([_v2_artifact('A')])
    actions = _capture_source_menu(panel, monkeypatch, _source_item(panel, 'A'))
    deleted = []
    panel.artifact_delete_requested.connect(lambda *args: deleted.append(args))
    panel.set_processing_input('L01', 'A', True)
    actions['删除此结果及依赖成果…'].trigger()
    assert deleted == []


@pytest.mark.parametrize('aid', ['', 'A'])
def test_running_placeholder_is_not_data_and_is_removed_on_finish(panel, aid):
    panel.set_project_info(types.SimpleNamespace(name='task placeholder'))
    panel.set_lines([_line('L01')])
    panel.set_artifacts([_v2_artifact('A')])
    panel.set_active_source('L01', aid)
    panel.set_processing_input('L01', aid, True)
    source = _source_item(panel, aid) if aid else _find_by_kind(panel._tree, 'raw')
    task = _find_by_kind(panel._tree, 'task')
    assert task is not None and task.parent() is source
    assert not task.flags() & Qt.ItemFlag.ItemIsSelectable
    assert not task.data(0, Qt.ItemDataRole.UserRole)
    activated = []
    panel.raw_focus_requested.connect(lambda *args: activated.append(args))
    panel.artifact_focus_requested.connect(lambda *args: activated.append(args))
    panel._on_item_clicked(task, 0)
    assert activated == []
    panel.apply_theme(True)
    assert _find_by_kind(panel._tree, 'task') is not None
    panel.set_processing_input('L01', aid, False)
    assert _find_by_kind(panel._tree, 'task') is None
    current = _source_item(panel, aid) if aid else _find_by_kind(panel._tree, 'raw')
    assert '输入' in current.text(1)


def test_no_active_data_clears_input_marker(panel):
    panel.set_project_info(types.SimpleNamespace(name='empty workspace'))
    panel.set_lines([_line('L01')])
    panel.set_active_source('L01', '')
    panel.set_active_source('', '')
    assert _find_by_kind(panel._tree, 'raw').text(1) == ''


def test_expanding_raw_arrow_does_not_activate_source(panel):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtTest import QTest
    panel.set_project_info(types.SimpleNamespace(name='branch arrow'))
    panel.set_lines([_line('L01')])
    panel.set_artifacts([_v2_artifact('A')])
    raw = _find_by_kind(panel._tree, 'raw')
    activated = []
    panel.raw_focus_requested.connect(activated.append)
    rect = panel._tree.visualItemRect(raw)
    pos = QPoint(2 * panel._tree.indentation() + 25, rect.center().y())
    was_expanded = raw.isExpanded()
    QTest.mouseClick(panel._tree.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    assert raw.isExpanded() is not was_expanded
    assert activated == []


def test_switch_project_drops_cached_sources_markers_and_task_placeholder(panel):
    panel.set_project_info(types.SimpleNamespace(project_id='first', name='first'))
    panel.set_lines([_line('L01')])
    panel.set_artifacts([_v2_artifact('A')])
    panel.set_active_source('L01', 'A')
    panel.set_processing_input('L01', 'A', True)
    panel.set_project_info(types.SimpleNamespace(project_id='second', name='second'))
    assert _find_by_kind(panel._tree, 'artifact') is None
    assert _find_by_kind(panel._tree, 'task') is None
    assert panel._active_source is None
    assert panel._processing_input is None


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
        panel.set_lines([_line(f'L{i:02d}') for i in range(1, 4)])
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
