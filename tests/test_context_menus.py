# -*- coding: utf-8 -*-
"""右键菜单/快捷交互相关测试（offscreen QApplication）。"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ------------------------------------------------------------------ line_source_path
class TestLineSourcePath:
    """ProjectController.line_source_path：读 raw/<line_id>/import_manifest.json。"""

    def _controller(self, root):
        from ui.controllers.project_controller import ProjectController
        controller = ProjectController()
        controller._current = SimpleNamespace(root_path=str(root))
        return controller

    def test_roundtrip(self, tmp_path):
        manifest = tmp_path / 'raw' / 'L01' / 'import_manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(
            {'schema': 'mygpr.import_manifest.v2', 'line_id': 'L01',
             'source_path': 'D:/data/营山/L01.csv'}), encoding='utf-8')
        controller = self._controller(tmp_path)
        assert controller.line_source_path('L01') == 'D:/data/营山/L01.csv'

    def test_missing_manifest_returns_none(self, tmp_path):
        controller = self._controller(tmp_path)
        assert controller.line_source_path('L99') is None

    def test_invalid_json_returns_none(self, tmp_path):
        manifest = tmp_path / 'raw' / 'L01' / 'import_manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{not json', encoding='utf-8')
        controller = self._controller(tmp_path)
        assert controller.line_source_path('L01') is None

    def test_empty_source_path_returns_none(self, tmp_path):
        manifest = tmp_path / 'raw' / 'L01' / 'import_manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'line_id': 'L01', 'source_path': ''}),
                            encoding='utf-8')
        controller = self._controller(tmp_path)
        assert controller.line_source_path('L01') is None

    def test_no_project_returns_none(self):
        from ui.controllers.project_controller import ProjectController
        controller = ProjectController()
        assert controller.line_source_path('L01') is None


# ------------------------------------------------------------------ BScanView 色标信号
def test_bscan_choose_colormap_emits_signal(qapp):
    """右键菜单选色标：应用到视图 + 发 sig_colormap_changed 供页面同步。"""
    from ui.widgets.bscan_view import BScanView
    view = BScanView()
    received = []
    view.sig_colormap_changed.connect(received.append)
    view._choose_colormap('viridis')
    assert received == ['viridis']
    assert view._cmap_name == 'viridis'


def test_bscan_native_menu_disabled(qapp):
    """接入自定义右键菜单后 pyqtgraph 原生菜单须关闭。"""
    from ui.widgets.bscan_view import BScanView
    view = BScanView()
    assert not view._plot.vb.menuEnabled()


# ------------------------------------------------------------------ PipelineList
def test_pipeline_toggle_enabled(qapp):
    """右键"启用/禁用"切换：状态翻转 + sig_changed。"""
    from ui.widgets.pipeline_list import PipelineList
    widget = PipelineList()
    widget.set_steps([{'method_id': 'gain', 'label': '增益',
                       'params': {}, 'enabled': True}])
    changed = []
    widget.sig_changed.connect(lambda: changed.append(1))
    widget._toggle_enabled(0)
    assert widget.steps()[0]['enabled'] is False
    assert changed == [1]
    widget._toggle_enabled(0)
    assert widget.steps()[0]['enabled'] is True


def test_pipeline_delete_shortcut_registered(qapp):
    """Delete 快捷键已注册在处理链列表上。"""
    from ui.widgets.pipeline_list import PipelineList
    widget = PipelineList()
    assert widget._delete_shortcut is not None
    assert widget._delete_shortcut.parent() is widget._list


# ------------------------------------------------------------------ MapView
def test_map_view_fit_to_tracks_uses_summaries(qapp):
    """fit_to_tracks 从 track_summaries 取范围（右键"适应全部测线"路径）。"""
    import numpy as np
    from ui.widgets.map_view import MapView
    view = MapView()
    view._track_summaries = [
        {'xs': np.array([0.0, 1000.0]), 'ys': np.array([0.0, 500.0])}]
    view.fit_to_tracks()
    rect = view._plot.vb.viewRect()
    assert rect.width() >= 1000.0 and rect.height() >= 500.0


def test_map_view_native_menu_disabled(qapp):
    from ui.widgets.map_view import MapView
    view = MapView()
    assert not view._plot.vb.menuEnabled()


# ------------------------------------------------------------------ MethodBrowser
def test_method_browser_context_menu_uses_method_id(qapp):
    """右键菜单依赖 _method_id_of：分类行返回 None、方法行返回 id。"""
    from ui.widgets.method_browser import MethodBrowser
    browser = MethodBrowser()
    browser.set_methods([{'method_id': 'agc', 'display_name': 'AGC 增益',
                          'category': 'gain', 'category_label': '增益',
                          'tags': [], 'parameter_schema': []}])
    top = browser._tree.topLevelItem(0)
    assert browser._method_id_of(top) is None
    assert browser._method_id_of(top.child(0)) == 'agc'


# ------------------------------------------------------------------ ProjectPage resolver
def test_project_page_resolver_default_none(qapp):
    """未注入 resolver 时右键菜单路径项应按无路径处理（不抛异常）。"""
    from ui.pages.project_page import ProjectPage
    page = ProjectPage()
    assert page._source_path_resolver is None
