# -*- coding: utf-8 -*-
"""Standalone legacy container tests and current ProcessingPage integration.

The 2026-10-16 contract retires automatic focus/thumbnail layouts, pinned raw
tabs, result galleries and automatic run-group expansion. Explicit group and
drag/drop behavior lives in test_bscan_workspace.py; input/draft/job behavior
lives in test_processing_workspace_flow.py. These tests cover page wiring and
metadata/preview arrival across those components.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.pages.processing_page import ProcessingPage  # noqa: E402
from ui.widgets import (  # noqa: E402
    BScanContainer,
    LAYOUT_DUAL,
    LAYOUT_FOCUS,
    LAYOUT_MODES,
    LAYOUT_SINGLE,
    MAX_PANELS,
)


def _bundle(tag: float) -> SimpleNamespace:
    """构造鸭子类型 PreviewBundle：矩阵全为 tag，便于断言分发去向。"""
    mat = np.full((8, 6), float(tag), dtype=np.float32)
    return SimpleNamespace(
        matrix=mat, vmin=0.0, vmax=float(tag), title=f'b{tag}',
        x_label='道数', y_label='采样点', trace_axis_m=None, sample_axis=None,
        sample_axis_label='', trace_count=6, sample_count=8,
        trace_elevation_m=None, depth_axis_m=None)


def _artifact(aid: str, method: str, group: str, step: int,
              kind: str = 'intermediate', created: str = '2026-09-24T10:00:00'):
    """构造鸭子类型 ProjectArtifact（B7 中间成果：manifest.params 归组）。"""
    return SimpleNamespace(
        artifact_id=aid, line_id='L01',
        name=f'run 步骤{step}_{method}' if kind == 'intermediate' else f'run_{method}',
        method_id=method, method_name=method,
        created_at=created,
        manifest={'params': {'artifact_kind': kind, 'run_group_id': group,
                             'run_step_index': step}})


class TestContainerPanels:
    """容器本身：面板数解析 / 面板访问（哑组件契约）。"""

    @pytest.fixture
    def container(self, qapp):
        return BScanContainer()

    def test_default_is_single(self, container):
        assert container.effective_mode() == LAYOUT_SINGLE
        assert len(container.views()) == 1

    def test_panel_counts_follow_resolve(self, container):
        assert len(container.views()) == 1
        assert container.resolve_auto(2) is True
        assert len(container.views()) == 2
        assert container.effective_mode() == LAYOUT_DUAL
        assert container.resolve_auto(4) is True
        assert len(container.views()) == 4
        assert container.effective_mode() == LAYOUT_FOCUS
        assert container.resolve_auto(1) is True
        assert len(container.views()) == 1

    def test_resolve_auto_idempotent(self, container):
        assert container.resolve_auto(2) is True
        assert container.resolve_auto(2) is False        # 幂等
        assert container.resolve_auto(2) is False

    def test_resolve_auto_clamps_to_bounds(self, container):
        assert container.resolve_auto(99) is True
        assert len(container.views()) == MAX_PANELS == 4
        assert container.resolve_auto(0) is True
        assert container.effective_mode() == LAYOUT_SINGLE

    def test_focus_hides_thumbs_beyond_count(self, container, qapp):
        """n=3 → 主窗 + 2 缩略（第 4 格隐藏）；n=4 → 3 缩略全显。"""
        container.resolve_auto(3)
        for index in range(3):
            assert container.view_at(index).isVisibleTo(container)
        assert not container.view_at(3).isVisibleTo(container)
        assert len(container.thumb_views()) == 2
        container.resolve_auto(4)
        assert container.view_at(3).isVisibleTo(container)
        assert len(container.thumb_views()) == 3

    def test_thumb_views_empty_unless_focus(self, container):
        assert container.thumb_views() == []
        container.resolve_auto(2)
        assert container.thumb_views() == []

    def test_resolve_auto_does_not_emit_signal(self, container):
        """数据驱动的重排不是用户偏好变化：绝不发持久化镜像信号。"""
        got = []
        container.sig_layout_changed.connect(got.append)
        container.resolve_auto(2)
        container.resolve_auto(4)
        assert got == []

    def test_views_returns_copy(self, container):
        views = container.views()
        views.clear()
        assert len(container.views()) == 1

    def test_view_at_out_of_range_returns_primary(self, container):
        container.resolve_auto(4)
        assert container.view_at(99) is container.primary_view()

    def test_all_views_spans_pages(self, container):
        assert len(container.all_views()) == 1 + 2 + 4

    def test_modes_constant(self):
        assert LAYOUT_MODES == ('auto', 'single', 'dual', 'focus')


@pytest.fixture
def page(qapp):
    """A fresh workbench, with deferred Qt destruction between cases."""
    from PyQt6.QtCore import QCoreApplication, QEvent

    value = ProcessingPage()
    yield value
    value.close()
    value.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


class TestProcessingPageTabModel:
    """Catalog registration and asynchronous previews never imply tab opening."""

    def test_original_bundle_opens_one_ordinary_tab(self, page):
        raw = _bundle(1)
        page.set_original_bundle(raw)
        workspace = page._workspace
        assert workspace.tab_keys() == ['original']
        assert len(workspace.views()) == 1
        assert workspace.primary_view()._matrix is raw.matrix
        assert not workspace.is_pinned('original')
        assert workspace.close_tab('original')
        page.set_original_bundle(_bundle(2))
        assert workspace.tab_keys() == []
        assert not page._run_btn.isEnabled()

    def test_artifact_preview_refill_does_not_open_or_activate(self, page):
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([_artifact('A1', 'dewow', 'G1', 1)])
        result = _bundle(2)
        page.set_artifact_bundle('A1', result)
        assert page._workspace.tab_keys() == ['original']
        assert page._workspace.active_key() == 'original'
        assert page.select_artifact('A1')
        assert page._workspace.primary_view()._matrix is result.matrix
        assert len(page._workspace.views()) == 1

    def test_same_artifact_updates_in_place_without_duplicate_tabs(self, page):
        page.set_artifacts([_artifact('A1', 'dewow', 'G1', 1)])
        page.select_artifact('A1')
        page.set_artifact_bundle('A1', _bundle(2))
        replacement = _bundle(3)
        page.set_artifact_bundle('A1', replacement)
        page.select_artifact('A1')
        assert page._workspace.tab_keys() == ['artifact:A1']
        assert page._workspace.primary_view()._matrix is replacement.matrix

    def test_preview_requests_are_lazy_and_deduplicated_across_refresh(self, page):
        requested = []
        page.artifact_preview_requested.connect(requested.append)
        artifacts = [_artifact('A1', 'dewow', 'G1', 1),
                     _artifact('A2', 'gain', 'G1', 2)]
        page.set_artifacts(artifacts)
        assert requested == []
        page.select_artifact('A1')
        page.set_artifacts(artifacts)
        page.select_artifact('A1')
        assert requested == ['A1']
        page.set_artifact_bundle('A1', _bundle(2))
        page.select_artifact('A2')
        page.select_artifact('A1')
        assert requested == ['A1', 'A2']

    def test_closed_or_deleted_source_is_not_reopened_by_late_preview(self, page):
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([_artifact('A1', 'dewow', 'G1', 1)])
        page.select_artifact('A1')
        page.close_artifact_tab('A1')
        page.set_artifact_bundle('A1', _bundle(2))
        assert page._workspace.tab_keys() == ['original']
        page.select_artifact('A1')
        page.set_artifacts([])
        page.set_artifact_bundle('A1', _bundle(3))
        assert page._workspace.tab_keys() == ['original']
        assert not page.has_artifact('A1')
        assert not page.select_artifact('A1')

    def test_many_selected_tabs_use_one_view_without_reordering(self, page):
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([_artifact(f'A{i}', f'm{i}', 'G1', i)
                            for i in range(1, 8)])
        for i in range(1, 8):
            page.set_artifact_bundle(f'A{i}', _bundle(i + 1))
            page.select_artifact(f'A{i}')
        expected = ['original'] + [f'artifact:A{i}' for i in range(1, 8)]
        assert page._workspace.tab_keys() == expected
        assert len(page._workspace.all_views()) == 1
        page.select_original()
        assert page._workspace.primary_view()._matrix.max() == 1
        assert page._workspace.tab_keys() == expected

    def test_close_all_artifacts_preserves_open_raw_and_catalog(self, page):
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([_artifact('A1', 'dewow', 'G1', 1)])
        page.select_artifact('A1')
        page.close_all_artifact_tabs()
        assert page._workspace.tab_keys() == ['original']
        assert page.has_artifact('A1')
        assert page.select_artifact('A1')

    def test_registered_artifact_identity_drives_title_and_tooltip(self, page):
        art = _artifact('A1', 'dewow', 'G1', 1)
        art.manifest['params'].update(lineage_version=2, run_input_artifact_id='',
            lineage=[{'method_id': 'dewow', 'params': {'window': 31}}])
        page.set_artifacts([art])
        page.set_artifact_bundle('A1', _bundle(2))
        page.select_artifact('A1')
        bar = page._workspace.tab_bar()
        assert art.name in bar.tabText(0)
        assert '低频漂移校正' in bar.tabText(0)
        assert 'G1.1' not in bar.tabText(0)
        assert '结果号：A1' in bar.tabToolTip(0)
        assert 'window' in bar.tabToolTip(0) and '31' in bar.tabToolTip(0)
        assert bar.tabText(0) != 'b2'  # bitmap title does not identify saved data

    def test_title_html_has_explicit_theme_color(self, page):
        """Retain readable rich-text titles in the active workspace canvas."""
        html = page._workspace.primary_view()._compact_title_html('X')
        assert 'color:#' in html

    def test_levels_row_removed_from_page(self, page):
        for attr in ('_cmap_combo', '_p_low_spin', '_p_high_spin',
                     '_apply_colormap', '_refresh_levels', '_sync_view_levels'):
            assert not hasattr(page, attr), f'{attr} should remain retired'

    def test_home_colormap_row_removed(self, qapp):
        from ui.pages.home_page import HomePage
        home = HomePage()
        try:
            assert not hasattr(home, '_cmap_combo')
            for attr in ('colormap', 'set_colormap'):
                assert not hasattr(home, attr)
        finally:
            home.close()
            home.deleteLater()

    def test_new_bundle_keeps_view_level_preference(self, page):
        page.set_original_bundle(_bundle(1))
        view = page._workspace.primary_view()
        view.set_display_levels(5.0, 95.0, notify=False)
        page.set_original_bundle(_bundle(3))
        assert view.display_levels() == (5.0, 95.0)

    def test_line_display_dedup(self, page):
        page.set_lines([SimpleNamespace(line_id='L01', name='L01')])
        assert page._line_combo.currentText() == 'L01'
        page.set_lines([SimpleNamespace(line_id='L02', name='测线二')])
        assert page._line_combo.currentText() == 'L02 测线二'


class TestExplicitWorkspaceLayout:
    """The old TestFocusLayout auto-thumbnails contract is retired."""

    def test_explicit_split_shares_bundle_but_preserves_independent_view_state(self, page):
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([_artifact('A1', 'dewow', 'G1', 1)])
        result = _bundle(2)
        page.set_artifact_bundle('A1', result)
        page.select_artifact('A1')
        workspace = page._workspace
        workspace.split_source('artifact:A1', 'right', copy=True)
        first, second = workspace.views()
        first.set_display_levels(4, 92)
        second.set_display_levels(11, 81)
        replacement = _bundle(3)
        page.set_artifact_bundle('A1', replacement)
        assert first._matrix is second._matrix is replacement.matrix
        assert first.display_levels() == (4.0, 92.0)
        assert second.display_levels() == (11.0, 81.0)
        page.set_artifacts([])
        assert 'artifact:A1' not in workspace.tab_keys()
        assert len(workspace.views()) == 1

    def test_line_switch_rejects_previous_line_preview(self, page):
        page.set_line_label('L01')
        page.set_original_bundle(_bundle(1))
        page.set_artifacts([_artifact('A1', 'dewow', 'G1', 1)])
        page.select_artifact('A1')
        page.set_line_label('L02')
        page.set_original_bundle(_bundle(9))
        page.set_artifact_bundle('A1', _bundle(2))
        assert not page.has_artifact('A1')
        assert page._workspace.tab_keys() == ['original']
        assert page._workspace.primary_view()._matrix.max() == 9
        assert page.current_pipeline()['input_artifact_id'] == ''


class TestRunGroupCatalog:
    """Old automatic run-group expansion is replaced by explicit navigation."""

    @pytest.mark.parametrize('flat', [False, True], ids=['nested', 'field-flat'])
    def test_group_metadata_refresh_never_opens_saved_steps(self, page, flat):
        page.set_original_bundle(_bundle(1))
        artifacts = [_artifact('S1', 'dewow', 'G1', 1),
                     _artifact('F1', 'bandpass', 'G1', 2, kind='processing')]
        if flat:
            for art in artifacts:
                art.manifest = art.manifest['params']
        page.set_artifacts(artifacts)
        page.set_artifacts(artifacts)
        assert page._workspace.tab_keys() == ['original']
        page.select_artifact('F1')
        page.set_artifacts(artifacts)
        assert page._workspace.tab_keys() == ['original', 'artifact:F1']
        assert page._workspace.active_key() == 'artifact:F1'
        assert page._history_tree.topLevelItemCount() == 1
        assert page._history_tree.topLevelItem(0).childCount() == 2

    def test_legacy_result_is_explicitly_openable_without_invented_lineage(self, page):
        page.set_artifacts([SimpleNamespace(
            artifact_id='OLD', name='旧成果', method_id='dewow',
            created_at='2026-09-24T08:00:00')])
        assert page._workspace.tab_keys() == []
        assert page.select_artifact('OLD')
        assert page.current_pipeline()['input_artifact_id'] == 'OLD'
        assert not page.can_reconfigure_artifact('OLD')

    def test_latest_command_requires_completed_job_identity(self, page):
        page.set_original_bundle(_bundle(1))
        old = _artifact('OLD', 'old', 'G0', 1)
        page.set_artifacts([old])
        page.show_latest_result()
        assert page._workspace.active_key() == 'original'
        page.set_running(True, 'job-mine')
        page.set_run_result('job-mine', 'MINE')
        page.select_artifact('OLD')  # user changed browsing intent during job
        page.set_running(False, success=True)
        page.set_artifacts([old, _artifact('MINE', 'dewow', 'G1', 1),
                            _artifact('NEWER', 'gain', 'G2', 1,
                                      created='2026-10-16T23:59:00')])
        assert page._workspace.active_key() == 'artifact:OLD'
        page._completion_btn.click()
        assert page._workspace.active_key() == 'artifact:MINE'
        assert 'artifact:NEWER' not in page._workspace.tab_keys()
        assert len(page._workspace.views()) == 1
