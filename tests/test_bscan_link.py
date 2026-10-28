import pytest
pytest.importorskip('PyQt6')

import numpy as np
from core.gui_rendering import PreviewBundle


def test_source_ids_preserve_uint64_and_are_separate_from_indices():
    ids = np.array([2**53 + 1, 2**53 + 9], dtype=np.uint64)
    bundle = PreviewBundle(np.zeros((3, 2)), -1, 1, 3, 2, source_trace_ids=ids)
    assert bundle.source_trace_ids.dtype == np.uint64
    assert bundle.source_trace_ids.tolist() == ids.tolist()


def test_mapping_uses_ids_and_preserves_time_origin():
    from ui.widgets.bscan_link import map_point
    from types import SimpleNamespace
    a = SimpleNamespace(_source_trace_ids=np.array([2**53+1, 2**53+2, 2**53+3], dtype=np.uint64), _trace_axis_m=None, _sample_axis=np.array([100., 150., 190.]), _sample_axis_label='时间 (ns)', _showing_elevation=False)
    b = SimpleNamespace(_source_trace_ids=np.array([2**53+2, 2**53+3], dtype=np.uint64), _trace_axis_m=None, _sample_axis=np.array([100., 130., 160., 190.]), _sample_axis_label='时间 (ns)', _showing_elevation=False)
    assert np.allclose(map_point(a, b, 1.5, 1.5), (0.5, 2.1666666667))
    assert map_point(a, b, 0.5, 1.5) is None
    b._sample_axis_label = '深度 (m)'
    assert map_point(a, b, 1.5, 1.5) is None


def test_link_requires_shared_ids_and_overlapping_time():
    from ui.widgets.bscan_link import compatible
    from types import SimpleNamespace
    def view(ids, axis):
        return SimpleNamespace(_source_trace_ids=np.array(ids, dtype=np.uint64), _trace_axis_m=None, _sample_axis=np.array(axis), _sample_axis_label='时间 (ns)', _showing_elevation=False)
    assert compatible(view([1, 2], [100., 190.]), view([2, 3], [130., 200.]))
    assert not compatible(view([1, 2], [100., 190.]), view([3, 4], [130., 200.]))
    assert not compatible(view([1, 2], [100., 190.]), view([2, 3], [200., 300.]))


def test_view_direction_does_not_reverse_data(qapp):
    from ui.widgets.bscan_view import BScanView
    view = BScanView()
    matrix = np.arange(6).reshape(2, 3)
    view.set_matrix(matrix, 0, 5)
    view.set_view_reversed(True)
    assert view._plot.vb.state['xInverted']
    assert np.array_equal(view._matrix, matrix)


def test_link_range_clips_to_shared_extent():
    from ui.widgets.bscan_link import map_range
    from types import SimpleNamespace
    def view(ids, axis):
        return SimpleNamespace(_source_trace_ids=np.array(ids, dtype=np.uint64), _trace_axis_m=None, _sample_axis=np.array(axis), _sample_axis_label='时间 (ns)', _showing_elevation=False)
    a = view([1, 2, 3, 4], [0., 100., 190., 250.])
    b = view([2, 3, 4], [100., 145., 190.])
    ranges = map_range(a, b, ((0, 4), (0, 4)))
    assert ranges is not None
    assert np.allclose(ranges[1], (0, 3))


def test_workspace_links_only_when_enabled_and_stops_on_close(qapp):
    from ui.widgets.bscan_workspace import BScanWorkspace
    ids = np.array([10, 11, 12, 13], dtype=np.uint64)
    bundle = PreviewBundle(np.zeros((4, 4)), -1, 1, 4, 4, source_trace_ids=ids,
                           sample_axis=np.array([100., 130., 160., 190.]), sample_axis_label='时间 (ns)')
    workspace = BScanWorkspace()
    workspace.set_sources([dict(key='a', bundle=bundle, project_id='P', line_id='L')])
    workspace.open_source('a')
    workspace.split_source('a', 'right', copy=True)
    groups = [workspace.group_widget(gid) for gid in workspace.group_ids()]
    assert len(groups) == 2
    workspace._link_enabled = True
    groups[0].view._plot.vb.setRange(xRange=(.5, 2.5), yRange=(.5, 2.5), padding=0)
    qapp.processEvents()
    assert np.allclose(groups[1].view._plot.vb.viewRange(), [[.5, 2.5], [.5, 2.5]])
    workspace._link_enabled = False
    before = groups[1].view._plot.vb.viewRange()
    groups[0].view._plot.vb.setRange(xRange=(1.5, 3.5), yRange=(1.5, 3.5), padding=0)
    assert np.allclose(groups[1].view._plot.vb.viewRange(), before)
    workspace.close()
    workspace.deleteLater()
    qapp.processEvents()



def test_reversed_bundle_preserves_source_ids(qapp):
    from ui.widgets.bscan_view import BScanView
    ids = np.array([2**53 + 1, 2**53 + 2], dtype=np.uint64)
    bundle = PreviewBundle(np.arange(6).reshape(3, 2), 0, 5, 3, 2,
                           source_trace_ids=ids, view_reversed=True)
    view = BScanView()
    view.set_bundle(bundle)
    assert view._plot.vb.state['xInverted']
    assert np.array_equal(view._source_trace_ids, ids)
    assert np.array_equal(view._matrix, bundle.matrix)
    view.deleteLater()
    qapp.processEvents()


def test_link_range_preserves_pixel_boundaries_and_single_trace():
    from ui.widgets.bscan_link import map_range
    from types import SimpleNamespace
    v = SimpleNamespace(_source_trace_ids=np.arange(4, dtype=np.uint64), _trace_axis_m=None,
                        _sample_axis=np.array([100., 130., 160., 190.]), _sample_axis_label='时间 (ns)', _showing_elevation=False)
    assert np.allclose(map_range(v, v, ((0, 4), (0, 4))), ((0, 4), (0, 4)))
    assert np.allclose(map_range(v, v, ((1, 2), (1, 3))), ((1, 2), (1, 3)))


def test_selection_to_processing_checks_identity_and_uses_current_indices(qapp):
    from PyQt6.QtCore import QRectF
    from ui.pages.processing_page import ProcessingPage
    from types import SimpleNamespace
    page = ProcessingPage()
    page.set_project_context('P')
    page.set_lines([SimpleNamespace(line_id='L', name='L')])
    page.set_line_label('L')
    bundle = PreviewBundle(np.zeros((4, 5)), -1, 1, 4, 5,
                           source_trace_ids=np.array([100, 102, 105, 110, 112], dtype=np.uint64))
    page.set_original_bundle(bundle)
    group = page._workspace.group_widget()
    group.view.set_view_selection(QRectF(1, 0, 2, 4))
    assert page._selection_to_step(group.group_id, 'original', 'trace_cut')
    assert page._pipeline_list.steps()[-1]['params'] == {'trace_start': 2, 'trace_end': 3}
    page._pipeline_list.set_steps([])
    assert page._selection_to_step(group.group_id, 'original', 'trace_qc')
    assert page._pipeline_list.steps()[-1]['params']['manual_trace_indices'] == '1-2'
    page.set_project_context('other')
    assert not page._selection_to_step(group.group_id, 'original', 'trace_cut')
    page.close()
    page.deleteLater()
    qapp.processEvents()



def test_selection_adds_absolute_time_window(qapp):
    from PyQt6.QtCore import QRectF
    from ui.pages.processing_page import ProcessingPage
    from types import SimpleNamespace
    page = ProcessingPage()
    page.set_project_context('P')
    page.set_lines([SimpleNamespace(line_id='L', name='L')])
    page.set_line_label('L')
    bundle = PreviewBundle(np.zeros((10, 5)), -1, 1, 10, 5,
                           sample_axis=np.arange(100., 200., 10.), sample_axis_label='时间 (ns)')
    page.set_original_bundle(bundle)
    group = page._workspace.group_widget()
    group.view.set_view_selection(QRectF(1, 2, 2, 3))
    assert page._selection_to_step(group.group_id, 'original', 'trace_cut_time')
    steps = page._pipeline_list.steps()
    assert steps[1]['params'] == {'mode': 'keep_range', 'time_start_ns': 120., 'time_end_ns': 150.}
    page.close()
    page.deleteLater()
    qapp.processEvents()


def test_qc_marks_and_focus_do_not_change_data(qapp):
    from ui.widgets.bscan_view import BScanView
    view = BScanView()
    matrix = np.arange(20).reshape(4, 5)
    view.set_matrix(matrix, 0, 20)
    assert view.set_qc_bad_mask([False, True, True, False, False])
    assert view.focus_trace_range(1, 3)
    assert np.array_equal(view._matrix, matrix)
    assert not view.focus_trace_range(-1, 3)
    view.deleteLater()
    qapp.processEvents()


def test_bundle_restores_saved_qc_marks(qapp):
    from ui.widgets.bscan_view import BScanView
    bundle = PreviewBundle(np.zeros((4, 3)), -1, 1, 4, 3, trace_qc_bad_mask=[0, 1, 0])
    view = BScanView()
    view.set_bundle(bundle)
    assert len(view._qc_marks) == 1
    view.set_bundle(PreviewBundle(np.zeros((4, 3)), -1, 1, 4, 3))
    assert not view._qc_marks
    view.deleteLater()
    qapp.processEvents()


def test_disabling_link_hides_crosshair_and_closing_workspace_disables(qapp):
    from PyQt6.QtCore import QPointF
    from ui.widgets.bscan_workspace import BScanWorkspace
    ids = np.arange(4, dtype=np.uint64)
    bundle = PreviewBundle(np.zeros((4, 4)), -1, 1, 4, 4, source_trace_ids=ids,
                           sample_axis=np.arange(4.), sample_axis_label='时间 (ns)')
    workspace = BScanWorkspace()
    workspace.set_sources([dict(key='a', bundle=bundle, project_id='P', line_id='L')])
    workspace.open_source('a')
    workspace.split_source('a', 'right', copy=True)
    workspace.show()
    qapp.processEvents()
    group = workspace.group_widget()
    menu = workspace.create_context_menu(group.group_id, 'a')
    action = next(item for item in menu.actions() if item.data() == 'link-views')
    action.trigger()
    assert workspace._link_enabled
    views = workspace.views()
    views[0]._vline.show()
    views[1]._hline.show()
    action.trigger()
    assert not workspace._link_enabled
    assert all(not view._vline.isVisible() and not view._hline.isVisible() for view in views)
    workspace._link_cursor(group, group.view._plot.vb.mapViewToScene(QPointF(1, 1)))
    assert all(not view._vline.isVisible() and not view._hline.isVisible() for view in views)
    workspace._set_link_enabled(True)
    workspace.close()
    assert not workspace._link_enabled
    workspace.deleteLater()
    qapp.processEvents()


def test_refresh_line_previews_invalidates_all_same_line_sources(qapp):
    from types import SimpleNamespace
    from ui.pages.processing_page import ProcessingPage
    page = ProcessingPage()
    page.set_project_context('P')
    page.set_lines([SimpleNamespace(line_id='L', name='L'), SimpleNamespace(line_id='other', name='other')])
    page.set_line_label('L')
    old = PreviewBundle(np.zeros((4, 3)), -1, 1, 4, 3)
    page.set_original_bundle(old)
    page._sources['artifact:A'] = dict(key='artifact:A', artifact_id='A', bundle=old, revision='1')
    page._sources['artifact:B'] = dict(key='artifact:B', artifact_id='B', bundle=old, revision='1')
    comparisons = page._comparison.sources
    same = f'compare:{page._comparison.epoch}:L:raw'
    other = f'compare:{page._comparison.epoch}:other:raw'
    comparisons[same] = dict(key=same, line_id='L', artifact_id='', bundle=old, comparison=True)
    comparisons[other] = dict(key=other, line_id='other', artifact_id='', bundle=old, comparison=True)
    page._sync_sources()
    seen = []
    page.original_preview_requested.connect(lambda: seen.append('raw'))
    page.artifact_preview_requested.connect(lambda aid: seen.append(aid))
    page.comparison_preview_requested.connect(lambda _p, line, _token: seen.append('comparison:' + line))
    before = page._draft_key()
    page.refresh_line_previews('L')
    assert set(seen) == {'raw', 'A', 'B', 'comparison:L'}
    assert all(source['bundle'] is None for source in page._sources.values())
    assert comparisons[same]['bundle'] is None and comparisons[other]['bundle'] is old
    updated = PreviewBundle(np.zeros((4, 3)), -1, 1, 4, 3, view_reversed=True, trace_axis_m=[0, 20, 40])
    token = page._comparison.pending[same]
    page.set_comparison_preview('P', 'L', token, updated, '')
    assert comparisons[same]['bundle'].view_reversed
    assert comparisons[same]['bundle'].trace_axis_m[-1] == 40
    assert page._draft_key() == before
    page.close()
    page.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize('command', ['selection-crop', 'selection-qc', 'selection-crop-time'])
def test_stale_selection_menu_cannot_apply_to_new_revision(qapp, command):
    from PyQt6.QtCore import QRectF
    from types import SimpleNamespace
    from ui.pages.processing_page import ProcessingPage
    page = ProcessingPage()
    page.set_project_context('P')
    page.set_lines([SimpleNamespace(line_id='L', name='L')])
    page.set_line_label('L')
    page.set_original_bundle(PreviewBundle(np.zeros((4, 5)), -1, 1, 4, 5,
                                          sample_axis=np.arange(4.), sample_axis_label='时间 (ns)'))
    group = page._workspace.group_widget()
    group.view.set_view_selection(QRectF(1, 0, 2, 4))
    menu = page._workspace.create_context_menu(group.group_id, 'original')
    action = next(a for a in menu.actions() if a.data() == command)
    assert action.isEnabled()
    page.set_original_revision('P', 'L', 'new')
    page.set_original_bundle(PreviewBundle(np.ones((4, 5)), -1, 1, 4, 5))
    group.view.set_view_selection(QRectF(2, 0, 2, 4))
    action.trigger()
    assert not page._pipeline_list.steps()
    page.close()
    page.deleteLater()
    qapp.processEvents()
