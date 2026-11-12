"""Large annotation lists and strips use one identity without touching picks."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtTest import QSignalSpy
from core.gui_rendering import make_preview_bundle
from mygpr.application.project.marker_service import MarkerView, project_rows
from mygpr.domain.markers.models import MarkerDocument, MarkerRecord, MarkerAnchor, EventInterval
from ui.widgets.bscan_view import BScanView
from ui.widgets.marker_panel import MarkerPanel
from ui.widgets.marker_editor import MarkerEditor


def marker_view(count=10000):
    doc = MarkerDocument('project', 'L01', 'raw:1', markers=tuple(
        MarkerRecord(uid=f'm{i}', name='同名', note=f'备注{i}', anchor=MarkerAnchor('raw:1', i % 20)) for i in range(count)))
    view = MarkerView(doc, 'raw:1', '', 20, {}, {}, (), False, False)
    return replace(view, rows=project_rows(view))


def test_editing_time_only_or_open_end_keeps_source_version(qapp):
    view = marker_view(0)
    event = EventInterval(time_start='2026-01-01T00:00:00Z', time_end='2026-01-01T00:00:01Z',
                          time_source_id='raw:previous')
    editor = MarkerEditor(view, event)
    editor.note.setPlainText('补充备注')
    assert editor._draft().time_source_id == 'raw:previous'
    editor.close()
    editor = MarkerEditor(replace(view, timezone='Asia/Shanghai'), event)
    assert editor.time_start.text().startswith('2026-01-01T08:00:00')
    assert editor._draft().time_start == event.time_start
    editor.close()
    opened = EventInterval(end=MarkerAnchor('raw:previous', 2))
    editor = MarkerEditor(view, opened)
    draft = editor._draft()
    assert draft.start is None and draft.end == opened.end
    editor.close()


def test_marker_navigation_uses_artifact_and_rejects_late_raw_preview(qapp):
    from types import SimpleNamespace
    from ui.coordinator_markers import MarkerChain
    calls = []
    tools = SimpleNamespace(select=lambda uid, **kw: calls.append(('select', uid)))
    page = SimpleNamespace(_workspace=SimpleNamespace(
        open_source=lambda key: calls.append(('open', key)), marker_tools=tools))
    coordinator = SimpleNamespace(current_project_id=lambda: 'project',
        project=SimpleNamespace(on_line_selected=lambda line: True,
            on_artifact_focus_requested=lambda line, aid: calls.append(('request', aid))),
        goto_page=lambda page: None, page=lambda name: page)
    chain = MarkerChain(coordinator)
    chain.navigate('processing', {'line_id': 'L01', 'uid': 'm0', 'artifact_id': 'result-1'})
    raw = marker_view(1)
    chain._locate_loaded(SimpleNamespace(marker_view=raw))
    assert calls == [('request', 'result-1')]
    result = replace(raw, artifact_id='result-1', target_id='artifact:result-1')
    chain._locate_loaded(SimpleNamespace(marker_view=result))
    assert calls == [('request', 'result-1'), ('open', 'artifact:result-1'), ('select', 'm0')]


def test_strip_aligns_with_image_column_after_pan_reverse_and_reparent(qapp):
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtWidgets import QWidget, QVBoxLayout
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(137, 20, 35, 20)
    view = BScanView()
    layout.addWidget(view)
    view.set_bundle(replace(make_preview_bundle(np.ones((12, 20), np.float32)), marker_view=marker_view(1)))
    host.resize(1000, 600)
    host.show()
    for reversed_ in (False, True):
        view.set_view_reversed(reversed_)
        view._plot.vb.setXRange(2, 15, padding=0)
        qapp.processEvents()
        scene = view._image_item.mapToScene(QPointF(7.5, 0))
        expected = view._glw.viewport().mapToGlobal(view._glw.mapFromScene(scene)).x()
        strip_x = view.marker_strip.pixel_x(7)
        actual = view.marker_strip.mapToGlobal(QPoint(round(strip_x), 0)).x()
        assert abs(actual - expected) <= 1
        assert view.marker_strip.trace_at(QPoint(round(strip_x), 10)) == 7
    host.close()


def test_ten_thousand_records_search_and_cluster_without_truncation(qapp):
    context = marker_view()
    panel = MarkerPanel()
    panel.set_view(context)
    assert panel.model.rowCount() == 10000
    panel.search.setText('备注9999')
    assert panel.proxy.rowCount() == 1
    view = BScanView()
    bundle = make_preview_bundle(np.ones((12, 20), np.float32))
    view.set_bundle(replace(bundle, marker_view=context))
    view.resize(850, 450)
    view.show()
    qapp.processEvents()
    assert len(view.marker_strip.rows) == 10000
    represented = {uid for _, ids in view.marker_strip._hits for uid in ids}
    assert represented == {record.uid for record in context.document.markers}
    hits = view.marker_strip._hits
    assert all(not rect.intersects(other) for i, (rect, _) in enumerate(hits) for other, _ in hits[i + 1:])
    picks = QSignalSpy(view.sig_point_picked)
    view.select_event_marker('m9999')
    qapp.processEvents()
    assert any(ids == ['m9999'] for _, ids in view.marker_strip._hits)
    assert len(picks) == 0
    assert len(view._event_guides) == 1
    view.set_acquisition_markers_visible(False)
    assert view.marker_strip.isHidden()
    assert len(view.marker_strip.rows) == 10000
    view.clear()
    assert view._marker_view is None
    panel.close()
    view.close()


def test_marker_strip_shares_axis_title_even_with_overlapping_records(qapp):
    view = BScanView()
    view.resize(850, 450)
    view.show()
    bundle = make_preview_bundle(np.ones((12, 20), np.float32))
    view.set_bundle(replace(bundle, trace_axis_m=np.arange(20, dtype=float), marker_view=marker_view(21)))
    view.set_x_axis_mode('distance')
    view.apply_theme(True)
    qapp.processEvents()

    assert view._bottom_axis.isVisible()
    assert not view._bottom_axis.label.isVisible()
    assert view.marker_strip.accessibleDescription() == '距离 (m)'
    hits = view.marker_strip._hits
    assert any(len(ids) > 1 for _, ids in hits)
    assert len({rect.center().y() for rect, _ in hits}) == 1

    view.set_acquisition_markers_visible(False)
    assert view.marker_strip.isHidden()
    assert view._bottom_axis.label.isVisible()
    view.close()


def test_refresh_preserves_selected_identity_and_updated_guides(qapp):
    context = marker_view(3)
    panel = MarkerPanel()
    panel.set_view(context)
    panel.resize(650, 500)
    panel.show()
    qapp.processEvents()
    assert sum(panel.table.columnWidth(i) for i in range(4)) == panel.table.viewport().width()
    assert panel.table.columnWidth(0) > panel.table.columnWidth(3)
    panel.select_id('m2')
    updated = replace(context, document=replace(context.document, revision=1))
    panel.set_view(updated)
    assert [row['uid'] for row in panel._selected_rows()] == ['m2']
    view = BScanView()
    view.set_bundle(replace(make_preview_bundle(np.ones((12, 20), np.float32)), marker_view=context))
    view.select_event_marker('m2')
    view.set_marker_view(updated)
    assert view.marker_strip.selected_id == 'm2'
    assert len(view._event_guides) == 1
    panel.close()
    view.close()


def test_project_list_refresh_uses_saved_revision_for_next_preview(qapp):
    context = marker_view(2)
    event = EventInterval(start=context.anchor(0), end=context.anchor(1))
    context = replace(context, document=replace(context.document, intervals=(event,)))
    context = replace(context, rows=project_rows(context))
    panel = MarkerPanel()
    panel.set_view(context)
    panel.select_id(event.uid)
    updated = replace(context, document=replace(context.document, revision=1))
    panel._requests['refresh'] = ('project', {})
    panel.deliver('refresh', (updated,), '')
    calls = QSignalSpy(panel.operation)
    panel.preview('keep')
    assert calls[0][1]['view'].document.revision == 1
    assert calls[0][1]['ids'] == [event.uid]
    panel.close()


def test_marker_navigation_updates_interpretation_source_and_releases_browser(qapp):
    from types import SimpleNamespace
    from ui.pages.interpretation_page import InterpretationPage
    from ui.widgets.marker_browser import MarkerBrowserButton
    page = InterpretationPage()
    page.set_session_source('result-1')
    page.set_artifacts([SimpleNamespace(artifact_id='result-1', name='保留区间')])
    assert page._current_artifact_id() == 'result-1'
    page.set_session_source('')
    assert page._current_artifact_id() == ''
    browser = MarkerBrowserButton()
    browser.open_browser()
    browser.panel.set_view(marker_view(1))
    navigated = QSignalSpy(browser.navigate)
    browser._locate('m0')
    assert not browser.dialog.isVisible()
    assert navigated[0][0] == 'processing'
    browser.close()
    page.close()


def test_open_interval_and_duplicate_points_survive_list_selection(qapp):
    context = marker_view(3)
    doc = replace(context.document, intervals=(EventInterval(start=MarkerAnchor('raw:1', 4)),))
    context = replace(context, document=doc)
    context = replace(context, rows=project_rows(context))
    panel = MarkerPanel()
    panel.set_view(context)
    panel.select_id('m2')
    assert panel._selected_rows()[0]['uid'] == 'm2'
    assert context.rows[-1]['status'] == '未闭合'
    panel.close()


def test_editor_previews_boundary_inclusion_and_keeps_failed_draft(qapp):
    from ui.widgets.marker_editor import MarkerEditor
    context = marker_view(2)
    event = EventInterval(start=context.anchor(3), end=context.anchor(7))
    editor = MarkerEditor(context, event)
    assert '5 道' in editor.range_preview.text()
    editor.inc_end.setChecked(False)
    assert '4 道' in editor.range_preview.text()
    editor.note.setPlainText('保存失败后不能丢失')
    saved = QSignalSpy(editor.save_requested)
    editor._save()
    assert len(saved) == 1 and not saved[0][0].include_end
    editor.failed('写入失败')
    assert editor.note.toPlainText() == '保存失败后不能丢失'
    assert editor.buttons.isEnabled()
    editor.close()


def test_editor_keeps_save_buttons_inside_short_high_dpi_screen(qapp, monkeypatch):
    from types import SimpleNamespace
    from PyQt6.QtCore import QRect
    monkeypatch.setattr(MarkerEditor, 'screen', lambda self: SimpleNamespace(availableGeometry=lambda: QRect(0, 0, 800, 480)))
    context = marker_view(2)
    editor = MarkerEditor(context, EventInterval(start=context.anchor(0), end=context.anchor(1)))
    editor.show()
    qapp.processEvents()
    assert editor.height() <= 432
    assert editor.buttons.geometry().bottom() < editor.height()
    assert editor.buttons.isVisible()
    editor.close()


def test_ten_thousand_overlapping_intervals_remain_in_virtualized_cluster(qapp):
    from ui.widgets.marker_cluster import MarkerCluster
    context = marker_view(0)
    document = replace(context.document, intervals=tuple(EventInterval(name=f'区间{i}',
        start=context.anchor(i % 10), end=context.anchor(i % 10 + 3)) for i in range(10000)))
    context = replace(context, document=document)
    rows = project_rows(context)
    popup = MarkerCluster(rows)
    assert popup.model.rowCount() == 10000
    popup._filter('区间9999')
    assert popup.proxy.rowCount() == 1
    selected = QSignalSpy(popup.selected)
    popup._choose(popup.proxy.index(0, 0))
    assert selected[0][0] == document.intervals[9999].uid
    popup.close()
