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
    assert calls == [('request', 'result-1'), ('open', 'result-1'), ('select', 'm0')]


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
    picks = QSignalSpy(view.sig_point_picked)
    view.select_event_marker('m9999')
    assert len(picks) == 0
    assert len(view._event_guides) == 1
    view.set_acquisition_markers_visible(False)
    assert view.marker_strip.isHidden()
    assert len(view.marker_strip.rows) == 10000
    view.clear()
    assert view._marker_view is None
    panel.close()
    view.close()


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
