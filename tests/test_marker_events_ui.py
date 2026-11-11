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


def marker_view(count=10000):
    doc = MarkerDocument('project', 'L01', 'raw:1', markers=tuple(
        MarkerRecord(uid=f'm{i}', name='同名', note=f'备注{i}', anchor=MarkerAnchor('raw:1', i % 20)) for i in range(count)))
    view = MarkerView(doc, 'raw:1', '', 20, {}, {}, (), False, False)
    return replace(view, rows=project_rows(view))


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
