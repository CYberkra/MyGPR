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
