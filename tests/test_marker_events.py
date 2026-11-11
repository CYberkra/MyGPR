"""Field events retain identity and never invent acquisition samples."""
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from mygpr.domain.markers.models import MarkerAnchor, MarkerDocument, MarkerRecord, EventInterval
from mygpr.domain.markers.operations import import_markers, put_record, remove_record
from mygpr.domain.markers.ranges import interval_indices, selection_indices
from mygpr.domain.markers.mapping import TraceMapping, project_anchor


def document():
    return import_markers('project', 'L01', 'raw:one', [
        {'marker_id': i, 'trace_no': 3, 'label': '停车', 'marker_type': 'general'}
        for i in (1, 2)
    ])


def test_identical_labels_keep_distinct_stable_identity():
    first, second = document().markers
    assert first.uid != second.uid
    assert document().markers[0].uid == first.uid
    assert document().markers[0].anchor.trace == 2


def test_source_record_is_immutable_and_revisions_invalidate_interval():
    doc = document()
    mark = doc.markers[0]
    interval = EventInterval(uid='event', name='停车等待', start=mark.anchor,
                             end=MarkerAnchor('raw:one', 5),
                             start_marker_id=mark.uid, start_marker_revision=mark.revision)
    doc = put_record(doc, interval)
    edited = replace(mark, name='等车', note='车辆通过')
    updated = put_record(doc, edited)
    assert updated.markers[0].original['label'] == '停车'
    assert updated.intervals[0].review_state == 'needs_review'
    with pytest.raises(ValueError, match='采集'):
        remove_record(updated, mark.uid)


def test_roundtrip_and_duplicate_identity_rejection():
    doc = document()
    assert MarkerDocument.from_dict(doc.to_dict()) == doc
    payload = doc.to_dict()
    payload['markers'].append(payload['markers'][0])
    with pytest.raises(ValueError, match='重复'):
        MarkerDocument.from_dict(payload)


@pytest.mark.parametrize('include_start,include_end,expected', [
    (True, True, [2, 3, 4]), (False, True, [3, 4]),
    (True, False, [2, 3]), (False, False, [3]),
])
def test_boundary_inclusion(include_start, include_end, expected):
    item = EventInterval(uid='e', start=MarkerAnchor('raw', 2),
                         end=MarkerAnchor('raw', 4),
                         include_start=include_start, include_end=include_end)
    np.testing.assert_array_equal(interval_indices(item, TraceMapping.identity('raw', 8)), expected)


def test_no_acquisition_during_wait_does_not_snap_to_neighbour():
    item = EventInterval(uid='e', time_source_id='raw', time_start='2026-11-11T00:00:02+00:00',
                         time_end='2026-11-11T00:00:05+00:00')
    origin = datetime.fromisoformat('2026-11-11T00:00:00+00:00').timestamp()
    times = origin + np.array([0, 1, 6], dtype=np.float64)
    result = interval_indices(item, TraceMapping.identity('raw', 3),
                              timestamps_s=times, time_basis_confirmed=True)
    assert result.size == 0
    with pytest.raises(ValueError, match='时间'):
        interval_indices(item, TraceMapping.identity('raw', 3), timestamps_s=times)


def test_time_reversal_and_open_interval_refuse_processing():
    item = EventInterval(uid='e', time_source_id='raw', time_start='2026-11-11T00:00:02Z',
                         time_end='2026-11-11T00:00:05Z')
    with pytest.raises(ValueError, match='时间'):
        interval_indices(item, TraceMapping.identity('raw', 3),
                         timestamps_s=np.array([3, 2, 4]), time_basis_confirmed=True)
    with pytest.raises(ValueError, match='闭合'):
        interval_indices(EventInterval(uid='e', start=MarkerAnchor('raw', 1)),
                         TraceMapping.identity('raw', 3))


def test_removed_anchor_is_not_assigned_to_adjacent_column():
    mapping = TraceMapping('raw', 'result', np.array([0., 1., 5., 6.]),
                           np.array([0, 0, 1, 1]), kind='selection')
    assert project_anchor(MarkerAnchor('raw', 3), mapping).status == 'outside'
    assert project_anchor(MarkerAnchor('raw', 5), mapping).positions == (2.0,)


def test_overlap_union_and_no_empty_output():
    np.testing.assert_array_equal(selection_indices(8, [np.arange(1, 4), np.arange(3, 6)], 'keep'),
                                  np.arange(1, 6))
    np.testing.assert_array_equal(selection_indices(8, [np.arange(1, 6)], 'exclude'), [0, 6, 7])
    with pytest.raises(ValueError, match='零道'):
        selection_indices(8, [np.arange(8)], 'exclude')


def test_resampling_is_explicit_and_does_not_interpolate_identity():
    mapping = TraceMapping('raw', 'resampled', np.array([0., 2., 4.]),
                           np.zeros(3, dtype=np.int64), kind='resampled')
    projection = project_anchor(MarkerAnchor('raw', 1), mapping)
    assert projection.status == 'interpolated'
    assert projection.positions == (0.5,)
    assert project_anchor(MarkerAnchor('foreign', 1), mapping).status == 'unmapped'


def test_time_event_does_not_rebind_replaced_source_or_accept_missing_source():
    item = EventInterval(time_source_id='old', time_start='2026-11-11T00:00:00Z', time_end='2026-11-11T00:00:01Z')
    mapping = TraceMapping('old', 'new', np.arange(3), np.zeros(3), 'unknown')
    for record in (item, replace(item, time_source_id='')):
        with pytest.raises(ValueError, match='时间'):
            interval_indices(record, mapping, timestamps_s=np.arange(3), time_basis_confirmed=True)


def test_new_names_stay_stable_and_referenced_deletion_is_rejected():
    doc = put_record(document(), MarkerRecord(anchor=MarkerAnchor('raw:one', 3)))
    marker = doc.markers[-1]
    assert marker.name.startswith('标记 ')
    interval = EventInterval(start=marker.anchor, end=marker.anchor,
                             start_marker_id=marker.uid, start_marker_revision=marker.revision)
    doc = put_record(doc, interval)
    with pytest.raises(ValueError, match='引用'):
        remove_record(doc, marker.uid)
    doc = remove_record(doc, interval.uid)
    assert remove_record(doc, marker.uid).markers == document().markers


def test_native_selection_ignores_qc_mask_and_preserves_nonuniform_distances():
    from mygpr.domain.processing.models import ProcessingRequest
    from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor
    from mygpr.domain.markers.mapping import SEGMENT_KEY
    data = np.arange(40, dtype=np.float32).reshape(5, 8)
    request = ProcessingRequest(data, 'trace_range_select', {'ranges': [[2, 4], [3, 5]], 'mode': 'exclude'},
        {'distance_axis_m': [0., .5, 1., 1., 2., 4., 5., 8.]},
        {'bad_trace_mask': np.ones(8, dtype=np.int8)})
    result = NativeProcessingExecutor().execute(request)
    np.testing.assert_array_equal(result.data, data[:, [0, 1, 5, 6, 7]])
    np.testing.assert_array_equal(result.header_info['distance_axis_m'], [0., .5, 4., 5., 8.])
    np.testing.assert_array_equal(result.trace_metadata[SEGMENT_KEY], [0, 0, 1, 1, 1])


@pytest.mark.parametrize('ranges', [[], [[-1, 1]], [[0, 9]], [[1, 1]], [[1., 2]], [[True, 2]]])
def test_native_selection_rejects_invalid_ranges(ranges):
    from mygpr.infrastructure.processing.algorithms.extended.trace_range_select import native_trace_range_select
    with pytest.raises(ValueError):
        native_trace_range_select(np.ones((4, 8)), {'ranges': ranges})
