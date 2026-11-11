"""Field events retain identity and never invent acquisition samples."""
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from mygpr.domain.markers.models import MarkerAnchor, MarkerDocument, EventInterval
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
    item = EventInterval(uid='e', time_start='2026-11-11T00:00:02+00:00',
                         time_end='2026-11-11T00:00:05+00:00')
    origin = datetime.fromisoformat('2026-11-11T00:00:00+00:00').timestamp()
    times = origin + np.array([0, 1, 6], dtype=np.float64)
    result = interval_indices(item, TraceMapping.identity('raw', 3),
                              timestamps_s=times, time_basis_confirmed=True)
    assert result.size == 0
    with pytest.raises(ValueError, match='时间'):
        interval_indices(item, TraceMapping.identity('raw', 3), timestamps_s=times)


def test_time_reversal_and_open_interval_refuse_processing():
    item = EventInterval(uid='e', time_start='2026-11-11T00:00:02Z',
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
