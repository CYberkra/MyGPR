"""Pure edits used by both the project service and desktop editor."""
from copy import deepcopy
from dataclasses import replace
from uuid import NAMESPACE_URL, uuid5
from typing import cast

from mygpr.domain.markers.models import EventInterval, MarkerAnchor, MarkerDocument, MarkerRecord


def import_markers(project_id, line_id, source_id, rows) -> MarkerDocument:
    markers = []
    for item in rows:
        original = deepcopy(dict(item))
        mid, trace = original['marker_id'], original['trace_no']
        if type(mid) is not int or mid < 1 or type(trace) is not int or trace < 1:
            raise ValueError('采集标记编号或道号无效')
        uid = str(uuid5(NAMESPACE_URL, f'{project_id}/{line_id}/{source_id}/{mid}'))
        markers.append(MarkerRecord(
            uid=uid, name=str(original.get('label') or f'标记 {mid}'),
            anchor=MarkerAnchor(source_id, trace - 1), origin='acquisition',
            original=original, created_at=original.get('marked_at', ''),
        ))
    return MarkerDocument(project_id, line_id, source_id, markers=tuple(markers))


def put_record(document: MarkerDocument, record: MarkerRecord | EventInterval) -> MarkerDocument:
    is_interval = isinstance(record, EventInterval)
    records = document.intervals if is_interval else document.markers
    other = document.markers if is_interval else document.intervals
    if any(item.uid == record.uid for item in other):
        raise ValueError('标记身份重复')
    previous = next((item for item in records if item.uid == record.uid), None)
    if previous:
        if previous.revision != record.revision:
            raise ValueError('标记已被修改，请重新加载')
        if isinstance(previous, MarkerRecord) and previous.origin == 'acquisition':
            if (not isinstance(record, MarkerRecord) or record.original != previous.original or record.anchor != previous.anchor
                    or record.origin != previous.origin or record.created_at != previous.created_at):
                raise ValueError('采集原记录不可修改')
        record = replace(record, revision=previous.revision + 1)
    updated = tuple(record if item.uid == record.uid else item for item in records)
    if previous is None:
        if not record.name:
            number = len(document.records) + 1
            existing = {item.name for item in document.records}
            while f'标记 {number}' in existing:
                number += 1
            record = replace(record, name=f'标记 {number}')
        updated += (record,)
    if is_interval:
        _validate_references(document, record)
        return replace(document, intervals=cast(tuple[EventInterval, ...], updated))
    intervals = tuple(
        replace(item, review_state='needs_review', revision=item.revision + 1)
        if previous and record.uid in (item.start_marker_id, item.end_marker_id) else item
        for item in document.intervals
    )
    return replace(document, markers=cast(tuple[MarkerRecord, ...], updated), intervals=intervals)


def _validate_references(document, interval):
    markers = {item.uid: item for item in document.markers}
    for side in ('start', 'end'):
        uid = getattr(interval, f'{side}_marker_id')
        if not uid:
            continue
        mark = markers.get(uid)
        if mark is None or mark.excluded:
            raise ValueError('区间引用的标记不存在或已排除')
        if (getattr(interval, side) != mark.anchor
                or getattr(interval, f'{side}_marker_revision') != mark.revision):
            raise ValueError('区间边界版本已变化，请重新确认')


def remove_record(document: MarkerDocument, uid: str) -> MarkerDocument:
    record = next((item for item in document.records if item.uid == uid), None)
    if record is None:
        raise ValueError('标记不存在')
    if isinstance(record, MarkerRecord) and record.origin == 'acquisition':
        raise ValueError('采集标记只能排除使用，不能删除原记录')
    dependants = [item.uid for item in document.intervals
                  if uid in (item.start_marker_id, item.end_marker_id)]
    if record.links or dependants:
        raise ValueError('标记仍被区间或解释对象引用，请先解除关联')
    return replace(document, markers=tuple(item for item in document.markers if item.uid != uid),
                   intervals=tuple(item for item in document.intervals if item.uid != uid))


def validate_transition(current: MarkerDocument, proposed: MarkerDocument) -> None:
    """Validate storage callers as well as GUI edits, preserving old boundary snapshots."""
    previous = {item.uid: item for item in current.records}
    following = {item.uid: item for item in proposed.records}
    for uid, old in previous.items():
        if uid not in following:
            remove_record(current, uid)
    for item in proposed.records:
        old = previous.get(item.uid)
        if old is None:
            if item.revision != 1 or isinstance(item, MarkerRecord) and item.origin != 'project':
                raise ValueError('新增项目记录来源或版本无效')
        elif type(old) is not type(item) or item != old and item.revision != old.revision + 1:
            raise ValueError('记录版本必须逐次增加')
        if isinstance(item, EventInterval):
            if any(uid and uid not in following for uid in (item.start_marker_id, item.end_marker_id)):
                raise ValueError('区间引用的标记不存在')
            if item.review_state != 'needs_review':
                _validate_references(proposed, item)
