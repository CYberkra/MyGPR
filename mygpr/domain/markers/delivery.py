"""Flat readable tables paired with lossless marker document snapshots."""
import json

from mygpr.domain.markers.models import record_from_dict
from mygpr.domain.markers.operations import interval_boundary_issue


def _position(record, kind):
    if kind == 'markers':
        anchor = record.get('anchor')
        return f"{anchor['trace'] + 1} 道" if anchor else ''
    start, end = record.get('start'), record.get('end')
    if start and end:
        first = start['trace'] + 1 + int(not record.get('include_start', True))
        last = end['trace'] + 1 - int(not record.get('include_end', True))
        return f'{first}–{last} 道' if first <= last else '区间内无道'
    if start or end:
        return f"{start['trace'] + 1 if start else '…'}–{end['trace'] + 1 if end else '…'} 道"
    return ''


def delivery_rows(documents):
    rows = []
    for document in documents:
        markers = {record['uid']: record_from_dict(record) for record in document.get('markers', ())}
        for kind in ('markers', 'intervals'):
            for record in document.get(kind, ()):
                anchor = record.get('anchor') or record.get('start') or record.get('end') or {}
                source = anchor.get('source_id') or record.get('time_source_id', '')
                state = '已排除' if record.get('excluded') else ''
                if kind == 'intervals' and not state:
                    interval = record_from_dict(record, interval=True)
                    state = interval_boundary_issue(interval, markers)
                    if not state and not interval.closed:
                        state = '未闭合'
                rows.append({'测线': document['line_id'], 'ID': record['uid'], '名称': record['name'],
                             '类型': '点' if kind == 'markers' else '区间', '用途': record['purpose'],
                             '版本': record['revision'], '信息来源': record['evidence'],
                             '道号范围': _position(record, kind),
                             '数据版本': '原始数据' if source == document['source_id'] else source,
                             '开始采集时间（UTC）': record.get('time_start', ''),
                             '结束采集时间（UTC）': record.get('time_end', ''),
                             '状态': state,
                             '备注': record['note'], '完整记录': json.dumps(record, ensure_ascii=False)})
    return rows
