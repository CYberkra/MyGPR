"""Flat readable tables paired with lossless marker document snapshots."""
import json


def delivery_rows(documents):
    rows = []
    for document in documents:
        for kind in ('markers', 'intervals'):
            for record in document.get(kind, ()):
                rows.append({'测线': document['line_id'], 'ID': record['uid'], '名称': record['name'],
                             '类型': '点' if kind == 'markers' else '区间', '用途': record['purpose'],
                             '版本': record['revision'], '信息来源': record['evidence'],
                             '备注': record['note'], '完整记录': json.dumps(record, ensure_ascii=False)})
    return rows
