"""Immutable JSON/CSV/GeoJSON snapshots, atomically published."""
from __future__ import annotations

import csv
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path

from mygpr.infrastructure.persistence.durable_write import default_durable_write


def snapshot(views):
    return {'format': 'mygpr-marker-export-1', 'documents': [view.document.to_dict() for view in views],
            'projections': [{'line_id': view.document.line_id, 'target_id': view.target_id,
                'records': [{'uid': row['uid'], 'projection': asdict(row['projection']),
                             'status': row['status'], 'geometry': row['geometry']} for row in view.rows]}
                for view in views]}


def export_views(views, path, format_name):
    if format_name == 'json':
        content = json.dumps(snapshot(views), ensure_ascii=False, indent=2, allow_nan=False)
        count = sum(len(view.rows) for view in views)
        omitted = 0
    elif format_name == 'csv':
        stream = StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['项目', '测线', '当前来源', 'ID', '名称', '用途', '位置或范围', '状态', '版本', '完整记录JSON', '投影JSON'])
        for view in views:
            for row in view.rows:
                values = [view.document.project_id, view.document.line_id, view.target_id, row['uid'],
                          row['name'], row['purpose'], row['position'], row['status'], row['record'].revision,
                          json.dumps(asdict(row['record']), ensure_ascii=False, allow_nan=False),
                          json.dumps(asdict(row['projection']), ensure_ascii=False, allow_nan=False)]
                writer.writerow([_cell(value) for value in values])
        content = '\ufeff' + stream.getvalue()
        count = sum(len(view.rows) for view in views)
        omitted = 0
    elif format_name == 'geojson':
        from mygpr.domain.markers.geographic import geographic_rows
        features, omitted = [], 0
        for view in views:
            for row in geographic_rows(view.rows, _to_wgs84):
                feature = geojson_feature(row, view.document.line_id)
                if feature:
                    features.append(feature)
                else:
                    omitted += 1
        content = json.dumps({'type': 'FeatureCollection', 'features': features,
                              'unlocated_records': omitted}, ensure_ascii=False, indent=2, allow_nan=False)
        count = len(features)
    else:
        raise ValueError('仅支持 CSV、JSON 或 GeoJSON')
    default_durable_write().write_text(Path(path), content)
    return f'已导出 {count} 条记录；无法空间定位 {omitted} 条' if format_name == 'geojson' else f'已导出 {count} 条记录'


def _cell(value):
    text = str(value)
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else text


def _to_wgs84(crs, xs, ys):
    if crs.upper().replace(' ', '') in ('EPSG:4326', 'WGS84', 'WGS1984'):
        return xs, ys
    import threading
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('投影坐标须在主线程转换后导出')
    from pyproj import Transformer
    return Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(xs, ys, errcheck=True)


def geojson_feature(row, line_id):
    native = row.get('geometry')
    if not native:
        return None
    # Geographic output is RFC 7946 longitude/latitude. Never relabel projected metres.
    if native['crs'].upper().replace(' ', '') not in ('EPSG:4326', 'WGS84', 'WGS1984'):
        return None
    parts = native['parts']
    points = [part[0] for part in parts if len(part) == 1]
    lines = [part for part in parts if len(part) > 1]
    geometries = []
    if points:
        geometries.append({'type': 'MultiPoint', 'coordinates': points})
    if lines:
        geometries.append({'type': 'MultiLineString', 'coordinates': lines})
    geometry = geometries[0] if len(geometries) == 1 else {'type': 'GeometryCollection', 'geometries': geometries}
    return {'type': 'Feature', 'id': row['uid'], 'geometry': geometry,
            'properties': {**asdict(row['record']), 'line_id': line_id,
                           'meaning': native['meaning'], 'coordinate_source': native['source']}}
