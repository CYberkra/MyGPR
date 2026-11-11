"""Field-time and result-coordinate detail strings, without inventing depth."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np


def measurement_details(view, record, projection):
    lines = []
    anchors = (getattr(record, 'anchor', None), getattr(record, 'start', None), getattr(record, 'end', None))
    for side, anchor in zip(('位置', '起始', '结束'), anchors):
        if anchor:
            origin = '原始道号' if anchor.source_id == view.document.source_id else '绑定来源道号'
            lines.append(f'{side}{origin}：{anchor.trace + 1}；来源 {anchor.source_id}')
    zone = ZoneInfo(view.timezone)
    marked = getattr(record, 'created_at', '')
    if marked:
        label = '采集端标记时间：' if getattr(record, 'origin', '') == 'acquisition' else '项目记录时间：'
        lines.append(label + datetime.fromisoformat(marked).astimezone(zone).isoformat(timespec='milliseconds'))
    if getattr(record, 'start_marker_id', '') and getattr(record, 'end_marker_id', ''):
        marks = {item.uid: item for item in view.document.markers}
        a, b = marks.get(record.start_marker_id), marks.get(record.end_marker_id)
        if a and b and a.created_at and b.created_at:
            gap = (datetime.fromisoformat(b.created_at) - datetime.fromisoformat(a.created_at)).total_seconds()
            lines.append(f'标记间隔：{gap:g} s')
    if projection:
        ends = [int(p) for p in projection.positions if float(p).is_integer()]
        ends += [value for a, b in projection.spans for value in (a, b - 1)]
        distance = view.trace_metadata.get('trace_distance_m', view.header.get('distance_axis_m'))
        reliable_distance = bool(view.header.get('distance_axis_status') not in (None, 'unknown', 'trace_index'))
        for index in dict.fromkeys(ends):
            if view.time_basis_confirmed:
                time = datetime.fromtimestamp(float(view.trace_metadata['trace_timestamp_s'][index]), timezone.utc)
                lines.append(f'当前道 {index + 1} 采集时刻：{time.astimezone(zone).isoformat(timespec="milliseconds")}')
            if reliable_distance and distance is not None and index < len(distance) and np.isfinite(distance[index]):
                lines.append(f'当前道 {index + 1} 距离：{distance[index]:g} m')
    available = {uid for uid, _ in view.interpretation_links}
    invalid = set(record.links) - available
    if invalid:
        lines.append('失效解释引用：' + ', '.join(sorted(invalid)))
    for entry in view.processing_history:
        selected = entry['selection']
        snapshot = next((item for item in selected['intervals'] if item['uid'] == record.uid), None)
        if snapshot:
            action = '保留' if selected['mode'] == 'keep' else '剔除'
            lines.append(f"已执行：{action} · 区间版本 {snapshot['revision']} · 成果 {entry['artifact_id']}")
            ranges = '、'.join(f'{a + 1}–{b}' for a, b in selected['ranges'])
            lines.append(f"输入：{selected['input_artifact_id'] or '原始数据'} · 道范围 {ranges}")
    return lines
