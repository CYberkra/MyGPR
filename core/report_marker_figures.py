"""A separate along-line strip in exported figures; no marker depth is inferred."""
import numpy as np
from mygpr.domain.markers.models import MarkerDocument, MarkerRecord
from mygpr.domain.markers.mapping import TraceMapping
from mygpr.domain.markers.ranges import interval_indices


def draw_marker_strip(ax, payload, dataset):
    if not payload:
        return
    document = MarkerDocument.from_dict(payload)
    mapping = TraceMapping.identity(document.source_id, dataset.trace_count)
    marks, unavailable = [], 0
    distance = np.asarray(dataset.distance_axis_m)
    for record in document.records:
        if record.excluded:
            continue
        if isinstance(record, MarkerRecord):
            if record.anchor and record.anchor.source_id == document.source_id and record.anchor.trace < len(distance):
                marks.append((float(distance[record.anchor.trace]), float(distance[record.anchor.trace]), record.name))
            else:
                unavailable += 1
        elif record.start and record.end:
            try:
                indices = interval_indices(record, mapping)
            except ValueError:
                unavailable += 1
                continue
            if indices.size:
                marks.append((float(distance[indices[0]]), float(distance[indices[-1]]), record.name))
            else:
                unavailable += 1
        else:
            unavailable += 1
    if not marks and not unavailable:
        return
    strip = ax.inset_axes([0, -.34, 1, .16])
    strip.set_xlim(ax.get_xlim())
    strip.set_ylim(0, 2)
    strip.set_yticks([])
    strip.set_xticks([])
    strip.set_ylabel('标记', rotation=0, labelpad=22)
    low, high = strip.get_xlim()
    occupied = set()
    counts = {}
    for a, b, name in marks:
        cell = int(30 * (a - low) / max(high - low, 1e-12))
        lane = 0 if cell not in occupied else 1
        if (cell, lane) in counts:
            counts[(cell, lane)] += 1
            continue
        occupied.add(cell)
        counts[(cell, lane)] = 1
        y = .4 + lane
        strip.plot([a, b], [y, y], marker='|', color='#2563a6', linewidth=1.5)
        strip.text(a, y + .12, name[:12], fontsize=6, clip_on=True)
    for (cell, lane), count in counts.items():
        if count > 1:
            x = low + cell * (high - low) / 30
            strip.text(x, .4 + lane, f'+{count - 1}', fontsize=7, backgroundcolor='white')
    if unavailable:
        strip.set_xlabel(f'{unavailable} 条未在图中定位；完整记录见标记清单', fontsize=7)
