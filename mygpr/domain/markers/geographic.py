"""Batch coordinate conversion with explicit CRS and no gap interpolation."""
import math


def geographic_rows(rows, transform):
    groups = {}
    for row in rows:
        native = row.get('geometry')
        if native:
            points = groups.setdefault(native['crs'], {})
            for part in native['parts']:
                for point in part:
                    points[tuple(point)] = None
    converted = {}
    for crs, points in groups.items():
        coords = list(points)
        try:
            lon, lat = transform(crs, [p[0] for p in coords], [p[1] for p in coords])
            converted[crs] = {key: [float(x), float(y)] for key, x, y in zip(coords, lon, lat)
                              if math.isfinite(x) and math.isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90}
        except (ValueError, RuntimeError):
            converted[crs] = {}
    result = []
    for row in rows:
        native = row.get('geometry')
        geometry = None
        if native:
            lookup = converted[native['crs']]
            parts, current = [], []
            for part in native['parts']:
                for point in part:
                    value = lookup.get(tuple(point))
                    if value is not None:
                        current.append(value)
                    elif current:
                        parts.append(current)
                        current = []
                if current:
                    parts.append(current)
                    current = []
            if parts:
                geometry = {**native, 'crs': 'EPSG:4326', 'parts': parts, 'original_crs': native['crs']}
        result.append({**row, 'geometry': geometry})
    return tuple(result)
