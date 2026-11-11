"""PROJ conversion stays on the GUI thread; file publication stays in the worker."""
from dataclasses import replace
from ui.desktop_backend_facade import geographic_rows


def prepare_geographic_views(views):
    from pyproj import Transformer
    from ui.widgets.proj_safe import _LOCK
    def convert(crs, xs, ys):
        if crs.upper().replace(' ', '') in ('EPSG:4326', 'WGS84', 'WGS1984'):
            return xs, ys
        with _LOCK:
            return Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(xs, ys, errcheck=True)
    return tuple(replace(view, rows=geographic_rows(view.rows, convert)) for view in views)
