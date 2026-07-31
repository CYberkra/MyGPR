from __future__ import annotations

import numpy as np

from core.crs_text import canonical_crs_text, embedded_epsg
from core.online_basemap import infer_source_crs, transform_xy


DESCRIPTIVE_CRS = "CGCS2000 / 3-degree Gauss-Kruger zone 39 (EPSG:4547)"


def test_descriptive_project_crs_extracts_embedded_epsg() -> None:
    assert embedded_epsg(DESCRIPTIVE_CRS) == "EPSG:4547"
    assert canonical_crs_text(DESCRIPTIVE_CRS) == "EPSG:4547"
    assert infer_source_crs((451000.0, 3487700.0, 451400.0, 3488100.0), DESCRIPTIVE_CRS) == "EPSG:4547"


def test_transform_xy_accepts_descriptive_project_crs() -> None:
    x, y = transform_xy(
        np.asarray([451060.0], dtype=np.float64),
        np.asarray([3487775.0], dtype=np.float64),
        DESCRIPTIVE_CRS,
        "EPSG:3857",
    )
    assert np.isfinite(x[0]) and np.isfinite(y[0])
