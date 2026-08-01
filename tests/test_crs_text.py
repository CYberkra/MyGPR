from __future__ import annotations

import numpy as np

from core.crs_text import canonical_crs_text, embedded_epsg


DESCRIPTIVE_CRS = "CGCS2000 / 3-degree Gauss-Kruger zone 39 (EPSG:4547)"


def test_descriptive_project_crs_extracts_embedded_epsg() -> None:
    assert embedded_epsg(DESCRIPTIVE_CRS) == "EPSG:4547"
    assert canonical_crs_text(DESCRIPTIVE_CRS) == "EPSG:4547"
