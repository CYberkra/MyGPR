#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kirchhoff result dialog metadata fallback tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ui.kirchhoff_result_dialog import KirchhoffResultDialog


def test_kirchhoff_result_extent_tolerates_invalid_metadata():
    dialog = SimpleNamespace(
        data=np.ones((10, 5), dtype=np.float32),
        header_info={
            "track_length_m": "bad",
            "trace_interval_m": "bad",
            "total_time_ns": "bad",
        },
    )

    extent = KirchhoffResultDialog._compute_extent(dialog)

    assert extent == [0, 5.0, 10.0, 0]


def test_kirchhoff_result_info_text_tolerates_invalid_metadata():
    dialog = SimpleNamespace(
        data=np.ones((10, 5), dtype=np.float32),
        header_info={
            "track_length_m": "bad",
            "is_elevation": True,
            "elevation_top_m": "bad",
            "elevation_bottom_m": "bad",
        },
    )

    text = KirchhoffResultDialog._build_info_text(dialog)

    assert text == "数据: 10×5"
