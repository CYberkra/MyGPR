#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kirchhoff result dialog display safeguards."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ui.kirchhoff_result_dialog import KirchhoffResultDialog


def test_kirchhoff_color_limits_ignore_nonfinite_values():
    dialog = KirchhoffResultDialog.__new__(KirchhoffResultDialog)
    dialog.data = np.array(
        [
            [np.nan, np.inf, -np.inf],
            [-2.0, 0.0, 1.0],
            [2.0, 3.0, -1.0],
        ],
        dtype=np.float32,
    )

    vmin, vmax = dialog._compute_vmin_vmax()

    assert np.isfinite(vmin)
    assert np.isfinite(vmax)
    assert vmin < 0.0 < vmax
    assert abs(vmin) == abs(vmax)


def test_kirchhoff_color_limits_fallback_when_all_nonfinite():
    dialog = KirchhoffResultDialog.__new__(KirchhoffResultDialog)
    dialog.data = np.array([[np.nan, np.inf]], dtype=np.float32)

    assert dialog._compute_vmin_vmax() == (-1.0, 1.0)
