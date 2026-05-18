#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-tune result dialog plotting safeguards."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ui.auto_tune_result_dialog import AutoTuneResultDialog


def test_normalize_series_uses_finite_range():
    dialog = AutoTuneResultDialog.__new__(AutoTuneResultDialog)

    normalized = dialog._normalize_series(np.array([np.nan, -1.0, 1.0, np.inf]))

    assert normalized[0] == 1.0
    assert normalized[1] == 0.0
    assert normalized[2] == 1.0
    assert normalized[3] == 1.0


def test_normalize_series_fallback_when_no_finite_values():
    dialog = AutoTuneResultDialog.__new__(AutoTuneResultDialog)

    normalized = dialog._normalize_series(np.array([np.nan, np.inf, -np.inf]))

    assert np.all(normalized == 1.0)
