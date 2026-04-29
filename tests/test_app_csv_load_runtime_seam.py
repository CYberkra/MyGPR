#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RED tests for app-level CSV load forwarding of optional sidecar kwargs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd

import app_qt


class _BoolFlag:
    def __init__(self, value: bool):
        self._value = value

    def isChecked(self) -> bool:
        return self._value


class _DummyLoaderHost:
    def __init__(self) -> None:
        self.page_advanced = SimpleNamespace(fast_preview_var=_BoolFlag(False))


def test_load_single_csv_with_progress_forwards_optional_sidecar_kwargs(monkeypatch):
    raw_data = np.arange(12, dtype=np.float32).reshape(3, 4)
    captured: dict[str, object] = {}

    def fake_detect_csv_header(path):
        return {"a_scan_length": 3, "num_traces": 4, "total_time_ns": 12.0}

    def fake_detect_skiprows(path):
        return 0

    def fake_read_csv(*args, **kwargs):
        return [pd.DataFrame(raw_data)]

    def fake_extract(raw, header_info, **kwargs):
        captured["raw_shape"] = raw.shape
        captured["kwargs"] = dict(kwargs)
        return raw.astype(np.float32), {"trace_index": np.arange(4, dtype=np.int32)}, header_info

    monkeypatch.setattr(app_qt, "detect_csv_header", fake_detect_csv_header)
    monkeypatch.setattr(app_qt, "_detect_skiprows", fake_detect_skiprows)
    monkeypatch.setattr(app_qt.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(app_qt, "extract_airborne_csv_payload", fake_extract)

    host = _DummyLoaderHost()
    trace_timestamps_s = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)

    result = app_qt.GPRGuiQt._load_single_csv_with_progress(
        cast(Any, host),
        "dummy.csv",
        trace_timestamps_s=trace_timestamps_s,
        rtk_path="rtk.csv",
        imu_path="imu.csv",
    )

    assert result["data"].shape == (3, 4)
    assert captured["raw_shape"] == (3, 4)
    forwarded = cast(dict[str, object], captured["kwargs"])
    assert np.array_equal(cast(np.ndarray, forwarded["trace_timestamps_s"]), trace_timestamps_s)
    assert forwarded["rtk_path"] == "rtk.csv"
    assert forwarded["imu_path"] == "imu.csv"


def test_load_single_csv_with_progress_preserves_legacy_call_shape(monkeypatch):
    raw_data = np.arange(6, dtype=np.float32).reshape(2, 3)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        app_qt,
        "detect_csv_header",
        lambda path: {"a_scan_length": 2, "num_traces": 3, "total_time_ns": 10.0},
    )
    monkeypatch.setattr(app_qt, "_detect_skiprows", lambda path: 0)
    monkeypatch.setattr(app_qt.pd, "read_csv", lambda *args, **kwargs: [pd.DataFrame(raw_data)])

    def fake_extract(raw, header_info, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return raw.astype(np.float32), None, header_info

    monkeypatch.setattr(app_qt, "extract_airborne_csv_payload", fake_extract)

    host = _DummyLoaderHost()
    result = app_qt.GPRGuiQt._load_single_csv_with_progress(cast(Any, host), "dummy.csv")

    assert result["data"].shape == (2, 3)
    assert cast(dict[str, object], captured["kwargs"]) == {}
