#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-only helpers for optional sidecar file integration."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def load_and_integrate_optional_sidecars(
    trace_metadata: dict[str, np.ndarray],
    *,
    trace_timestamps_s: np.ndarray | None = None,
    rtk_path: str | Path | None = None,
    imu_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Parse optional RTK/IMU sidecar files and merge them into trace metadata."""
    parser_module = importlib.import_module("core.sidecar_parsers")
    metadata_module = importlib.import_module("core.trace_metadata_utils")

    if rtk_path is None and imu_path is None:
        return metadata_module.integrate_optional_sidecars(trace_metadata)

    if trace_timestamps_s is None:
        raise ValueError("trace_timestamps_s is required when sidecar files are provided")

    rtk_payload = (
        parser_module.parse_sidecar_csv(rtk_path, kind="rtk") if rtk_path is not None else None
    )
    imu_payload = (
        parser_module.parse_sidecar_csv(imu_path, kind="imu") if imu_path is not None else None
    )

    return metadata_module.integrate_optional_sidecars(
        trace_metadata,
        trace_timestamps_s=trace_timestamps_s,
        rtk_payload=rtk_payload,
        imu_payload=imu_payload,
    )
