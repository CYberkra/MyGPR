#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public compatibility facade for sensor synchronization.

The implementation is split into stable models, a numerical engine and
persistence helpers so callers retain the historical import path without
loading unrelated responsibilities.
"""
from core.sensor_sync_engine import synchronize_sensor_streams
from core.sensor_sync_io import save_sensor_sync_result
from core.sensor_sync_models import (
    SYNC_SCHEMA, ClockDomain, SensorCalibrationProfile, SensorSyncConfig,
    SensorSyncDiagnostics, SensorSyncResult, StreamDiagnostics, TimeAlignmentModel,
)

__all__ = [
    "SYNC_SCHEMA", "ClockDomain", "SensorCalibrationProfile", "SensorSyncConfig",
    "TimeAlignmentModel", "SensorSyncDiagnostics", "SensorSyncResult",
    "StreamDiagnostics", "save_sensor_sync_result", "synchronize_sensor_streams",
]
