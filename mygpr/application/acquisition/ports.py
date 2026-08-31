#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ports for source import, sidecar parsing and sensor synchronization."""
from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.acquisition.models import (
    AcquisitionDataset,
    ImportPreflight,
    SensorKind,
    SensorStream,
    SensorSyncSettings,
    SynchronizedSensorData,
)


class AcquisitionReaderPort(Protocol):
    def preflight(
        self,
        source: Path,
        *,
        line_id: str,
        dielectric_constant: float,
        context: ExecutionContext,
    ) -> ImportPreflight: ...

    def open_dataset(
        self,
        source: Path,
        *,
        line_id: str,
        length_m: float,
        dielectric_constant: float,
        context: ExecutionContext,
    ) -> AbstractContextManager[AcquisitionDataset]: ...


class SensorSidecarParserPort(Protocol):
    def parse(self, source: Path, *, kind: SensorKind) -> SensorStream: ...


class SensorSynchronizerPort(Protocol):
    def synchronize(
        self,
        *,
        trace_timestamps_s: np.ndarray,
        rtk: SensorStream,
        imu: SensorStream | None,
        altimeter: SensorStream | None,
        settings: SensorSyncSettings,
        line_id: str,
        trace_distance_hint_m: np.ndarray | None,
        context: ExecutionContext,
    ) -> SynchronizedSensorData: ...


__all__ = ["AcquisitionReaderPort", "SensorSidecarParserPort", "SensorSynchronizerPort"]
