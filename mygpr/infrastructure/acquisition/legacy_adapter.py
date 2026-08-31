#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infrastructure adapters over verified legacy import and sensor kernels."""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np

from core.field_import_preview import build_import_preflight
from core.gpr_data_model import load_gpr_dataset_for_import
from core.sensor_sync import SensorSyncConfig, synchronize_sensor_streams
from core.sidecar_parsers import parse_sidecar_csv
from mygpr.application.acquisition.ports import (
    AcquisitionReaderPort,
    SensorSidecarParserPort,
    SensorSynchronizerPort,
)
from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.acquisition.models import (
    AcquisitionDataset,
    ImportPreflight,
    SensorKind,
    SensorStream,
    SensorSyncSettings,
    SynchronizedSensorData,
)


class LegacyAcquisitionReader(AcquisitionReaderPort):
    """Cancellable, chunked reader adapter without exposing legacy datasets."""

    def preflight(
        self,
        source: Path,
        *,
        line_id: str,
        dielectric_constant: float,
        context: ExecutionContext,
    ) -> ImportPreflight:
        result = build_import_preflight(
            source,
            line_id=line_id,
            dielectric_constant=dielectric_constant,
            cancel_requested=context.is_cancelled,
            progress_callback=lambda stage, done, total: context.report_progress(done, total, stage),
        )
        return ImportPreflight(
            path=result.path,
            exists=result.exists,
            is_file=result.is_file,
            extension=result.extension,
            format_name=result.format_name,
            support=result.support,
            can_import=result.can_import,
            message=result.message,
            suggestions=tuple(result.suggestions),
            sample_count=result.sample_count,
            trace_count=result.trace_count,
            length_m=result.length_m,
            time_window_ns=result.time_window_ns,
            dielectric_constant=result.dielectric_constant,
            data_min=result.data_min,
            data_max=result.data_max,
            source_kind=result.source_kind,
            has_trajectory=result.has_trajectory,
            column_summary=result.column_summary,
        )

    @contextmanager
    def open_dataset(
        self,
        source: Path,
        *,
        line_id: str,
        length_m: float,
        dielectric_constant: float,
        context: ExecutionContext,
    ) -> Iterator[AcquisitionDataset]:
        context.raise_if_cancelled()
        with tempfile.TemporaryDirectory(prefix="mygpr_acquisition_") as temporary:
            dataset = load_gpr_dataset_for_import(
                source,
                line_id=line_id,
                staging_dir=temporary,
                length_m=length_m if length_m > 0 else None,
                dielectric_constant=dielectric_constant,
                cancel_requested=context.is_cancelled,
                progress_callback=lambda stage, done, total: context.report_progress(done, total, stage),
            )
            trace_metadata = {
                str(key): np.array(value, copy=True)
                for key, value in dataset.metadata.items()
                if isinstance(value, np.ndarray)
                and value.ndim == 1
                and value.size == int(dataset.trace_count)
            }
            yield AcquisitionDataset(
                line_id=line_id,
                data=dataset.matrix,
                length_m=dataset.length_m,
                time_window_ns=dataset.time_window_ns,
                dielectric_constant=dataset.dielectric_constant,
                format_name=dataset.format_name,
                source_path=dataset.source_path or str(source),
                metadata=dict(dataset.metadata),
                trace_metadata=trace_metadata,
            )


class LegacySensorSidecarParser(SensorSidecarParserPort):
    def parse(self, source: Path, *, kind: SensorKind) -> SensorStream:
        payload = parse_sidecar_csv(source, kind=kind.value)
        fields = {
            str(key): np.asarray(value)
            for key, value in payload.items()
            if key != "source_kind"
        }
        return SensorStream(kind=kind, fields=fields, source_path=str(source))


class LegacySensorSynchronizer(SensorSynchronizerPort):
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
    ) -> SynchronizedSensorData:
        context.raise_if_cancelled()
        context.report_progress(0, 1, "Synchronizing radar, RTK, IMU and altimeter clocks")
        result = synchronize_sensor_streams(
            trace_timestamps_s=np.asarray(trace_timestamps_s, dtype=np.float64),
            rtk_payload=_payload(rtk),
            imu_payload=_payload(imu),
            altimeter_payload=_payload(altimeter),
            config=SensorSyncConfig(**settings.to_mapping()),
            line_id=str(line_id),
            trace_distance_hint_m=(
                None
                if trace_distance_hint_m is None
                else np.asarray(trace_distance_hint_m, dtype=np.float64)
            ),
        )
        context.raise_if_cancelled()
        context.report_progress(1, 1, "Sensor synchronization completed")
        return SynchronizedSensorData(
            trace_metadata=result.trace_metadata,
            diagnostics=result.diagnostics.to_dict(),
            config=asdict(result.config),
            trajectory=tuple(asdict(point) for point in result.trajectory.points),
        )


def _payload(stream: SensorStream | None) -> dict[str, object] | None:
    if stream is None:
        return None
    return {
        "source_kind": stream.kind.value,
        **{key: np.array(value, copy=True) for key, value in stream.fields.items()},
    }


__all__ = ["LegacyAcquisitionReader", "LegacySensorSidecarParser", "LegacySensorSynchronizer"]
