#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless import -> sensor sync -> motion compensation smoke workflow."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from mygpr.application.jobs.models import JobStatus
from mygpr.interfaces.backend import MyGPRBackend


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="mygpr_acquisition_smoke_") as temporary:
        root = Path(temporary)
        source = root / "line.npy"
        samples, traces = 64, 8
        rng = np.random.default_rng(17)
        np.save(source, rng.normal(size=(samples, traces)).astype(np.float32))
        timestamps = np.arange(traces, dtype=np.float64) + 1_000.0
        trace_times = _write_csv(
            root / "trace_times.csv",
            [{"trace_timestamp_s": value} for value in timestamps],
        )
        rtk = _write_csv(
            root / "rtk.csv",
            [
                {
                    "timestamp_s": value,
                    "longitude": 104.0 + index * 1e-5,
                    "latitude": 30.0,
                    "local_x_m": float(index),
                    "local_y_m": 0.0,
                    "local_z_m": 100.0,
                    "flight_height_m": 2.0 + 0.05 * np.sin(index),
                    "rtk_fix_type": 4,
                }
                for index, value in enumerate(timestamps)
            ],
        )
        imu = _write_csv(
            root / "imu.csv",
            [
                {
                    "timestamp_s": value,
                    "roll_deg": 0.0,
                    "pitch_deg": 0.0,
                    "yaw_deg": 90.0,
                }
                for value in timestamps
            ],
        )
        backend = MyGPRBackend.create_default(max_workers=1)
        try:
            project = backend.projects.create_project(
                root / "project",
                name="Acquisition smoke",
                coordinate_system="EPSG:32648",
            )
            imported = backend.acquisition.import_line(
                project.project_id,
                source,
                line_id="L01",
            )
            synced = backend.acquisition.synchronize_project_line(
                project.project_id,
                "L01",
                rtk_path=rtk,
                trace_timestamps_path=trace_times,
                imu_path=imu,
            )
            job_id = backend.submit_project_pipeline(
                project.project_id,
                "L01",
                backend.acquisition.motion_pipeline(),
                result_name="motion-v2-smoke",
            )
            snapshot = backend.jobs.wait(job_id, timeout=30)
            if snapshot.status is not JobStatus.COMPLETED:
                raise RuntimeError(snapshot.error_message or "motion processing failed")
            return {
                "backend_api": backend.api_version,
                "project_id": project.project_id,
                "line_id": imported.line_id,
                "shape": list(imported.shape),
                "rtk_coverage_ratio": synced.diagnostics["rtk"]["coverage_ratio"],
                "motion_artifact_id": snapshot.result.artifact_id,
                "qt_loaded": any(name.startswith("PyQt") for name in __import__("sys").modules),
            }
        finally:
            backend.shutdown()


def main() -> int:
    print(json.dumps(run(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
