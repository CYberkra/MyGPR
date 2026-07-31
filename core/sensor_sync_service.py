#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project service for radar/RTK/IMU synchronization and durable artifacts."""
from __future__ import annotations

import csv
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.field_project_models import local_now, validate_line_id
from core.sensor_sync import SensorSyncConfig, save_sensor_sync_result, synchronize_sensor_streams
from core.sidecar_parsers import parse_sidecar_csv


def load_trace_timestamps(path: str | Path, *, expected_count: int | None = None) -> np.ndarray:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)
    if src.suffix.lower() == ".npy":
        values = np.asarray(np.load(src, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        with src.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError("雷达道时间戳文件缺少表头。")
            aliases = ("trace_timestamp_s", "timestamp_s", "timestamp", "time_s", "gps_time")
            column = next((name for name in aliases if name in reader.fieldnames), None)
            if column is None:
                # Accept a single-column CSV exported without a known name.
                if len(reader.fieldnames) != 1:
                    raise ValueError("雷达道时间戳文件需包含 trace_timestamp_s 或 timestamp_s 列。")
                column = reader.fieldnames[0]
            values = np.asarray([float(row[column]) for row in reader if row.get(column) not in (None, "")], dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("雷达道时间戳为空。")
    if expected_count is not None and values.size != int(expected_count):
        raise ValueError(f"雷达道时间戳数量 {values.size} 与 B-scan 道数 {expected_count} 不一致。")
    return values


def infer_trace_timestamps(dataset: Any) -> np.ndarray | None:
    metadata = dict(getattr(dataset, "metadata", {}) or {})
    for key in ("trace_timestamp_s", "trace_timestamps_s", "timestamps_s"):
        value = metadata.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == int(dataset.trace_count):
            return arr
    start = metadata.get("trace_start_timestamp_s")
    interval = metadata.get("trace_interval_s")
    if start is not None and interval is not None and float(interval) > 0:
        return float(start) + np.arange(int(dataset.trace_count), dtype=np.float64) * float(interval)
    return None


def synchronize_project_line_sensors(
    store: Any,
    *,
    line_id: str,
    rtk_path: str | Path,
    trace_timestamps_path: str | Path | None = None,
    imu_path: str | Path | None = None,
    altimeter_path: str | Path | None = None,
    config: SensorSyncConfig | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    line_id = validate_line_id(line_id)
    dataset = store.load_gpr_dataset(line_id)

    def check() -> None:
        if cancel_checker is not None and cancel_checker():
            from core.job_manager import JobCancelled
            raise JobCancelled("传感器同步已取消")

    def report(done: int, total: int, message: str) -> None:
        check()
        if progress is not None:
            progress(done, total, message)

    report(0, 7, "读取雷达道时间戳")
    if trace_timestamps_path:
        trace_t = load_trace_timestamps(trace_timestamps_path, expected_count=dataset.trace_count)
    else:
        trace_t = infer_trace_timestamps(dataset)
        if trace_t is None:
            raise ValueError("当前 B-scan 不含逐道时间戳；请提供雷达道时间戳 CSV/NPY 文件。")

    report(1, 7, "解析 RTK 数据")
    rtk_payload = parse_sidecar_csv(rtk_path, kind="rtk")
    report(2, 7, "解析 IMU 数据")
    imu_payload = parse_sidecar_csv(imu_path, kind="imu") if imu_path else None
    report(3, 7, "解析测高数据")
    altimeter_payload = parse_sidecar_csv(altimeter_path, kind="altimeter") if altimeter_path else None

    cfg = config or SensorSyncConfig(project_crs=str(store.manifest.coordinate_system or ""))
    if not cfg.project_crs:
        cfg.project_crs = str(store.manifest.coordinate_system or "")
    report(4, 7, "按时间轴同步 RTK/IMU 与雷达道")
    result = synchronize_sensor_streams(
        trace_timestamps_s=trace_t,
        rtk_payload=rtk_payload,
        imu_payload=imu_payload,
        altimeter_payload=altimeter_payload,
        config=cfg,
        line_id=line_id,
        trace_distance_hint_m=np.asarray(dataset.distance_axis_m, dtype=np.float64),
    )

    report(5, 7, "写入同步成果")
    raw_dir = store.root / "raw" / line_id
    final_dir = raw_dir / "sensors"
    staging = raw_dir / f".sensor_sync_staging_{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        sources_dir = staging / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        source_paths = {
            "rtk": Path(rtk_path),
            "trace_timestamps": Path(trace_timestamps_path) if trace_timestamps_path else None,
            "imu": Path(imu_path) if imu_path else None,
            "altimeter": Path(altimeter_path) if altimeter_path else None,
        }
        copied_sources: dict[str, str] = {}
        for role, src in source_paths.items():
            if src is None:
                continue
            dest = sources_dir / src.name
            shutil.copy2(src, dest)
            copied_sources[role] = dest.relative_to(staging).as_posix()
        artifacts = save_sensor_sync_result(result, staging, basename=f"{line_id}_sensor_sync")
        manifest_path = artifacts["manifest"]
        import json
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["line_id"] = line_id
        payload["generated_at"] = local_now()
        payload["source_files"] = copied_sources
        payload["dataset"] = {
            "trace_count": int(dataset.trace_count),
            "sample_count": int(dataset.sample_count),
            "source_path": str(dataset.source_path),
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        check()
        if final_dir.exists():
            backup = raw_dir / f"sensors_previous_{local_now().replace(':','-').replace(' ','_')}"
            final_dir.replace(backup)
        staging.replace(final_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    report(6, 7, "更新项目清单")
    line = store.get_line(line_id)
    trajectory_rel = (final_dir / f"{line_id}_sensor_sync_trajectory.csv").relative_to(store.root).as_posix()
    manifest_rel = (final_dir / f"{line_id}_sensor_sync_manifest.json").relative_to(store.root).as_posix()
    metadata_rel = (final_dir / f"{line_id}_sensor_sync_trace_metadata.npz").relative_to(store.root).as_posix()
    line.trajectory_path = trajectory_rel
    line.sensor_sync_manifest_path = manifest_rel
    line.trace_metadata_path = metadata_rel
    fixed = result.diagnostics.fixed_solution_ratio
    coverage = result.diagnostics.rtk.coverage_ratio
    line.sensor_sync_status = f"已同步 · RTK覆盖{coverage:.0%} · 固定解{fixed:.0%}"
    line.rtk_status = "已定位" if coverage >= 0.99 else "部分定位"
    line.updated_at = local_now()
    store.upsert_line(line)
    store.append_log(
        f"传感器同步 {line_id}: coverage={coverage:.3f}, fixed={fixed:.3f}, warnings={len(result.diagnostics.warnings)}"
    )
    report(7, 7, "传感器同步完成")
    return {
        "line_id": line_id,
        "trajectory_path": trajectory_rel,
        "manifest_path": manifest_rel,
        "trace_metadata_path": metadata_rel,
        "diagnostics": result.diagnostics.to_dict(),
        "config": asdict(cfg),
        "summary": line.sensor_sync_status,
    }


__all__ = [
    "infer_trace_timestamps",
    "load_trace_timestamps",
    "synchronize_project_line_sensors",
]
