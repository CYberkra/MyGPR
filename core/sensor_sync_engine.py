#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerical synchronization engine for radar, RTK, IMU and altimeter streams."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from core.sensor_sync_models import (
    ClockDomain, SensorCalibrationProfile, SensorSyncConfig, SensorSyncDiagnostics,
    SensorSyncResult, StreamDiagnostics, TimeAlignmentModel,
)
from core.trace_metadata_utils import derive_local_xy_m
from core.trajectory_model import TrajectoryModel, TrajectoryPoint

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None

def _as_1d(payload: dict[str, Any], key: str, dtype=float, *, required: bool = False) -> np.ndarray | None:
    if key not in payload:
        if required:
            raise ValueError(f"传感器数据缺少字段：{key}")
        return None
    array = np.asarray(payload[key], dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"传感器字段必须是一维数组：{key}")
    return array

def _normalize_stream(
    payload: dict[str, Any] | None,
    clock: ClockDomain,
    alignment: TimeAlignmentModel,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    if payload is None:
        return np.array([], dtype=np.float64), {}, {"raw_nonmonotonic": 0, "raw_duplicate": 0}
    timestamps = _as_1d(payload, "timestamp_s", np.float64, required=True)
    assert timestamps is not None
    raw_seconds = timestamps * clock.factor_to_seconds()
    raw_diffs = np.diff(raw_seconds)
    stats = {
        "raw_nonmonotonic": int(np.count_nonzero(raw_diffs < 0)),
        "raw_duplicate": int(np.count_nonzero(raw_diffs == 0)),
    }
    normalized = alignment.apply(raw_seconds)
    if not np.isfinite(normalized).all():
        valid = np.isfinite(normalized)
    else:
        valid = np.ones(normalized.size, dtype=bool)
    order = np.argsort(normalized[valid], kind="stable")
    sorted_timestamps = normalized[valid][order]
    fields: dict[str, np.ndarray] = {}
    for key, values in payload.items():
        if key in {"timestamp_s", "source_kind"}:
            continue
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.size != timestamps.size:
            raise ValueError(f"传感器字段长度不一致：{key}")
        fields[key] = arr[valid][order]
    return sorted_timestamps, fields, stats

def _nearest_residual(source_t: np.ndarray, trace_t: np.ndarray) -> np.ndarray:
    if source_t.size == 0:
        return np.full(trace_t.size, np.inf, dtype=np.float64)
    insert = np.searchsorted(source_t, trace_t, side="left")
    right = np.clip(insert, 0, source_t.size - 1)
    left = np.clip(insert - 1, 0, source_t.size - 1)
    return np.minimum(np.abs(trace_t - source_t[left]), np.abs(source_t[right] - trace_t))

def _stream_diagnostics(
    name: str,
    source_t: np.ndarray,
    trace_t: np.ndarray,
    *,
    gap_warning_s: float,
    accepted: np.ndarray | None = None,
    raw_stats: dict[str, int] | None = None,
) -> StreamDiagnostics:
    if source_t.size == 0:
        return StreamDiagnostics(name=name, source_count=0, coverage_ratio=0.0, accepted_ratio=0.0, invalid_trace_count=int(trace_t.size))
    diffs = np.diff(source_t)
    residual = _nearest_residual(source_t, trace_t)
    covered = (trace_t >= source_t[0]) & (trace_t <= source_t[-1])
    accepted_mask = covered if accepted is None else np.asarray(accepted, dtype=bool)
    return StreamDiagnostics(
        name=name,
        source_count=int(source_t.size),
        coverage_ratio=float(np.mean(covered)) if trace_t.size else 0.0,
        accepted_ratio=float(np.mean(accepted_mask)) if trace_t.size else 0.0,
        invalid_trace_count=int(np.count_nonzero(~accepted_mask)),
        duplicate_timestamp_count=int(np.count_nonzero(diffs == 0)),
        nonmonotonic_step_count=int(np.count_nonzero(diffs < 0)),
        raw_nonmonotonic_step_count=int((raw_stats or {}).get("raw_nonmonotonic", 0)),
        raw_duplicate_timestamp_count=int((raw_stats or {}).get("raw_duplicate", 0)),
        gap_count=int(np.count_nonzero(diffs > float(gap_warning_s))),
        median_nearest_residual_s=float(np.median(residual)) if residual.size else 0.0,
        max_nearest_residual_s=float(np.max(residual)) if residual.size else 0.0,
    )

def _coverage(source_t: np.ndarray, trace_t: np.ndarray) -> np.ndarray:
    if source_t.size == 0:
        return np.zeros(trace_t.size, dtype=bool)
    return (trace_t >= source_t[0]) & (trace_t <= source_t[-1])

def _interp_numeric(source_t: np.ndarray, values: np.ndarray | None, trace_t: np.ndarray, valid: np.ndarray, *, circular: bool = False) -> np.ndarray:
    if values is None or source_t.size == 0:
        return np.full(trace_t.size, np.nan, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if circular:
        arr = np.unwrap(np.radians(arr))
        out = np.degrees(np.interp(trace_t, source_t, arr)) % 360.0
    else:
        out = np.interp(trace_t, source_t, arr)
    out = np.asarray(out, dtype=np.float64)
    out[~valid] = np.nan
    return out

def _nearest_values(source_t: np.ndarray, values: np.ndarray | None, trace_t: np.ndarray, valid: np.ndarray, default: Any) -> np.ndarray:
    if values is None or source_t.size == 0:
        return np.full(trace_t.size, default)
    insert = np.searchsorted(source_t, trace_t, side="left")
    right = np.clip(insert, 0, source_t.size - 1)
    left = np.clip(insert - 1, 0, source_t.size - 1)
    choose_left = np.abs(trace_t - source_t[left]) <= np.abs(source_t[right] - trace_t)
    idx = np.where(choose_left, left, right)
    out = np.asarray(values)[idx].copy()
    if np.issubdtype(out.dtype, np.number):
        out = out.astype(np.float64)
        out[~valid] = np.nan
    else:
        out = out.astype("<U48")
        out[~valid] = str(default)
    return out

def _project_lonlat(lon: np.ndarray, lat: np.ndarray, project_crs: str) -> tuple[np.ndarray, np.ndarray, str]:
    valid = np.isfinite(lon) & np.isfinite(lat)
    x: np.ndarray = np.full(lon.size, np.nan, dtype=np.float64)
    y: np.ndarray = np.full(lat.size, np.nan, dtype=np.float64)
    if valid.any() and project_crs:
        if Transformer is None or CRS is None:
            raise RuntimeError("配置了工程 CRS，但 pyproj 不可用。")
        try:
            target = CRS.from_user_input(project_crs)
        except Exception as exc:
            raise ValueError(f"无法解析工程坐标系：{project_crs}") from exc
        transformer = Transformer.from_crs("EPSG:4326", target, always_xy=True)
        xx, yy = transformer.transform(lon[valid], lat[valid])
        x[valid] = xx
        y[valid] = yy
        return x, y, target.to_wkt()
    if valid.any():
        lx, ly = derive_local_xy_m(lon[valid], lat[valid])
        x[valid] = lx
        y[valid] = ly
    return x, y, "LOCAL_TANGENT_PLANE"

def _rotation_body_to_world(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx

def _apply_lever_arm(x: np.ndarray, y: np.ndarray, z: np.ndarray, roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray, lever: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.allclose(lever, 0.0):
        return x, y, z
    out_x, out_y, out_z = x.copy(), y.copy(), z.copy()
    for idx in range(x.size):
        if not np.all(np.isfinite([x[idx], y[idx], z[idx], roll[idx], pitch[idx], yaw[idx]])):
            continue
        correction = _rotation_body_to_world(float(roll[idx]), float(pitch[idx]), float(yaw[idx])) @ lever
        out_x[idx] += correction[0]
        out_y[idx] += correction[1]
        out_z[idx] += correction[2]
    return out_x, out_y, out_z

def _fix_label(value: Any) -> str:
    # Nearest-neighbour selection promotes numeric fix codes to float so values
    # may arrive as 4.0/5.0.  Normalize numeric and textual vendor contracts.
    try:
        if np.isfinite(float(value)):
            numeric = int(round(float(value)))
            if numeric == 4:
                return "固定解"
            if numeric == 5:
                return "浮动解"
            if numeric == 1:
                return "单点解"
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"fixed", "fix", "rtk_fixed", "fixed solution", "固定", "固定解"} or "fixed" in text or "固定" in text:
        return "固定解"
    if text in {"float", "rtk_float", "浮点", "浮动解"} or "float" in text or "浮" in text:
        return "浮动解"
    if text in {"single", "autonomous", "单点"} or "single" in text or "单点" in text:
        return "单点解"
    return "未知"

def synchronize_sensor_streams(
    *,
    trace_timestamps_s: np.ndarray,
    rtk_payload: dict[str, Any],
    imu_payload: dict[str, Any] | None = None,
    altimeter_payload: dict[str, Any] | None = None,
    config: SensorSyncConfig | None = None,
    line_id: str = "",
    trace_distance_hint_m: np.ndarray | None = None,
) -> SensorSyncResult:
    cfg = config or SensorSyncConfig()
    radar_clock = ClockDomain.from_value(cfg.radar_clock, default_name="radar")
    trace_t = np.asarray(trace_timestamps_s, dtype=np.float64) * radar_clock.factor_to_seconds() + float(cfg.radar_trigger_delay_s)
    if trace_t.ndim != 1 or trace_t.size == 0:
        raise ValueError("雷达道时间戳必须是一维非空数组。")
    if not np.isfinite(trace_t).all():
        raise ValueError("雷达道时间戳包含非有限值。")

    rtk_t, rtk, rtk_raw_stats = _normalize_stream(
        rtk_payload, ClockDomain.from_value(cfg.rtk_clock, default_name="rtk"),
        TimeAlignmentModel.from_value(cfg.rtk_alignment, offset_s=cfg.rtk_time_offset_s),
    )
    imu_t, imu, imu_raw_stats = _normalize_stream(
        imu_payload, ClockDomain.from_value(cfg.imu_clock, default_name="imu"),
        TimeAlignmentModel.from_value(cfg.imu_alignment, offset_s=cfg.imu_time_offset_s),
    )
    alt_t, alt, alt_raw_stats = _normalize_stream(
        altimeter_payload, ClockDomain.from_value(cfg.altimeter_clock, default_name="altimeter"),
        TimeAlignmentModel.from_value(cfg.altimeter_alignment, offset_s=cfg.altimeter_time_offset_s),
    )
    if rtk_t.size < 2:
        raise ValueError("RTK 数据至少需要两个有效时间点。")

    rtk_cov = _coverage(rtk_t, trace_t)
    imu_cov = _coverage(imu_t, trace_t) if imu_t.size else np.zeros(trace_t.size, dtype=bool)
    alt_cov = _coverage(alt_t, trace_t) if alt_t.size else np.zeros(trace_t.size, dtype=bool)
    rtk_residual = _nearest_residual(rtk_t, trace_t)
    imu_residual = _nearest_residual(imu_t, trace_t) if imu_t.size else np.full(trace_t.size, np.inf)
    alt_residual = _nearest_residual(alt_t, trace_t) if alt_t.size else np.full(trace_t.size, np.inf)
    residual_limit = max(float(cfg.maximum_nearest_residual_s), 0.0)
    if cfg.allow_extrapolation:
        rtk_valid = np.ones(trace_t.size, dtype=bool)
        imu_valid = np.ones(trace_t.size, dtype=bool) if imu_t.size else np.zeros(trace_t.size, dtype=bool)
        alt_valid = np.ones(trace_t.size, dtype=bool) if alt_t.size else np.zeros(trace_t.size, dtype=bool)
    else:
        rtk_valid = rtk_cov & (rtk_residual <= residual_limit)
        imu_valid = imu_cov & (imu_residual <= residual_limit) if imu_t.size else np.zeros(trace_t.size, dtype=bool)
        alt_valid = alt_cov & (alt_residual <= residual_limit) if alt_t.size else np.zeros(trace_t.size, dtype=bool)

    lon = _interp_numeric(rtk_t, rtk.get("longitude"), trace_t, rtk_valid)
    lat = _interp_numeric(rtk_t, rtk.get("latitude"), trace_t, rtk_valid)
    local_x = _interp_numeric(rtk_t, rtk.get("local_x_m"), trace_t, rtk_valid)
    local_y = _interp_numeric(rtk_t, rtk.get("local_y_m"), trace_t, rtk_valid)
    coord_name = cfg.project_crs or ""
    if not np.isfinite(local_x).any() or not np.isfinite(local_y).any():
        local_x, local_y, coord_name = _project_lonlat(lon, lat, cfg.project_crs)
    local_z = _interp_numeric(rtk_t, rtk.get("local_z_m"), trace_t, rtk_valid)
    ground_z = _interp_numeric(rtk_t, rtk.get("ground_elevation_m"), trace_t, rtk_valid)
    flight_height = _interp_numeric(rtk_t, rtk.get("flight_height_m"), trace_t, rtk_valid)
    if not np.isfinite(local_z).any():
        local_z = ground_z + flight_height
    height_agl = _interp_numeric(alt_t, alt.get("height_agl_m"), trace_t, alt_valid) if alt_t.size else flight_height.copy()
    height_agl = height_agl + float(cfg.antenna_height_offset_m)

    roll = _interp_numeric(imu_t, imu.get("roll_deg"), trace_t, imu_valid) if imu_t.size else np.full(trace_t.size, np.nan)
    pitch = _interp_numeric(imu_t, imu.get("pitch_deg"), trace_t, imu_valid) if imu_t.size else np.full(trace_t.size, np.nan)
    yaw = _interp_numeric(imu_t, imu.get("yaw_deg"), trace_t, imu_valid, circular=True) if imu_t.size else np.full(trace_t.size, np.nan)

    lever: np.ndarray = np.asarray([cfg.lever_arm_x_m, cfg.lever_arm_y_m, cfg.lever_arm_z_m], dtype=np.float64)
    lever_arm_applied = False
    if not np.allclose(lever, 0.0):
        attitude_valid = np.isfinite(roll) & np.isfinite(pitch) & np.isfinite(yaw)
        if attitude_valid.all():
            local_x, local_y, local_z = _apply_lever_arm(local_x, local_y, local_z, roll, pitch, yaw, lever)
            lever_arm_applied = True

    fix_raw = _nearest_values(rtk_t, rtk.get("rtk_fix_type"), trace_t, rtk_valid, "未知")
    fix = np.asarray([_fix_label(v) for v in fix_raw], dtype="<U16")
    satellites = _nearest_values(rtk_t, rtk.get("satellites"), trace_t, rtk_valid, np.nan)
    hdop = _interp_numeric(rtk_t, rtk.get("hdop"), trace_t, rtk_valid)
    pdop = _interp_numeric(rtk_t, rtk.get("pdop"), trace_t, rtk_valid)
    speed_mps = _interp_numeric(rtk_t, rtk.get("speed_mps"), trace_t, rtk_valid)

    steps = np.hypot(np.diff(local_x), np.diff(local_y))
    distance = np.zeros(trace_t.size, dtype=np.float64)
    if steps.size:
        clean_steps = np.where(np.isfinite(steps), steps, 0.0)
        distance[1:] = np.cumsum(clean_steps)
    if trace_distance_hint_m is not None:
        hint = np.asarray(trace_distance_hint_m, dtype=np.float64)
        if hint.size == trace_t.size and np.all(np.diff(hint[np.isfinite(hint)]) >= 0):
            distance = hint.copy()

    status = np.full(trace_t.size, "aligned", dtype="<U32")
    status[~rtk_cov] = "rtk_out_of_range"
    status[rtk_cov & ~rtk_valid] = "rtk_time_residual"
    if imu_t.size:
        status[~imu_cov & rtk_valid] = "imu_out_of_range"
        status[imu_cov & ~imu_valid & rtk_valid] = "imu_time_residual"
    if alt_t.size:
        status[~alt_cov & rtk_valid & (status == "aligned")] = "altimeter_out_of_range"
        status[alt_cov & ~alt_valid & rtk_valid & (status == "aligned")] = "altimeter_time_residual"

    points: list[TrajectoryPoint] = []
    for idx in range(trace_t.size):
        points.append(
            TrajectoryPoint(
                distance_m=float(distance[idx]),
                x=float(local_x[idx]),
                y=float(local_y[idx]),
                z=float(local_z[idx]),
                yaw_deg=float(yaw[idx]),
                quality=str(fix[idx]),
                longitude=float(lon[idx]),
                latitude=float(lat[idx]),
                coordinate_system=coord_name,
                timestamp_s=float(trace_t[idx]),
                roll_deg=float(roll[idx]),
                pitch_deg=float(pitch[idx]),
                satellites=int(satellites[idx]) if np.isfinite(satellites[idx]) else -1,
                hdop=float(hdop[idx]),
                pdop=float(pdop[idx]),
                flight_height_m=float(height_agl[idx]),
                alignment_status=str(status[idx]),
                trace_index=idx,
            )
        )

    finite_steps = steps[np.isfinite(steps)]
    median_step = float(np.median(finite_steps)) if finite_steps.size else 0.0
    maximum_step = float(np.max(finite_steps)) if finite_steps.size else 0.0
    jump_threshold = max(median_step * 8.0, 5.0)
    profile = SensorCalibrationProfile.from_value(cfg.calibration_profile)
    base_position_sigma = float(profile.position_sigma_m if profile else 0.05)
    timing_sigma = float(profile.timing_sigma_s if profile else 0.01)
    attitude_sigma = float(profile.attitude_sigma_deg if profile else 0.5)
    height_sigma = float(profile.height_sigma_m if profile else 0.10)
    speed_for_sigma = np.where(np.isfinite(speed_mps), np.maximum(speed_mps, 0.0), 0.0)
    position_sigma = np.sqrt(base_position_sigma ** 2 + (speed_for_sigma * (rtk_residual + timing_sigma)) ** 2)
    position_sigma[~rtk_valid] = np.nan

    diagnostics = SensorSyncDiagnostics(
        trace_count=int(trace_t.size),
        trace_start_s=float(trace_t[0]),
        trace_end_s=float(trace_t[-1]),
        trace_nonmonotonic_step_count=int(np.count_nonzero(np.diff(trace_t) < 0)),
        trace_duplicate_timestamp_count=int(np.count_nonzero(np.diff(trace_t) == 0)),
        rtk=_stream_diagnostics("rtk", rtk_t, trace_t, gap_warning_s=cfg.gap_warning_s, accepted=rtk_valid, raw_stats=rtk_raw_stats),
        imu=_stream_diagnostics("imu", imu_t, trace_t, gap_warning_s=cfg.gap_warning_s, accepted=imu_valid, raw_stats=imu_raw_stats),
        altimeter=_stream_diagnostics("altimeter", alt_t, trace_t, gap_warning_s=cfg.gap_warning_s, accepted=alt_valid, raw_stats=alt_raw_stats),
        fixed_solution_ratio=float(np.mean(fix == "固定解")),
        float_solution_ratio=float(np.mean(fix == "浮动解")),
        unknown_solution_ratio=float(np.mean((fix != "固定解") & (fix != "浮动解"))),
        distance_reverse_count=int(np.count_nonzero(np.diff(distance) < 0)),
        position_jump_count=int(np.count_nonzero(finite_steps > jump_threshold)),
        maximum_position_step_m=maximum_step,
        median_position_step_m=median_step,
        lever_arm_applied=lever_arm_applied,
        calibration_profile_id=str(profile.profile_id if profile else ""),
        uncertainty_summary={
            "median_position_sigma_m": float(np.nanmedian(position_sigma)) if np.isfinite(position_sigma).any() else float("nan"),
            "attitude_sigma_deg": attitude_sigma,
            "height_sigma_m": height_sigma,
            "timing_sigma_s": timing_sigma,
        },
        clock_models={
            "radar": asdict(ClockDomain.from_value(cfg.radar_clock, default_name="radar")),
            "rtk": {"clock": asdict(ClockDomain.from_value(cfg.rtk_clock, default_name="rtk")), "alignment": asdict(TimeAlignmentModel.from_value(cfg.rtk_alignment, offset_s=cfg.rtk_time_offset_s))},
            "imu": {"clock": asdict(ClockDomain.from_value(cfg.imu_clock, default_name="imu")), "alignment": asdict(TimeAlignmentModel.from_value(cfg.imu_alignment, offset_s=cfg.imu_time_offset_s))},
            "altimeter": {"clock": asdict(ClockDomain.from_value(cfg.altimeter_clock, default_name="altimeter")), "alignment": asdict(TimeAlignmentModel.from_value(cfg.altimeter_alignment, offset_s=cfg.altimeter_time_offset_s))},
        },
    )
    if diagnostics.rtk.coverage_ratio < 1.0:
        diagnostics.warnings.append(f"RTK 仅覆盖 {diagnostics.rtk.coverage_ratio:.1%} 的雷达道，超出区间未静默夹取。")
    if imu_t.size and diagnostics.imu.coverage_ratio < 1.0:
        diagnostics.warnings.append(f"IMU 仅覆盖 {diagnostics.imu.coverage_ratio:.1%} 的雷达道。")
    if diagnostics.fixed_solution_ratio < 0.9:
        diagnostics.warnings.append(f"RTK 固定解覆盖率为 {diagnostics.fixed_solution_ratio:.1%}，低于建议值 90%。")
    if diagnostics.rtk.max_nearest_residual_s > cfg.maximum_nearest_residual_s:
        diagnostics.warnings.append(
            f"RTK 最大时间残差 {diagnostics.rtk.max_nearest_residual_s:.3f} s，超过阈值 {cfg.maximum_nearest_residual_s:.3f} s；超限道已标为无效。"
        )
    if diagnostics.position_jump_count:
        diagnostics.warnings.append(f"检测到 {diagnostics.position_jump_count} 个轨迹跳点。")
    if diagnostics.trace_nonmonotonic_step_count:
        diagnostics.warnings.append("雷达道时间戳存在倒退，需检查触发记录。")
    if diagnostics.rtk.raw_nonmonotonic_step_count:
        diagnostics.warnings.append(f"RTK 原始记录顺序存在 {diagnostics.rtk.raw_nonmonotonic_step_count} 次时间倒退；排序后仍保留该诊断。")
    if not np.allclose(lever, 0.0) and not lever_arm_applied:
        diagnostics.warnings.append("配置了杆臂偏移，但 IMU 姿态缺失或无效；本次未应用杆臂改正。")

    metadata: dict[str, np.ndarray] = {
        "trace_index": np.arange(trace_t.size, dtype=np.int32),
        "trace_timestamp_s": trace_t,
        "trace_distance_m": distance.astype(np.float32),
        "longitude": lon,
        "latitude": lat,
        "local_x_m": local_x.astype(np.float64),
        "local_y_m": local_y.astype(np.float64),
        "local_z_m": local_z.astype(np.float64),
        "roll_deg": roll.astype(np.float32),
        "pitch_deg": pitch.astype(np.float32),
        "yaw_deg": yaw.astype(np.float32),
        "flight_height_m": height_agl.astype(np.float32),
        "rtk_fix_type": fix,
        "satellites": satellites.astype(np.float32),
        "hdop": hdop.astype(np.float32),
        "pdop": pdop.astype(np.float32),
        "speed_mps": speed_mps.astype(np.float32),
        "alignment_status": status,
        "rtk_valid": rtk_valid.astype(np.uint8),
        "imu_valid": imu_valid.astype(np.uint8),
        "altimeter_valid": alt_valid.astype(np.uint8),
        "position_valid": (rtk_valid & np.isfinite(local_x) & np.isfinite(local_y)).astype(np.uint8),
        "attitude_valid": (np.isfinite(roll) & np.isfinite(pitch) & np.isfinite(yaw)).astype(np.uint8),
        "position_sigma_m": position_sigma.astype(np.float32),
        "attitude_sigma_deg": np.full(trace_t.size, attitude_sigma, dtype=np.float32),
        "height_sigma_m": np.full(trace_t.size, height_sigma, dtype=np.float32),
        "timing_sigma_s": np.full(trace_t.size, timing_sigma, dtype=np.float32),
    }
    return SensorSyncResult(
        trajectory=TrajectoryModel(points),
        trace_metadata=metadata,
        diagnostics=diagnostics,
        config=cfg,
    )

__all__ = ["synchronize_sensor_streams"]
