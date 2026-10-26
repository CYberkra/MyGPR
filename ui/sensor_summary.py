"""Compact presentation of trace-aligned sensor availability (no file I/O)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorAvailability:
    """One trace-aligned group; unknown counts remain distinct from zero valid values."""

    label: str
    state: str
    total: int
    valid_count: int | None
    missing_count: int | None


@dataclass(frozen=True)
class SensorCompletenessReport:
    """Immutable statistics shared by structured UI and legacy text consumers."""

    total: int
    rows: tuple[SensorAvailability, ...]
    error: str = ''

    def as_text(self) -> str:
        if self.error:
            return self.error
        if self.total <= 0:
            return '传感器完整性：无雷达道，无法统计。'
        lines = [f'传感器完整性 · 共 {self.total} 道']
        for row in self.rows:
            if row.state in ('missing', 'incomplete'):
                state = '未提供' if row.state == 'missing' else '字段不全'
                lines.append(f'{row.label}：{state}，缺失/无效 {row.total} 道')
            elif row.state in ('unreadable', 'mismatch'):
                state = '字段不可读' if row.state == 'unreadable' else '长度不符'
                lines.append(f'{row.label}：{state}，无法统计')
            else:
                count = row.valid_count
                lines.append(
                    f'{row.label}：有效 {count}/{row.total} 道（{count / row.total:.1%}），'
                    f'缺失/无效 {row.missing_count} 道')
        lines.append('有效率仅表示字段可用，不代表RTK固定解、精度或时间同步已验收。')
        return '\n'.join(lines)


def sensor_completeness_report(metadata: dict, trace_count: int) -> SensorCompletenessReport:
    """Count joint finite values once, without filling or sensor-quality claims."""
    total = int(trace_count)
    if total <= 0:
        return SensorCompletenessReport(total=total, rows=())
    height = 'height_agl_m' if 'height_agl_m' in metadata else 'flight_height_m'
    groups = (
        ('RTK经纬度', ('longitude', 'latitude')),
        ('高度', (height,)),
        ('IMU姿态', ('roll_deg', 'pitch_deg', 'yaw_deg')),
        ('IMU加速度', ('accel_x_g', 'accel_y_g', 'accel_z_g')),
        ('IMU角速度', ('angular_rate_x_dps', 'angular_rate_y_dps', 'angular_rate_z_dps')),
        ('采集时间', ('trace_timestamp_s',)),
    )
    rows = []
    for label, fields in groups:
        present = [key in metadata for key in fields]
        if not all(present):
            state = 'missing' if not any(present) else 'incomplete'
            rows.append(SensorAvailability(label, state, total, None, total))
            continue
        try:
            arrays = [np.asarray(metadata[key], dtype=np.float64) for key in fields]
        except (TypeError, ValueError):
            rows.append(SensorAvailability(label, 'unreadable', total, None, None))
            continue
        if any(array.shape != (total,) for array in arrays):
            rows.append(SensorAvailability(label, 'mismatch', total, None, None))
            continue
        valid = np.ones(total, dtype=bool)
        for array in arrays:
            valid &= np.isfinite(array)
        if label == 'RTK经纬度':
            valid &= (np.abs(arrays[0]) <= 180) & (np.abs(arrays[1]) <= 90)
        elif label in ('高度', '采集时间'):
            valid &= arrays[0] > 0
        count = int(np.count_nonzero(valid))
        state = 'complete' if count == total else ('partial' if count else 'invalid')
        rows.append(SensorAvailability(label, state, total, count, total - count))
    return SensorCompletenessReport(total=total, rows=tuple(rows))


def sensor_completeness_text(metadata: dict, trace_count: int) -> str:
    """Compatibility presentation of the structured availability report."""
    return sensor_completeness_report(metadata, trace_count).as_text()
