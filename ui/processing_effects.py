"""Compact, truthful processing-action descriptions for saved result presentation."""

from collections.abc import Mapping
from dataclasses import dataclass
import json

from ui.desktop_backend_facade import METHOD_CATEGORY_LABELS, METHOD_METADATA


_ACTIONS = {
    "dewow": "去漂移",
    "set_zero_time": "零时校正",
    "time_cut": "时间裁剪",
    "trace_cut": "测道裁剪",
    "trace_qc": "坏道处理",
    "subtracting_average_2D": "抑背景",
    "median_background_2D": "中值抑背景",
    "svd_bg": "SVD抑背景",
    "ccbs": "相关抑背景",
    "sliding_avg": "滑窗抑背景",
    "agcGain": "增益",
    "sec_gain": "SEC增益",
    "energy_decay_gain": "衰减补偿",
    "compensatingGain": "线性增益",
    "amplitude_scale": "振幅缩放",
    "frequency_filter_1d": "频域滤波",
    "fk_filter": "F-K滤波",
    "trace_median_filter": "道间去噪",
    "trace_savgol_filter": "道间平滑",
    "hankel_svd": "秩滤波去噪",
    "svd_subspace": "子空间去噪",
    "wavelet_2d": "小波去噪",
    "motion_compensation_height": "高度补偿",
    "motion_compensation_speed": "速度补偿",
    "motion_compensation_attitude": "姿态补偿",
    "motion_compensation_vibration": "振动抑制",
    "motion_compensation_v2": "运动补偿",
    "stolt_migration": "Stolt偏移",
    "kirchhoff_migration": "Kirchhoff偏移",
    "rtm_migration": "逆时偏移",
    "time_to_depth": "时深转换",
}

_PARAMETER_NAMES = {
    "window": "window",
    "window_size": "window",
    "low_hz": "low",
    "high_hz": "high",
    "low_mhz": "low",
    "high_mhz": "high",
    "dielectric_constant": "εr",
    "velocity_m_per_ns": "v",
    "start_time_ns": "start",
    "end_time_ns": "end",
    "gain": "gain",
    "alpha": "α",
}


@dataclass(frozen=True)
class StepEffect:
    action: str
    category: str
    parameter_summary: str
    outcome: str
    warnings: tuple[str, ...] = ()


def _format_value(value):
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (str, int)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _parameters(step):
    params = step.get("params") or {}
    if not isinstance(params, Mapping):
        return ""
    parts = []
    for key, value in params.items():
        if key not in _PARAMETER_NAMES:
            continue
        parts.append(f"{_PARAMETER_NAMES[key]}={_format_value(value)}")
        if len(parts) == 3:
            break
    if not parts:
        for key, value in list(params.items())[:2]:
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}={_format_value(value)}")
    return ", ".join(parts)


def describe_step(step: Mapping) -> StepEffect:
    method_id = str(step.get("method_id") or "未知方法")
    metadata = METHOD_METADATA.get(method_id, {})
    action = _ACTIONS.get(method_id)
    if not action:
        action = str(metadata.get("display_name") or method_id).split("（", 1)[0]
    category_key = str(metadata.get("category") or "experimental")
    category = str(METHOD_CATEGORY_LABELS.get(category_key) or category_key)
    details = step.get("metadata") or {}
    if not isinstance(details, Mapping):
        details = {}
    skipped = bool(details.get("skipped") or (details.get("effective_params") or {}).get("skipped"))
    reason = str(details.get("reason") or (details.get("effective_params") or {}).get("reason") or "")
    warnings = tuple(
        str(item.get("message") or item.get("body") or item)
        for item in step.get("runtime_warnings") or []
        if isinstance(item, Mapping) or str(item)
    )
    if skipped:
        outcome = f"已跳过（{reason}）" if reason else "已跳过"
    elif warnings:
        outcome = "已执行，有警告"
    else:
        outcome = "已执行"
    if skipped and reason:
        warnings = warnings + (reason,)
    return StepEffect(action, category, _parameters(step), outcome, warnings)
