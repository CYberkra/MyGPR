#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge between the field workbench data model and the existing processors.

Round 5 intentionally connects the new 1080P workbench to the existing
``methods_registry`` / ``processing_engine`` stack without introducing preset
pipelines.  The bridge keeps UI concerns out of the processing engine and keeps
legacy algorithms independent of the field project store.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.gpr_data_model import GPRDataSet
from core.methods_registry import PROCESSING_METHODS
from core.processing_engine import (
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.trajectory_model import TrajectoryModel


# Keep trace-count-changing operations out of the legacy field workbench.  The
# method remains available through the backend processing catalog, where callers
# must explicitly accept the new geometry and trace metadata contract.
HIDDEN_FIELD_METHOD_IDS: set[str] = {"equidistant_trace_resample"}


COMPATIBILITY_CHECK_METHOD_IDS = [
    "dewow",
    "subtracting_average_2D",
    "median_background_2D",
    "svd_bg",
    "frequency_filter_1d",
    "sec_gain",
    "agcGain",
    "trace_median_filter",
    "wavelet_2d",
    "time_to_depth",
]


FIELD_CATEGORY_ORDER = [
    "校正预处理",
    "背景抑制",
    "频率滤波",
    "增益补偿",
    "去噪增强",
    "运动补偿",
    "迁移与深度",
    "其他算法",
]


METHOD_CATEGORY_OVERRIDES = {
    "set_zero_time": "校正预处理",
    "time_cut": "校正预处理",
    "trace_qc": "校正预处理",
    "dewow": "校正预处理",
    "equidistant_trace_resample": "校正预处理",
    "subtracting_average_2D": "背景抑制",
    "median_background_2D": "背景抑制",
    "running_average_2D": "背景抑制",
    "svd_bg": "背景抑制",
    "rpca_background": "背景抑制",
    "fk_filter": "背景抑制",
    "ccbs": "背景抑制",
    "frequency_filter_1d": "频率滤波",
    "compensatingGain": "增益补偿",
    "agcGain": "增益补偿",
    "sec_gain": "增益补偿",
    "energy_decay_gain": "增益补偿",
    "amplitude_scale": "增益补偿",
    "trace_median_filter": "去噪增强",
    "trace_savgol_filter": "去噪增强",
    "hankel_svd": "去噪增强",
    "svd_subspace": "去噪增强",
    "wavelet_2d": "去噪增强",
    "wavelet_svd": "去噪增强",
    "hilbert_envelope": "去噪增强",
    "trajectory_smoothing": "运动补偿",
    "motion_compensation_speed": "运动补偿",
    "motion_compensation_attitude": "运动补偿",
    "motion_compensation_height": "运动补偿",
    "motion_compensation_vibration": "运动补偿",
    "motion_compensation_v2": "运动补偿",
    "stolt_migration": "迁移与深度",
    "kirchhoff_migration": "迁移与深度",
    "time_to_depth": "迁移与深度",
}


METHOD_DISPLAY_NAMES = {
    "set_zero_time": "零时校正",
    "time_cut": "时间窗裁剪",
    "trace_qc": "坏道质检",
    "dewow": "去低频漂移 dewow",
    "equidistant_trace_resample": "等距道重采样（改变道数）",
    "subtracting_average_2D": "平均背景去除",
    "median_background_2D": "中值背景去除",
    "running_average_2D": "尖锐杂波抑制",
    "svd_bg": "SVD 背景抑制",
    "rpca_background": "RPCA 背景抑制",
    "fk_filter": "F-K 滤波",
    "ccbs": "CCBS 滤波",
    "frequency_filter_1d": "一维频率滤波",
    "compensatingGain": "手动增益补偿",
    "agcGain": "AGC 增益",
    "sec_gain": "SEC 增益",
    "energy_decay_gain": "能量衰减增益",
    "amplitude_scale": "振幅归一化",
    "trace_median_filter": "道向中值滤波",
    "trace_savgol_filter": "道向 Savitzky-Golay 平滑",
    "hankel_svd": "Hankel-SVD 去噪",
    "svd_subspace": "SVD 子空间处理",
    "wavelet_2d": "二维小波去噪",
    "wavelet_svd": "小波-SVD 去噪",
    "hilbert_envelope": "Hilbert 包络",
    "trajectory_smoothing": "轨迹平滑",
    "motion_compensation_speed": "速度补偿",
    "motion_compensation_attitude": "姿态补偿",
    "motion_compensation_height": "高度补偿",
    "motion_compensation_vibration": "振动补偿",
    "motion_compensation_v2": "运动补偿 v2",
    "stolt_migration": "Stolt 偏移",
    "kirchhoff_migration": "Kirchhoff 偏移",
    "time_to_depth": "时间-深度转换",
}


PARAM_LABELS = {
    "window": "窗口长度",
    "spacing_m": "目标道间距 m（0=自动）",
    "ntraces": "窗口道数",
    "time_start_ns": "起始时间 ns",
    "time_end_ns": "结束时间 ns",
    "rank": "秩 / 分量数",
    "gain_min": "最小增益",
    "gain_max": "最大增益",
    "strength": "强度",
    "smoothing_samples": "平滑采样点",
    "max_gain": "最大增益",
    "filter_type": "滤波类型",
    "low_freq_mhz": "低截止 MHz",
    "high_freq_mhz": "高截止 MHz",
    "notch_freq_mhz": "陷波中心 MHz",
    "notch_width_mhz": "陷波宽度 MHz",
    "angle_low": "起始角度 °",
    "angle_high": "结束角度 °",
    "taper_width": "过渡宽度 °",
    "mode": "模式",
    "window_traces": "窗口道数",
    "polyorder": "多项式阶数",
    "preserve_mean": "保持均值",
    "new_zero_time": "新零时 ns",
    "scale": "缩放系数",
    "target": "目标幅值",
}


@dataclass(frozen=True)
class FieldMethodDescriptor:
    method_id: str
    category: str
    display_name: str
    raw_name: str
    params_schema: list[dict[str, Any]]
    auto_tune_enabled: bool = False


@dataclass(frozen=True)
class FieldMethodCompatibilityRecord:
    """Deterministic compatibility result for one exposed processing method."""

    method_id: str
    method_name: str
    category: str
    params_ok: bool
    execution_ok: bool
    input_shape: tuple[int, int]
    output_shape: tuple[int, int] | None = None
    sample_count_preserved: bool = False
    trace_count_preserved: bool = False
    finite_output: bool = False
    elapsed_s: float = 0.0
    error: str = ""
    warning_count: int = 0

    @property
    def status(self) -> str:
        if not self.params_ok:
            return "参数异常"
        if not self.execution_ok:
            return "执行失败"
        if not self.finite_output:
            return "输出异常"
        if not self.trace_count_preserved:
            return "需人工复核"
        return "通过"

    def to_report_row(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "method_name": self.method_name,
            "category": self.category,
            "params_ok": self.params_ok,
            "execution_ok": self.execution_ok,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "sample_count_preserved": self.sample_count_preserved,
            "trace_count_preserved": self.trace_count_preserved,
            "finite_output": self.finite_output,
            "elapsed_s": round(float(self.elapsed_s), 4),
            "status": self.status,
            "warning_count": self.warning_count,
            "error": self.error,
        }


def _strip_registry_prefix(name: str) -> str:
    text = str(name).strip()
    # Registry names often start with "4 method_name (...)"; keep the useful part.
    parts = text.split(" ", 1)
    if parts and parts[0].replace(".", "", 1).isdigit() and len(parts) > 1:
        return parts[1].strip()
    return text


def display_name(method_id: str, info: dict[str, Any] | None = None) -> str:
    if method_id in METHOD_DISPLAY_NAMES:
        return METHOD_DISPLAY_NAMES[method_id]
    info = info or PROCESSING_METHODS.get(method_id, {})
    return _strip_registry_prefix(str(info.get("name") or method_id))


def field_category(method_id: str, info: dict[str, Any] | None = None) -> str:
    if method_id in METHOD_CATEGORY_OVERRIDES:
        return METHOD_CATEGORY_OVERRIDES[method_id]
    info = info or PROCESSING_METHODS.get(method_id, {})
    family = str(info.get("auto_tune_family") or info.get("auto_tune_stage") or "")
    if family in {"background", "fk", "impulse"}:
        return "背景抑制"
    if family == "gain":
        return "增益补偿"
    if family == "drift" or family == "zero_time":
        return "校正预处理"
    if family == "motion_comp":
        return "运动补偿"
    if family in {"denoise", "wavelet"}:
        return "去噪增强"
    return "其他算法"


def iter_field_methods() -> list[FieldMethodDescriptor]:
    methods: list[FieldMethodDescriptor] = []
    for method_id, info in PROCESSING_METHODS.items():
        if method_id in HIDDEN_FIELD_METHOD_IDS:
            continue
        methods.append(
            FieldMethodDescriptor(
                method_id=method_id,
                category=field_category(method_id, info),
                display_name=display_name(method_id, info),
                raw_name=str(info.get("name") or method_id),
                params_schema=list(info.get("params") or []),
                auto_tune_enabled=bool(info.get("auto_tune_enabled", False)),
            )
        )
    order = {name: idx for idx, name in enumerate(FIELD_CATEGORY_ORDER)}
    methods.sort(key=lambda m: (order.get(m.category, 999), m.display_name.lower(), m.method_id))
    return methods


def get_field_method_categories() -> dict[str, list[FieldMethodDescriptor]]:
    grouped: dict[str, list[FieldMethodDescriptor]] = {name: [] for name in FIELD_CATEGORY_ORDER}
    for method in iter_field_methods():
        grouped.setdefault(method.category, []).append(method)
    return {category: methods for category, methods in grouped.items() if methods}


def get_method_info(method_id: str) -> dict[str, Any]:
    if method_id in HIDDEN_FIELD_METHOD_IDS:
        raise KeyError(f"Field workbench does not expose method: {method_id}")
    if method_id not in PROCESSING_METHODS:
        raise KeyError(method_id)
    return PROCESSING_METHODS[method_id]


def get_method_params_schema(method_id: str) -> list[dict[str, Any]]:
    return list(get_method_info(method_id).get("params") or [])


def default_params(method_id: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for spec in get_method_params_schema(method_id):
        if "default" in spec:
            params[str(spec["name"])] = spec.get("default")
    return params


def recommended_params(method_id: str, dataset: GPRDataSet | None = None) -> dict[str, Any]:
    """Return safe UI defaults for the currently selected method only."""
    info = get_method_info(method_id)
    params = default_params(method_id)
    candidates = info.get("auto_tune_candidates") or {}
    if isinstance(candidates, dict):
        for key, values in candidates.items():
            if str(key).startswith("_"):
                continue
            if isinstance(values, (list, tuple)) and values:
                # Use a middle candidate where possible instead of the smallest value.
                params[str(key)] = values[min(len(values) // 2, len(values) - 1)]
    if dataset is not None:
        traces = int(dataset.trace_count)
        samples = int(dataset.sample_count)
        if "ntraces" in params:
            # Keep window odd and visibly local; do not alter the trace count.
            value = int(min(max(7, traces // 12), 101))
            if value % 2 == 0:
                value += 1
            params["ntraces"] = value
        if "window" in params:
            value = int(min(max(11, samples // 16), 257))
            if value % 2 == 0:
                value += 1
            params["window"] = value
    return params


def build_header_info(dataset: GPRDataSet) -> dict[str, Any]:
    trace_interval = 0.0
    if dataset.distance_axis_m.size >= 2:
        diffs = np.diff(np.asarray(dataset.distance_axis_m, dtype=float))
        finite = diffs[np.isfinite(diffs)]
        if finite.size:
            trace_interval = float(np.median(finite))
    return {
        "line_id": dataset.line_id,
        "a_scan_length": int(dataset.sample_count),
        "num_traces": int(dataset.trace_count),
        "total_time_ns": float(dataset.time_window_ns),
        "trace_interval_m": float(trace_interval),
        "track_length_m": float(dataset.length_m),
        "dielectric_constant": float(dataset.dielectric_constant),
        "source_path": dataset.source_path,
        "format_name": dataset.format_name,
    }


def build_trace_metadata(dataset: GPRDataSet, trajectory: TrajectoryModel | None = None) -> dict[str, np.ndarray]:
    metadata: dict[str, np.ndarray] = {
        "trace_distance_m": np.asarray(dataset.distance_axis_m, dtype=np.float64),
    }
    if trajectory is not None:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        quality: list[float] = []
        for dist in dataset.distance_axis_m:
            point = trajectory.interpolate(float(dist))
            xs.append(point.x)
            ys.append(point.y)
            zs.append(point.z)
            quality.append(1.0 if "固定" in point.quality else 0.5)
        metadata.update(
            {
                "x_m": np.asarray(xs, dtype=np.float64),
                "y_m": np.asarray(ys, dtype=np.float64),
                "z_m": np.asarray(zs, dtype=np.float64),
                "rtk_quality_code": np.asarray(quality, dtype=np.float64),
            }
        )
    return metadata


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _array_sha256(
    array: np.ndarray,
    *,
    cancel_checker=None,
    progress_callback=None,
    label: str = "数据校验",
) -> str:
    arr = np.asanyarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(int(v) for v in arr.shape)).encode("utf-8"))
    rows = int(arr.shape[0]) if arr.ndim else 1
    chunk_rows = max(1, min(rows, 256))
    for start in range(0, rows, chunk_rows):
        if cancel_checker is not None and cancel_checker():
            from core.job_manager import JobCancelled
            raise JobCancelled("任务已取消")
        end = min(rows, start + chunk_rows)
        chunk = np.ascontiguousarray(arr[start:end] if arr.ndim else arr)
        digest.update(memoryview(chunk).cast("B"))
        if progress_callback is not None:
            progress_callback(end, rows, label)
    return digest.hexdigest()


def _finite_summary(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return {"finite_ratio": 0.0, "nan_ratio": 1.0, "amplitude_min": 0.0, "amplitude_max": 0.0}
    mask = np.isfinite(arr)
    finite = arr[mask]
    ratio = float(mask.sum() / arr.size)
    if finite.size == 0:
        return {"finite_ratio": ratio, "nan_ratio": 1.0 - ratio, "amplitude_min": 0.0, "amplitude_max": 0.0}
    return {
        "finite_ratio": ratio,
        "nan_ratio": 1.0 - ratio,
        "amplitude_min": float(np.nanmin(finite)),
        "amplitude_max": float(np.nanmax(finite)),
    }


def run_registered_method(
    dataset: GPRDataSet,
    method_id: str,
    params: dict[str, Any] | None = None,
    *,
    trajectory: TrajectoryModel | None = None,
    cancel_checker=None,
    progress_callback=None,
) -> tuple[GPRDataSet, dict[str, Any]]:
    """Execute one registered method and return a normalized output dataset."""
    info = get_method_info(method_id)
    ui_params = dict(params or {})
    header_info = build_header_info(dataset)
    trace_metadata = build_trace_metadata(dataset, trajectory)
    runtime_params = prepare_runtime_params(
        method_id,
        ui_params,
        header_info,
        trace_metadata,
        dataset.matrix.shape,
    )
    if cancel_checker is not None:
        runtime_params.setdefault("cancel_checker", cancel_checker)
    if progress_callback is not None:
        progress_callback(0, 4, "准备算法参数")
    if cancel_checker is not None and cancel_checker():
        from core.job_manager import JobCancelled
        raise JobCancelled("任务已取消")
    start = time.perf_counter()
    if progress_callback is not None:
        progress_callback(1, 4, f"执行 {display_name(method_id, info)}")
    result, result_meta = run_processing_method(
        dataset.matrix, method_id, runtime_params, cancel_checker=cancel_checker
    )
    elapsed_s = time.perf_counter() - start
    header_out = merge_result_header_info(header_info, result_meta, result.shape)
    trace_out = merge_result_trace_metadata(trace_metadata, result_meta)

    # Most exposed methods preserve trace count.  If a method returns a different
    # shape, rebuild axes safely from the project length/time window instead of
    # attempting to reuse incompatible arrays.
    output = GPRDataSet.from_matrix(
        dataset.line_id,
        result,
        length_m=float(header_out.get("track_length_m", dataset.length_m) or dataset.length_m),
        time_window_ns=float(header_out.get("total_time_ns", dataset.time_window_ns) or dataset.time_window_ns),
        dielectric_constant=float(header_out.get("dielectric_constant", dataset.dielectric_constant) or dataset.dielectric_constant),
        source_path=dataset.source_path,
        format_name=f"processed:{method_id}",
        metadata={
            **dict(dataset.metadata or {}),
            "processing_method_id": method_id,
            "processing_method_name": display_name(method_id, info),
        },
    )
    input_shape = tuple(int(v) for v in dataset.matrix.shape)
    output_shape = tuple(int(v) for v in result.shape)
    warnings = result_meta.get("runtime_warnings", []) if isinstance(result_meta, dict) else []
    warnings = list(warnings or [])
    if output_shape[1] != input_shape[1]:
        warnings.append(
            {
                "code": "trace_count_changed",
                "message": "输出道数与输入不一致，保存前需要人工复核。",
                "method_id": method_id,
                "input_shape": list(input_shape),
                "output_shape": list(output_shape),
            }
        )
    if not np.isfinite(result).all():
        warnings.append(
            {
                "code": "non_finite_output",
                "message": "算法输出包含 NaN 或 Inf，保存前需要人工复核。",
                "method_id": method_id,
            }
        )
    artifact_role = "display_compare_transform" if method_id == "time_to_depth" else "processing_result"
    axis_transform = {}
    if method_id == "time_to_depth":
        axis_transform = {
            "kind": "time_to_depth",
            "source_axis": "time_ns",
            "target_axis": "depth_m",
            "display_compare_page": "planned_subpanel",
            "note": "时间-深度转换作为显示与对比能力保留，输出轴变化必须由结果页/标注页读取 manifest 判断。",
        }

    if progress_callback is not None:
        progress_callback(2, 4, "生成处理清单")
    manifest = {
        "schema": "mygpr.processing_manifest.v2",
        "status": "success",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "artifact_role": artifact_role,
        "axis_transform": axis_transform,
        "engine": "core.processing_engine",
        "bridge": "core.field_processing_bridge",
        "line_id": dataset.line_id,
        "source_line_id": dataset.line_id,
        "method_id": method_id,
        "method_name": display_name(method_id, info),
        "category": field_category(method_id, info),
        "params": _json_safe(ui_params),
        "runtime_params": _json_safe({k: v for k, v in runtime_params.items() if k not in {"header_info", "trace_metadata", "cancel_checker"}}),
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "sample_count_changed": output_shape[0] != input_shape[0],
        "trace_count_changed": output_shape[1] != input_shape[1],
        "elapsed_s": round(float(elapsed_s), 4),
        "input_dataset": _json_safe(dataset.to_metadata()),
        "output_dataset": _json_safe(output.to_metadata()),
        "input_data_sha256": _array_sha256(
            dataset.matrix, cancel_checker=cancel_checker, label="校验输入数据"
        ),
        "output_data_sha256": _array_sha256(
            output.matrix, cancel_checker=cancel_checker, label="校验输出数据"
        ),
        "input_finite_summary": _finite_summary(dataset.matrix),
        "output_finite_summary": _finite_summary(output.matrix),
        "result_meta": _json_safe(result_meta),
        "trace_metadata_out": _json_safe(trace_out),
        "warnings": _json_safe(warnings),
    }
    if progress_callback is not None:
        progress_callback(4, 4, "处理完成")
    return output, manifest


def check_method_compatibility(
    dataset: GPRDataSet,
    method_id: str,
    *,
    trajectory: TrajectoryModel | None = None,
) -> FieldMethodCompatibilityRecord:
    """Run a small compatibility check without changing project state."""
    input_shape = tuple(int(v) for v in dataset.matrix.shape)
    info = PROCESSING_METHODS.get(method_id, {})
    method_name = display_name(method_id, info)
    category = field_category(method_id, info)
    try:
        params = recommended_params(method_id, dataset)
    except Exception as exc:
        return FieldMethodCompatibilityRecord(
            method_id=method_id,
            method_name=method_name,
            category=category,
            params_ok=False,
            execution_ok=False,
            input_shape=input_shape,
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        output, manifest = run_registered_method(
            dataset,
            method_id,
            params,
            trajectory=trajectory,
        )
    except Exception as exc:
        return FieldMethodCompatibilityRecord(
            method_id=method_id,
            method_name=method_name,
            category=category,
            params_ok=True,
            execution_ok=False,
            input_shape=input_shape,
            error=f"{type(exc).__name__}: {exc}",
        )
    output_shape = tuple(int(v) for v in output.matrix.shape)
    return FieldMethodCompatibilityRecord(
        method_id=method_id,
        method_name=method_name,
        category=category,
        params_ok=True,
        execution_ok=True,
        input_shape=input_shape,
        output_shape=output_shape,
        sample_count_preserved=output_shape[0] == input_shape[0],
        trace_count_preserved=output_shape[1] == input_shape[1],
        finite_output=bool(np.isfinite(output.matrix).all()),
        elapsed_s=float(manifest.get("elapsed_s", 0.0)),
        warning_count=len(manifest.get("warnings", []) or []),
    )


def run_priority_compatibility_checks(
    dataset: GPRDataSet | None = None,
) -> list[FieldMethodCompatibilityRecord]:
    """Check the v0.8.73 priority methods on a deterministic dataset."""
    data = dataset or GPRDataSet.synthetic("L03", rows=160, cols=96, length_m=40.0)
    return [check_method_compatibility(data, method_id) for method_id in COMPATIBILITY_CHECK_METHOD_IDS]


__all__ = [
    "FieldMethodCompatibilityRecord",
    "FieldMethodDescriptor",
    "COMPATIBILITY_CHECK_METHOD_IDS",
    "FIELD_CATEGORY_ORDER",
    "HIDDEN_FIELD_METHOD_IDS",
    "PARAM_LABELS",
    "build_header_info",
    "build_trace_metadata",
    "check_method_compatibility",
    "default_params",
    "display_name",
    "field_category",
    "get_field_method_categories",
    "get_method_info",
    "get_method_params_schema",
    "iter_field_methods",
    "recommended_params",
    "run_priority_compatibility_checks",
    "run_registered_method",
]
