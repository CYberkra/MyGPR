#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DesktopBackendFacade — ui/ 唯一允许接触的 core/ 与跨层导入通道。

所有 ``ui/`` 模块必须通过本文件获取核心数据模型、GUI 渲染辅助、
方法注册表元数据、文件格式过滤器及 job 状态类型；禁止在 ui/ 内直接
``import core.*`` 或 ``import mygpr.domain.*`` / ``import mygpr.application.*``。

本文件是 **deepened** facade：对外只暴露 UI-facing DTO 和 wrapped functions，
raw core / domain 类型只在内部翻译后通过 DTO 返回。
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from ui.desktop_backend_dtos import (
    UiJobSnapshot,
    UiMethodEntry,
    UiPipelineDefinition,
    UiPipelineStep,
)
from mygpr.application.jobs.models import JobEventType, JobResultSummary  # noqa: E402 — re-exported for ui/controllers

# ---------------------------------------------------------------------------
# Backend imports (the only place in ui/ that touches core/mygpr directly)
# ---------------------------------------------------------------------------
from core.gpr_data_model import GPRDataSet
from core.gui_rendering import bundle_from_dataset, compute_levels
from core.gpr_format_registry import supported_file_dialog_filter
from core.app_paths import get_tile_cache_dir
from core.method_registry_metadata import (
    METHOD_CATEGORY_LABELS,
    METHOD_METADATA,
    METHOD_TAGS,
    PREFERRED_METHOD_ORDER,
)
from core.methods_registry import PROCESSING_METHODS
from mygpr.domain.acquisition.models import SensorSyncSettings
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

__all__ = [
    "GPRDataSet",
    "JobEventType",
    "JobResultSummary",
    "PipelineDefinition",
    "PipelineStep",
    "SensorSyncSettings",
]

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public wrapped functions
# ---------------------------------------------------------------------------

def compute_display_levels(
    matrix: Any,
    *,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> tuple[float, float]:
    """Compute nan/inf-safe percentile display levels for a matrix."""
    return compute_levels(matrix, p_low=p_low, p_high=p_high)


def build_preview_bundle(
    line_id: str,
    matrix: Any,
    *,
    distance_axis_m: Any = None,
    time_axis_ns: Any = None,
    title: str = "",
    p_low: float = 2.0,
    p_high: float = 98.0,
    time_window_ns: float = 250.0,
) -> Any:
    """Build a :class:`PreviewBundle` from raw array data.

    Constructs a minimal ``GPRDataSet``-like object internally and forwards
    to ``core.gui_rendering.bundle_from_dataset``.  The UI never sees
    ``GPRDataSet``.  ``time_window_ns`` 必须为真实测线时窗（P1-5），否则
    预览纵轴物理刻度错误。
    """
    sample_count = int(getattr(matrix, "shape", (0,))[0])
    trace_count = int(getattr(matrix, "shape", (0, 1))[1])
    time_window_ns = float(time_window_ns or 250.0)
    dataset = GPRDataSet(
        line_id=line_id,
        matrix=matrix,
        distance_axis_m=distance_axis_m if distance_axis_m is not None else [],
        time_axis_ns=time_axis_ns if time_axis_ns is not None else [],
        depth_axis_m=[],
        sample_count=sample_count,
        trace_count=trace_count,
        time_window_ns=time_window_ns,
    )
    return bundle_from_dataset(dataset, title=title, p_low=p_low, p_high=p_high)


def file_dialog_filter() -> str:
    """Return the ``;;``-joined file filter string for GPR import dialogs."""
    return supported_file_dialog_filter()


def tile_cache_dir() -> str:
    """Return the shared user-writable basemap and terrain cache directory."""
    return get_tile_cache_dir()


# 常见参数名 → 中文标签（注册表未提供 label 时的显示回退）
_PARAM_LABELS = {
    "mode": "模式", "time_start_ns": "起始时间 (ns)", "time_end_ns": "结束时间 (ns)",
    "new_zero_time": "新零时刻 (ns)", "window": "窗口长度", "strength": "强度",
    "smoothing_samples": "平滑采样数", "min_gain": "最小增益", "max_gain": "最大增益",
    "floor_ratio": "基底比例", "scale": "缩放系数", "target": "目标值",
    "ntraces": "道数", "wavelet": "小波基", "levels": "分解层数",
    "threshold": "阈值", "threshold_strategy": "阈值策略", "rank_start": "起始秩",
    "rank_end": "终止秩", "normalize": "归一化", "log_compress": "对数压缩",
    "use_custom_ref": "自定义参考", "dt": "时间采样 (ns)", "v": "波速 (m/ns)",
    "dz": "深度步长 (m)", "gain_min": "增益下限", "gain_max": "增益上限",
    "empty_rms_threshold": "空道 RMS 阈值", "spike_zscore": "尖峰 Z 分数",
    "manual_trace_indices": "手动指定道号", "spacing_m": "道间距 (m)",
    "t0": "起始时刻 (ns)", "t1": "结束时刻 (ns)", "order": "阶数",
    "fmin": "下限频率 (MHz)", "fmax": "上限频率 (MHz)", "rank": "秩",
    "power": "幂次", "tmax": "最大时窗 (ns)", "factor": "系数",
    "alpha": "权重系数", "lambda_": "正则系数", "max_iter": "最大迭代次数",
    "tol": "收敛容差", "sigma": "标准差", "radius": "半径", "width": "宽度",
    "height": "高度", "depth": "深度", "velocity": "速度", "frequency": "频率",
    "amplitude": "幅度", "phase": "相位", "offset": "偏移量", "ratio": "比例",
    "method": "方法", "axis": "轴向", "dtype": "数据类型", "enabled": "启用",
    "start_trace": "起始道", "end_trace": "结束道", "start_sample": "起始采样点",
    "end_sample": "结束采样点", "traces": "道数", "samples": "采样数",
    "cutoff": "截止频率", "lowcut": "低通截止", "highcut": "高通截止",
    "filter_order": "滤波器阶数", "apply_agc": "应用 AGC", "agc_window": "AGC 窗口",
    "clip": "限幅", "eps": "最小除数", "start": "起始", "end": "结束",
    "step": "步长", "size": "尺寸", "length": "长度", "num": "数量",
}


def _extract_parameter_schema(raw: Mapping[str, Any]) -> tuple[dict, ...]:
    """从注册表原始描述提取参数 schema（统一为 list[dict]，每项必含 name）。

    兼容两种注册表形态：
    - ``parameter_schema`` 为 dict[name -> spec]（domain/infrastructure 侧）
    - ``params`` 为 list[spec]（core.methods_registry 侧，spec 自带 name）

    此前只读 ``parameter_schema`` 键，而 core 注册表实际叫 ``params``，
    导致 36 个方法的参数 schema 传到 UI 全为空，参数表单无法构建、
    "应用到选中步骤" 永远禁用 —— 本函数即该缺陷的修复点。
    """
    items: list[dict] = []
    dict_schema = raw.get("parameter_schema")
    if isinstance(dict_schema, Mapping) and dict_schema:
        for key, item in dict_schema.items():
            entry = dict(item) if isinstance(item, Mapping) else {}
            entry.setdefault("name", str(key))
            items.append(entry)
    else:
        list_schema = raw.get("params")
        if isinstance(list_schema, (list, tuple)):
            for index, item in enumerate(list_schema):
                entry = dict(item) if isinstance(item, Mapping) else {}
                entry.setdefault("name", "param_%d" % index)
                items.append(entry)
    for entry in items:
        entry.setdefault("label", _PARAM_LABELS.get(
            str(entry.get("name", "")), str(entry.get("name", ""))))
    return tuple(items)


def method_catalog() -> tuple[UiMethodEntry, ...]:
    """Return the processing method catalog as typed, immutable DTOs.

    Each entry is built by merging the raw registry descriptor with
    display metadata from ``core.method_registry_metadata``.
    """
    order = {method_id: index for index, method_id in enumerate(PREFERRED_METHOD_ORDER)}
    items: list[UiMethodEntry] = []
    for method_id, raw in PROCESSING_METHODS.items():
        meta = METHOD_METADATA.get(method_id, {})
        category = str(meta.get("category") or raw.get("category") or "experimental")
        tag = METHOD_TAGS.get(method_id)
        entry = UiMethodEntry(
            method_id=str(method_id),
            name=str(raw.get("name", method_id)),
            display_name=str(meta.get("display_name") or raw.get("name", method_id)),
            category=category,
            category_label=str(METHOD_CATEGORY_LABELS.get(category, category)),
            tags=(str(tag),) if tag else (),
            auto_tune_enabled=bool(raw.get("auto_tune_enabled", False)),
            parameter_schema=_extract_parameter_schema(raw),
            description=str(raw.get("description") or ""),
        )
        items.append(entry)
    items.sort(key=lambda entry: (order.get(entry.method_id, len(order)), entry.method_id))
    return tuple(items)


def pipeline_from_dicts(
    steps: list[dict[str, Any]],
    name: str = "",
) -> UiPipelineDefinition:
    """Convert a page-provided list of step dicts into an immutable pipeline DTO."""
    ui_steps = tuple(
        UiPipelineStep(
            method_id=str(step.get("method_id", "")),
            params=dict(step.get("params") or {}),
            enabled=bool(step.get("enabled", True)),
            label=str(step.get("label", "") or step.get("method_id", "")),
        )
        for step in steps
    )
    return UiPipelineDefinition(steps=ui_steps, name=str(name or "Processing pipeline"))


def pipeline_to_raw(ui_pipeline: UiPipelineDefinition) -> PipelineDefinition:
    """Convert a UI pipeline DTO back to the domain ``PipelineDefinition`` the backend expects."""
    raw_steps = tuple(
        PipelineStep(
            method_id=step.method_id,
            params=dict(step.params),
            enabled=step.enabled,
            label=step.label,
        )
        for step in ui_pipeline.steps
    )
    return PipelineDefinition(steps=raw_steps, name=ui_pipeline.name)


def job_snapshot_from_raw(raw: Any) -> UiJobSnapshot:
    """Adapt a raw backend ``JobSnapshot`` into a UI-safe DTO."""
    return UiJobSnapshot.from_raw(raw)


# ---------------------------------------------------------------------------
# Re-exports required by ui/controllers/* (no raw core/mygpr types in __all__)
# ---------------------------------------------------------------------------

__all__ = [
    # wrapped functions
    "compute_display_levels",
    "build_preview_bundle",
    "file_dialog_filter",
    "tile_cache_dir",
    "method_catalog",
    "pipeline_from_dicts",
    "pipeline_to_raw",
    "job_snapshot_from_raw",
    # DTOs (for isinstance checks and signal payloads)
    "UiMethodEntry",
    "UiPipelineStep",
    "UiPipelineDefinition",
    "UiJobSnapshot",
    # re-exported enums / lightweight types
    "JobEventType",
    "JobResultSummary",
]
