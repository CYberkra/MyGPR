#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""智能处理工作流 - 数据结构和配置管理

定义方法分类、流程配置结构和配置管理器
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.app_paths import get_workflow_templates_dir


logger = logging.getLogger(__name__)


# ============ 方法分类定义 ============

METHOD_CATEGORIES = {
    "preprocessing": {
        "id": "preprocessing",
        "name": "预处理",
        "icon": "🔧",
        "description": "数据准备和基础校正",
        "methods": [
            "set_zero_time",
            "time_cut",
            "trace_qc",
            "dc_shift",
            "equidistant_trace_resample",
            "dewow",
        ],
    },
    "background_removal": {
        "id": "background_removal",
        "name": "背景抑制",
        "icon": "🧹",
        "description": "去除直达波和地表反射",
        "methods": [
            "frequency_filter_1d",
            "subtracting_average_2D",
            "median_background_2D",
            "svd_bg",
            "fk_filter",
            "ccbs",
        ],
    },
    "gain": {
        "id": "gain",
        "name": "增益补偿",
        "icon": "📈",
        "description": "能量恢复和深度补偿",
        "methods": [
            "sec_gain",
            "energy_decay_gain",
            "compensatingGain",
            "agcGain",
            "amplitude_scale",
        ],
    },
    "denoising": {
        "id": "denoising",
        "name": "去噪",
        "icon": "✨",
        "description": "结构化去噪和信号提纯",
        "methods": ["hankel_svd", "svd_subspace", "wavelet_2d", "wavelet_svd"],
    },
    "attribute_analysis": {
        "id": "attribute_analysis",
        "name": "属性分析",
        "icon": "〰",
        "description": "包络、瞬时属性和解释辅助",
        "methods": ["hilbert_envelope"],
    },
    "migration": {
        "id": "migration",
        "name": "迁移与标定",
        "icon": "🎯",
        "description": "几何校正和深度转换",
        "methods": [
            "manual_velocity_model",
            "geometry_depth_context",
            "stolt_migration",
            "kirchhoff_migration",
            "time_to_depth",
        ],
    },
    "motion_compensation": {
        "id": "motion_compensation",
        "name": "运动补偿",
        "icon": "🚁",
        "description": "无人机GPR运动误差校正",
        "methods": [
            "trajectory_smoothing",
            "motion_compensation_speed",
            "motion_compensation_attitude",
            "motion_compensation_height",
            "motion_compensation_vibration",
            "motion_compensation_v2",
        ],
    },
}


# ============ 标准预设定义 ============

QUICK_PRESETS = {
    "robust_imaging": {
        "name": "稳健成像",
        "description": "标准GPR数据处理流程",
        "methods": [
            {
                "category": "preprocessing",
                "method_id": "set_zero_time",
                "enabled": True,
                "params": {"new_zero_time": 5.0},
            },
            {
                "category": "preprocessing",
                "method_id": "dewow",
                "enabled": True,
                "params": {"window": 41},
            },
            {
                "category": "background_removal",
                "method_id": "fk_filter",
                "enabled": True,
                "params": {"angle_low": 12, "angle_high": 55, "taper_width": 4},
            },
            {
                "category": "background_removal",
                "method_id": "subtracting_average_2D",
                "enabled": True,
                "params": {},
            },
            {
                "category": "gain",
                "method_id": "sec_gain",
                "enabled": True,
                "params": {"gain_min": 1.0, "gain_max": 4.5, "power": 1.1},
            },
            {
                "category": "denoising",
                "method_id": "svd_subspace",
                "enabled": True,
                "params": {"rank_start": 1, "rank_end": 20},
            },
        ],
    },
    "mygpr_standard": {
        "name": "MyGPR 标准流程",
        "description": (
            "原 MyGPR 经典五步处理链：零时矫正、低频漂移矫正、"
            "背景抑制、增益、去噪。"
        ),
        "methods": [
            {
                "category": "preprocessing",
                "stage_id": "zero_time",
                "method_id": "set_zero_time",
                "enabled": True,
                "params": {"new_zero_time": 5.0},
            },
            {
                "category": "preprocessing",
                "stage_id": "trace_correction",
                "method_id": "dewow",
                "enabled": True,
                "params": {"window": 61},
            },
            {
                "category": "background_removal",
                "stage_id": "background_clutter",
                "method_id": "subtracting_average_2D",
                "enabled": True,
                "params": {"ntraces": 51},
            },
            {
                "category": "gain",
                "stage_id": "gain",
                "method_id": "sec_gain",
                "enabled": True,
                "params": {"gain_min": 1.0, "gain_max": 4.2, "power": 1.2},
            },
            {
                "category": "denoising",
                "stage_id": "spatial_denoise",
                "method_id": "svd_subspace",
                "enabled": True,
                "params": {"rank_start": 1, "rank_end": 20},
            },
        ],
    },
    "high_quality_uav_gpr": {
        "name": "高质量 UAV-GPR",
        "description": "面向无人机实测数据的完整高质量处理链",
        "methods": [
            {
                "category": "preprocessing",
                "stage_id": "zero_time",
                "method_id": "set_zero_time",
                "enabled": True,
                "params": {"new_zero_time": 5.0},
            },
            {
                "category": "preprocessing",
                "stage_id": "trace_correction",
                "method_id": "dc_shift",
                "enabled": True,
                "params": {"estimator": "mean", "scope": "per_trace"},
            },
            {
                "category": "preprocessing",
                "stage_id": "trace_correction",
                "method_id": "dewow",
                "enabled": True,
                "params": {"window": 61},
            },
            {
                "category": "background_removal",
                "stage_id": "trace_correction",
                "method_id": "frequency_filter_1d",
                "enabled": True,
                "params": {
                    "filter_type": "bandpass",
                    "low_freq_mhz": 20.0,
                    "high_freq_mhz": 170.0,
                    "taper_ratio": 0.08,
                },
            },
            {
                "category": "motion_compensation",
                "stage_id": "motion_compensation",
                "method_id": "motion_compensation_v2",
                "enabled": True,
                "params": {
                    "height_reference_mode": "mean",
                    "height_source": "auto",
                    "compensate_time_shift": True,
                    "compensate_amplitude": True,
                    "max_shift_samples": 0.0,
                    "max_shift_ns": 20.0,
                    "max_amplitude_scale": 2.0,
                    "resample_spacing_m": 0.0,
                    "apc_offset_x_m": 0.0,
                    "apc_offset_y_m": 0.0,
                    "apc_offset_z_m": 0.0,
                    "max_abs_tilt_deg": 20.0,
                },
            },
            {
                "category": "background_removal",
                "stage_id": "background_clutter",
                "method_id": "subtracting_average_2D",
                "enabled": True,
                "params": {"ntraces": 51},
            },
            {
                "category": "denoising",
                "stage_id": "spatial_denoise",
                "method_id": "wavelet_svd",
                "enabled": True,
                "params": {
                    "wavelet": "db4",
                    "levels": 2,
                    "threshold": 0.05,
                    "rank_start": 1,
                    "rank_end": 20,
                },
            },
            {
                "category": "migration",
                "stage_id": "velocity_model",
                "method_id": "manual_velocity_model",
                "enabled": True,
                "params": {
                    "mode": "velocity",
                    "velocity_m_per_ns": 0.10,
                    "epsilon_r": 9.0,
                    "uncertainty_fraction": 0.10,
                },
            },
            {
                "category": "migration",
                "stage_id": "geometry_depth",
                "method_id": "geometry_depth_context",
                "enabled": True,
                "params": {
                    "require_velocity_model": True,
                    "require_trace_spacing": True,
                    "require_time_window": True,
                    "require_agl": False,
                },
            },
            {
                "category": "gain",
                "stage_id": "gain",
                "method_id": "sec_gain",
                "enabled": True,
                "params": {"gain_min": 1.0, "gain_max": 4.5, "power": 1.1},
            },
            {
                "category": "migration",
                "stage_id": "migration",
                "method_id": "kirchhoff_migration",
                "enabled": False,
                "hidden": True,
                "params": {},
            },
        ],
    },
    "motion_compensation_v1": {
        "name": "运动补偿 V1",
        "description": "无人机GPR五维运动误差校正流程（确定性V1阶段）",
        "methods": [
            {
                "category": "motion_compensation",
                "method_id": "trajectory_smoothing",
                "enabled": True,
                "params": {"method": "savgol", "window_length": 21, "polyorder": 3},
            },
            {
                "category": "motion_compensation",
                "method_id": "motion_compensation_speed",
                "enabled": True,
                "params": {"spacing_m": 0.0},
            },
            {
                "category": "motion_compensation",
                "method_id": "motion_compensation_attitude",
                "enabled": True,
                "params": {
                    "apc_offset_x_m": 0.0,
                    "apc_offset_y_m": 0.0,
                    "apc_offset_z_m": 0.0,
                    "max_abs_tilt_deg": 20.0,
                },
            },
            {
                "category": "motion_compensation",
                "method_id": "motion_compensation_height",
                "enabled": True,
                "params": {
                    "reference_height_mode": "mean",
                    "compensate_amplitude": True,
                    "compensate_time_shift": True,
                    "wave_speed_m_per_ns": 0.1,
                },
            },
            {
                "category": "motion_compensation",
                "method_id": "motion_compensation_vibration",
                "enabled": True,
                "params": {
                    "smooth_window": 9,
                    "preserve_row_percentile": 94.0,
                    "preserve_mix": 0.35,
                    "background_mix": 0.02,
                    "max_restore_gain": 1.25,
                },
            },
        ],
    },
    "motion_compensation_v2": {
        "name": "运动补偿 V2",
        "description": "统一的 UAV-GPR RTK/IMU/高度计运动补偿流程",
        "methods": [
            {
                "category": "motion_compensation",
                "method_id": "motion_compensation_v2",
                "enabled": True,
                "params": {
                    "height_reference_mode": "mean",
                    "height_source": "auto",
                    "compensate_time_shift": True,
                    "compensate_amplitude": True,
                    "max_shift_samples": 0.0,
                    "max_shift_ns": 20.0,
                    "max_amplitude_scale": 2.0,
                    "resample_spacing_m": 0.0,
                    "apc_offset_x_m": 0.0,
                    "apc_offset_y_m": 0.0,
                    "apc_offset_z_m": 0.0,
                    "max_abs_tilt_deg": 20.0,
                },
            },
        ],
    },
}


# ============ UAV-GPR 实时工作流阶段定义 ============

WORKFLOW_STAGE_DEFINITIONS = [
    {
        "id": "zero_time",
        "label": "零时校正",
        "default_method": "set_zero_time",
        "candidate_methods": ["set_zero_time"],
        "warning": "",
    },
    {
        "id": "trace_correction",
        "label": "基础迹线域校正",
        "default_method": "dewow",
        "candidate_methods": [
            "dc_shift",
            "dewow",
            "frequency_filter_1d",
            "trace_qc",
        ],
        "warning": "",
    },
    {
        "id": "motion_compensation",
        "label": "UAV-GPR 采集几何校正与运动补偿",
        "default_method": "motion_compensation_v2",
        "candidate_methods": ["motion_compensation_v2"],
        "warning": "缺少 RTK/IMU/AGL 侧车数据时应跳过或仅记录风险。",
    },
    {
        "id": "background_clutter",
        "label": "背景与杂波抑制",
        "default_method": "subtracting_average_2D",
        "candidate_methods": [
            "subtracting_average_2D",
            "median_background_2D",
            "svd_bg",
            "fk_filter",
            "ccbs",
        ],
        "warning": "F-K / 方向性滤波属于可选空间滤波，需检查是否损伤目标倾角。",
    },
    {
        "id": "spatial_denoise",
        "label": "可选空间滤波与去噪增强",
        "default_method": "wavelet_svd",
        "candidate_methods": [
            "wavelet_svd",
            "wavelet_2d",
            "svd_subspace",
            "hankel_svd",
        ],
        "warning": "",
    },
    {
        "id": "velocity_model",
        "label": "速度模型建立",
        "default_method": "manual_velocity_model",
        "candidate_methods": ["manual_velocity_model"],
        "warning": "第一版仅支持手动常速度 / 介电常数。",
    },
    {
        "id": "geometry_depth",
        "label": "几何-深度校正",
        "default_method": "geometry_depth_context",
        "candidate_methods": ["geometry_depth_context"],
        "warning": "第一版做上下文校验和参数传递，不伪造成熟地形校正。",
    },
    {
        "id": "gain",
        "label": "增益补偿",
        "default_method": "sec_gain",
        "candidate_methods": [
            "sec_gain",
            "energy_decay_gain",
            "compensatingGain",
            "agcGain",
        ],
        "warning": "AGC 偏显示增强，非严格保幅；论文定量分析需谨慎使用。",
    },
    {
        "id": "migration",
        "label": "成像 / 迁移",
        "default_method": "kirchhoff_migration",
        "candidate_methods": [
            "kirchhoff_migration",
            "stolt_migration",
            "time_to_depth",
        ],
        "warning": "迁移为重计算步骤，实时模式下等待参数稳定后运行。",
    },
]

WORKFLOW_STAGE_BY_ID = {
    stage["id"]: stage for stage in WORKFLOW_STAGE_DEFINITIONS
}


# ============ 流程配置结构 ============


class WorkflowMethod:
    """单个方法的配置"""

    def __init__(
        self,
        category: str,
        method_id: str,
        enabled: bool = True,
        order: int = 0,
        params: Optional[Dict[str, Any]] = None,
        stage_id: str = "",
        hidden: bool = False,
        status: str = "pending",
        node_id: str = "",
        input_shape: Optional[tuple] = None,
        output_shape: Optional[tuple] = None,
        error_message: str = "",
        elapsed_ms: float = 0.0,
    ):
        self.category = category
        self.stage_id = stage_id
        self.method_id = method_id
        self.enabled = enabled
        self.order = order
        self.params = params or {}
        self.hidden = hidden
        self.status = status
        self.node_id = node_id or _make_node_id(method_id, order)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.error_message = error_message
        self.elapsed_ms = elapsed_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "stage_id": self.stage_id,
            "method_id": self.method_id,
            "enabled": self.enabled,
            "order": self.order,
            "params": self.params,
            "hidden": self.hidden,
            "status": self.status,
            "node_id": self.node_id,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "error_message": self.error_message,
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowMethod":
        return cls(
            category=data.get("category", ""),
            stage_id=data.get("stage_id", ""),
            method_id=data["method_id"],
            enabled=data.get("enabled", True),
            order=data.get("order", 0),
            params=data.get("params", {}),
            hidden=data.get("hidden", False),
            status=data.get("status", "pending"),
            node_id=data.get("node_id", ""),
            input_shape=data.get("input_shape"),
            output_shape=data.get("output_shape"),
            error_message=data.get("error_message", ""),
            elapsed_ms=data.get("elapsed_ms", 0.0),
        )


class WorkflowLink:
    """Canvas-level connection between two workflow node ports."""

    def __init__(
        self,
        from_node: str,
        to_node: str,
        from_port: str = "output",
        to_port: str = "input",
        kind: str = "data",
    ):
        self.from_node = str(from_node)
        self.from_port = str(from_port)
        self.to_node = str(to_node)
        self.to_port = str(to_port)
        self.kind = str(kind)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_node": self.from_node,
            "from_port": self.from_port,
            "to_node": self.to_node,
            "to_port": self.to_port,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowLink":
        return cls(
            from_node=data.get("from_node", ""),
            from_port=data.get("from_port", "output"),
            to_node=data.get("to_node", ""),
            to_port=data.get("to_port", "input"),
            kind=data.get("kind", "data"),
        )


class WorkflowConfig:
    """完整流程配置"""

    def __init__(
        self,
        name: str = "未命名流程",
        methods: Optional[List[WorkflowMethod]] = None,
        version: str = "1.0",
        template_type: str = "user",
        realtime_enabled: bool | None = None,
        canvas_links: Optional[List[WorkflowLink]] = None,
        canvas_layout: Optional[Dict[str, Any]] = None,
        _links_initialized: bool = False,
    ):
        self.version = version
        self.name = name
        self.methods = methods or []
        ensure_workflow_method_ids(self.methods)
        self.template_type = template_type
        self.realtime_enabled = (
            bool(realtime_enabled)
            if realtime_enabled is not None
            else template_type == "user"
        )
        self.created_at = datetime.now().isoformat()
        self.last_modified = datetime.now().isoformat()
        self.canvas_links = canvas_links or []
        self.canvas_layout = canvas_layout or {"nodes": {}}
        self._links_initialized = _links_initialized
        self.ensure_canvas_links()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "template_type": self.template_type,
            "realtime_enabled": self.realtime_enabled,
            "methods": [m.to_dict() for m in self.methods],
            "canvas_links": [link.to_dict() for link in self.canvas_links],
            "canvas_layout": self.canvas_layout,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "_links_initialized": self._links_initialized,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowConfig":
        config = cls(
            name=data.get("name", "未命名流程"),
            version=data.get("version", "1.0"),
            template_type=data.get("template_type", "user"),
            realtime_enabled=data.get("realtime_enabled"),
        )
        config.methods = [WorkflowMethod.from_dict(m) for m in data.get("methods", [])]
        ensure_workflow_method_ids(config.methods)
        config.canvas_links = [
            WorkflowLink.from_dict(link) for link in data.get("canvas_links", [])
        ]
        config.canvas_layout = data.get("canvas_layout", {"nodes": {}})
        # 从 data 中读取 _links_initialized；如果不存在，说明是旧模板
        # 旧模板没有 canvas_links 时，需要生成默认链接
        has_links_field = "canvas_links" in data
        config._links_initialized = data.get("_links_initialized", False)
        # 如果是旧模板（没有 _links_initialized 字段）且没有 canvas_links，则允许生成默认链接
        if not config._links_initialized and not has_links_field:
            config._links_initialized = False
        config.ensure_canvas_links()
        config.created_at = data.get("created_at", datetime.now().isoformat())
        config.last_modified = data.get("last_modified", datetime.now().isoformat())
        return config

    def ensure_canvas_links(self) -> None:
        """Ensure method node IDs are valid; create default links only if not initialized yet."""
        ensure_workflow_method_ids(self.methods)
        if not self._links_initialized and not self.canvas_links and self.methods:
            sorted_methods = sorted(self.methods, key=lambda item: item.order)
            self.canvas_links = [
                WorkflowLink(left.node_id, right.node_id)
                for left, right in zip(sorted_methods, sorted_methods[1:])
            ]
            self._links_initialized = True

    def get_enabled_methods(self) -> List[WorkflowMethod]:
        """获取启用的方法列表（按顺序排序）"""
        enabled = [m for m in self.methods if m.enabled and not m.hidden]
        return sorted(enabled, key=lambda x: x.order)

    def add_method(
        self,
        category: str,
        method_id: str,
        params: Optional[Dict] = None,
        stage_id: str = "",
    ) -> WorkflowMethod:
        """添加新方法"""
        method = WorkflowMethod(
            category=category,
            stage_id=stage_id,
            method_id=method_id,
            order=len(self.methods),
            params=params or {},
        )
        self.methods.append(method)
        ensure_workflow_method_ids(self.methods)
        self.ensure_canvas_links()
        self.last_modified = datetime.now().isoformat()
        return method

    def remove_method(self, index: int):
        """删除方法"""
        if 0 <= index < len(self.methods):
            removed_node_id = self.methods[index].node_id
            del self.methods[index]
            # 重新排序
            for i, m in enumerate(self.methods):
                m.order = i
            self.canvas_links = [
                link
                for link in self.canvas_links
                if link.from_node != removed_node_id
                and link.to_node != removed_node_id
            ]
            nodes = self.canvas_layout.setdefault("nodes", {})
            if isinstance(nodes, dict):
                nodes.pop(removed_node_id, None)
            self.last_modified = datetime.now().isoformat()

    def move_method(self, from_index: int, to_index: int):
        """移动方法位置"""
        if 0 <= from_index < len(self.methods) and 0 <= to_index < len(self.methods):
            method = self.methods.pop(from_index)
            self.methods.insert(to_index, method)
            # 重新排序
            for i, m in enumerate(self.methods):
                m.order = i
            self.last_modified = datetime.now().isoformat()

    def apply_preset(self, preset_key: str):
        """应用快速预设"""
        preset = QUICK_PRESETS.get(preset_key)
        if not preset:
            return False

        self.methods = []
        for i, method_data in enumerate(preset["methods"]):
            method = WorkflowMethod(
                category=method_data["category"],
                stage_id=method_data.get("stage_id", ""),
                method_id=method_data["method_id"],
                enabled=method_data.get("enabled", True),
                hidden=method_data.get("hidden", False),
                order=i,
                params=method_data.get("params", {}),
                node_id=method_data.get("node_id", ""),
            )
            self.methods.append(method)

        self.name = preset["name"]
        self.template_type = "system"
        self.realtime_enabled = False
        self.canvas_links = []
        self.canvas_layout = {"nodes": {}}
        self.ensure_canvas_links()
        self.last_modified = datetime.now().isoformat()
        return True

    def clear(self):
        """清空所有方法"""
        self.methods = []
        self.canvas_links = []
        self.canvas_layout = {"nodes": {}}
        self.last_modified = datetime.now().isoformat()


def _make_node_id(method_id: str, order: int) -> str:
    clean_method = "".join(ch if ch.isalnum() else "_" for ch in str(method_id))[:32]
    return f"node_{int(order):03d}_{clean_method}_{uuid.uuid4().hex[:8]}"


def ensure_workflow_method_ids(methods: List[WorkflowMethod]) -> None:
    """Ensure every workflow method has a stable canvas node id."""
    seen: set[str] = set()
    for index, method in enumerate(methods):
        if not getattr(method, "node_id", "") or method.node_id in seen:
            method.node_id = _make_node_id(method.method_id, index)
        seen.add(method.node_id)


def build_default_workflow_config(
    preset_key: str = "high_quality_uav_gpr",
    *,
    template_type: str = "system",
) -> WorkflowConfig:
    """Build a workflow config from a built-in preset."""
    config = WorkflowConfig(
        name=QUICK_PRESETS.get(preset_key, {}).get("name", "高质量 UAV-GPR"),
        template_type=template_type,
        realtime_enabled=template_type == "user",
    )
    if not config.apply_preset(preset_key):
        config.apply_preset("high_quality_uav_gpr")
    config.template_type = template_type
    config.realtime_enabled = template_type == "user"
    return config


# ============ 配置管理器 ============


class WorkflowConfigManager:
    """流程配置管理器"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # Store user workflow templates in a writable app-data directory.
            config_dir = os.path.join(get_workflow_templates_dir(), "workflow_configs")

        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

        self.last_config_file = os.path.join(config_dir, "_last_config.json")

    def save_config(
        self, config: WorkflowConfig, filename: Optional[str] = None
    ) -> str:
        """保存配置"""
        if filename is None:
            filename = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        if not filename.endswith(".json"):
            filename += ".json"

        filepath = os.path.join(self.config_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

        return filepath

    def load_config(self, filename: str) -> Optional[WorkflowConfig]:
        """加载配置"""
        if not filename.endswith(".json"):
            filename += ".json"

        filepath = os.path.join(self.config_dir, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return WorkflowConfig.from_dict(data)
        except Exception as e:
            print(f"加载配置失败: {e}")
            return None

    def save_last_config(self, config: WorkflowConfig):
        """保存上次使用的配置"""
        with open(self.last_config_file, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

    def load_last_config(self) -> Optional[WorkflowConfig]:
        """加载上次使用的配置"""
        if not os.path.exists(self.last_config_file):
            return None

        try:
            with open(self.last_config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return WorkflowConfig.from_dict(data)
        except Exception as e:
            print(f"加载上次配置失败: {e}")
            return None

    def list_configs(self) -> List[Dict[str, str]]:
        """列出所有保存的配置"""
        configs = []

        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json") and not filename.startswith("_"):
                filepath = os.path.join(self.config_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    configs.append(
                        {
                            "filename": filename,
                            "name": data.get("name", "未命名"),
                            "created_at": data.get("created_at", ""),
                            "last_modified": data.get("last_modified", ""),
                        }
                    )
                except Exception as exc:
                    logger.warning("跳过无法读取的工作流配置 %s: %s", filepath, exc)

        # 按最后修改时间排序
        configs.sort(key=lambda x: x["last_modified"], reverse=True)
        return configs

    def delete_config(self, filename: str) -> bool:
        """删除配置"""
        if not filename.endswith(".json"):
            filename += ".json"

        filepath = os.path.join(self.config_dir, filename)

        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False


# ============ 全局实例 ============

_config_manager = None


def get_config_manager() -> WorkflowConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = WorkflowConfigManager()
    return _config_manager
