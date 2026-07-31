#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""智能处理流程工作台 - 数据结构和配置管理

定义方法分类、流程配置结构和配置管理器
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from core.storage_primitives import atomic_write_json
from typing import Dict, List, Optional, Any

LOGGER = logging.getLogger(__name__)


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
            "compensatingGain",
            "agcGain",
            "sec_gain",
            "energy_decay_gain",
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
    "artifact_suppression": {
        "id": "artifact_suppression",
        "name": "伪影抑制",
        "icon": "▧",
        "description": "周期条带、尖锐杂波等高级增强",
        "methods": ["motion_compensation_vibration", "running_average_2D"],
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
        "methods": ["stolt_migration", "kirchhoff_migration", "time_to_depth"],
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
    "high_quality_uav_gpr": {
        "name": "高质量 UAV-GPR",
        "description": "面向无人机实测数据的完整高质量处理链",
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
                "params": {"window": 61},
            },
            {
                "category": "background_removal",
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
                "method_id": "subtracting_average_2D",
                "enabled": True,
                "params": {"ntraces": 51},
            },
            {
                "category": "denoising",
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
                "category": "gain",
                "method_id": "sec_gain",
                "enabled": True,
                "params": {"gain_min": 1.0, "gain_max": 4.5, "power": 1.1},
            },
        ],
    },
    "motion_compensation_v1": {
        "name": "分项运动补偿（兼容入口）",
        "description": "保留旧 preset key，但四个用户可见原子节点已共享 UAV 运动补偿 V2 物理假设。",
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
                    "height_source": "auto",
                    "compensate_amplitude": True,
                    "compensate_time_shift": True,
                    "wave_speed_m_per_ns": 0.299792458,
                },
            },
        ],
    },
    "motion_compensation_core_v1": {
        "name": "分项运动补偿（共享 V2 核心）",
        "description": "无人机GPR四项原子运动补偿流程，共享 UAV 运动补偿 V2 物理假设。",
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
                    "height_source": "auto",
                    "compensate_amplitude": True,
                    "compensate_time_shift": True,
                    "wave_speed_m_per_ns": 0.299792458,
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
    ):
        self.category = category
        self.method_id = method_id
        self.enabled = enabled
        self.order = order
        self.params = params or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "method_id": self.method_id,
            "enabled": self.enabled,
            "order": self.order,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowMethod":
        return cls(
            category=data["category"],
            method_id=data["method_id"],
            enabled=data.get("enabled", True),
            order=data.get("order", 0),
            params=data.get("params", {}),
        )


class WorkflowConfig:
    """完整流程配置"""

    def __init__(
        self,
        name: str = "未命名流程",
        methods: Optional[List[WorkflowMethod]] = None,
        version: str = "1.0",
    ):
        self.version = version
        self.name = name
        self.methods = methods or []
        self.created_at = datetime.now().isoformat()
        self.last_modified = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "methods": [m.to_dict() for m in self.methods],
            "created_at": self.created_at,
            "last_modified": self.last_modified,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowConfig":
        config = cls(
            name=data.get("name", "未命名流程"),
            version=data.get("version", "1.0"),
        )
        config.methods = [WorkflowMethod.from_dict(m) for m in data.get("methods", [])]
        config.created_at = data.get("created_at", datetime.now().isoformat())
        config.last_modified = data.get("last_modified", datetime.now().isoformat())
        return config

    def get_enabled_methods(self) -> List[WorkflowMethod]:
        """获取启用的方法列表（按顺序排序）"""
        enabled = [m for m in self.methods if m.enabled]
        return sorted(enabled, key=lambda x: x.order)

    def add_method(
        self, category: str, method_id: str, params: Optional[Dict] = None
    ) -> WorkflowMethod:
        """添加新方法"""
        method = WorkflowMethod(
            category=category,
            method_id=method_id,
            order=len(self.methods),
            params=params or {},
        )
        self.methods.append(method)
        self.last_modified = datetime.now().isoformat()
        return method

    def remove_method(self, index: int):
        """删除方法"""
        if 0 <= index < len(self.methods):
            del self.methods[index]
            # 重新排序
            for i, m in enumerate(self.methods):
                m.order = i
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
                method_id=method_data["method_id"],
                enabled=method_data.get("enabled", True),
                order=i,
                params=method_data.get("params", {}),
            )
            self.methods.append(method)

        self.name = preset["name"]
        self.last_modified = datetime.now().isoformat()
        return True

    def clear(self):
        """清空所有方法"""
        self.methods = []
        self.last_modified = datetime.now().isoformat()


# ============ 配置管理器 ============


class WorkflowConfigManager:
    """流程配置管理器"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # 默认保存在项目目录下的 workflows/ 文件夹
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(base_dir, "workflows")

        self.config_dir = os.path.abspath(config_dir)
        os.makedirs(self.config_dir, exist_ok=True)

        self.last_config_file = os.path.join(self.config_dir, "_last_config.json")

    def _safe_config_path(self, filename: str) -> str:
        """Return a JSON config path constrained to ``self.config_dir``.

        Workflow names and imported filenames can come from UI state.  Keep all
        load/save/delete operations inside the workflow config directory and
        reject path traversal such as ``../other.json``.
        """
        raw = str(filename or "").strip()
        if not raw:
            raise ValueError("配置文件名不能为空")
        if not raw.endswith(".json"):
            raw += ".json"
        name = Path(raw).name
        if name != raw or name in {".", ".."}:
            raise ValueError(f"非法配置文件名: {filename}")
        target = (Path(self.config_dir) / name).resolve()
        root = Path(self.config_dir).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"配置路径越界: {filename}") from exc
        return str(target)

    def save_config(
        self, config: WorkflowConfig, filename: Optional[str] = None
    ) -> str:
        """保存配置"""
        if filename is None:
            filename = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self._safe_config_path(filename)

        atomic_write_json(filepath, config.to_dict())
        return filepath

    def load_config(self, filename: str) -> Optional[WorkflowConfig]:
        """加载配置"""
        try:
            filepath = self._safe_config_path(filename)
        except ValueError as exc:
            LOGGER.warning("拒绝加载非法工作流配置名 %s: %s", filename, exc)
            return None

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return WorkflowConfig.from_dict(data)
        except Exception as exc:
            LOGGER.warning("加载配置失败: %s", exc)
            return None

    def save_last_config(self, config: WorkflowConfig):
        """保存上次使用的配置"""
        atomic_write_json(self.last_config_file, config.to_dict())

    def load_last_config(self) -> Optional[WorkflowConfig]:
        """加载上次使用的配置"""
        if not os.path.exists(self.last_config_file):
            return None

        try:
            with open(self.last_config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return WorkflowConfig.from_dict(data)
        except Exception as exc:
            LOGGER.warning("加载上次配置失败: %s", exc)
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
                    LOGGER.warning("跳过无效工作流配置 %s: %s", filename, exc)

        # 按最后修改时间排序
        configs.sort(key=lambda x: x["last_modified"], reverse=True)
        return configs

    def delete_config(self, filename: str) -> bool:
        """删除配置"""
        try:
            filepath = self._safe_config_path(filename)
        except ValueError as exc:
            LOGGER.warning("拒绝删除非法工作流配置名 %s: %s", filename, exc)
            return False

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
