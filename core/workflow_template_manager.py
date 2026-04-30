#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流程模板管理器 - 创建、编辑、运行自定义流程"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.app_paths import get_workflow_templates_dir
from core.methods_registry import PROCESSING_METHODS

logger = logging.getLogger(__name__)


class WorkflowTemplate:
    """流程模板类"""

    def __init__(
        self,
        name: str,
        description: str = "",
        methods: Optional[List[Dict]] = None,
    ):
        self.name = name
        self.description = description
        self.methods = methods or []
        self.created_at = datetime.now().isoformat()
        self.last_modified = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "methods": self.methods,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTemplate":
        template = cls(
            name=data.get("name", "未命名流程"),
            description=data.get("description", ""),
            methods=data.get("methods", []),
        )
        template.created_at = data.get("created_at", datetime.now().isoformat())
        template.last_modified = data.get("last_modified", datetime.now().isoformat())
        return template

    def add_method(self, method_id: str, params: dict):
        """添加方法到流程"""
        self.methods.append(
            {
                "method_id": method_id,
                "params": params,
                "order": len(self.methods),
            }
        )
        self.last_modified = datetime.now().isoformat()

    def remove_method(self, index: int):
        """从流程中移除方法"""
        if 0 <= index < len(self.methods):
            self.methods.pop(index)
            # 重新排序
            for i, m in enumerate(self.methods):
                m["order"] = i
            self.last_modified = datetime.now().isoformat()

    def move_method(self, from_index: int, to_index: int):
        """移动方法位置"""
        if 0 <= from_index < len(self.methods) and 0 <= to_index < len(self.methods):
            method = self.methods.pop(from_index)
            self.methods.insert(to_index, method)
            # 重新排序
            for i, m in enumerate(self.methods):
                m["order"] = i
            self.last_modified = datetime.now().isoformat()


class WorkflowTemplateManager:
    """流程模板管理器"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = get_workflow_templates_dir()

        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

        self.templates_file = os.path.join(config_dir, "templates.json")
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, WorkflowTemplate]:
        """加载模板"""
        if not os.path.exists(self.templates_file):
            return {}

        try:
            with open(self.templates_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            templates = {}
            for name, template_data in data.items():
                templates[name] = WorkflowTemplate.from_dict(template_data)

            return templates
        except Exception as e:
            logger.warning("加载模板失败: %s", e)
            return {}

    def _save_templates(self):
        """保存模板"""
        try:
            data = {
                name: template.to_dict() for name, template in self.templates.items()
            }

            with open(self.templates_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存模板失败: %s", e)

    def _generate_unique_name(self, base_name: str) -> str:
        if base_name not in self.templates:
            return base_name
        index = 2
        while f"{base_name} ({index})" in self.templates:
            index += 1
        return f"{base_name} ({index})"

    def create_template(self, name: str, description: str = "") -> WorkflowTemplate:
        """创建新模板"""
        unique_name = self._generate_unique_name(name)
        template = WorkflowTemplate(unique_name, description)
        self.templates[unique_name] = template
        self._save_templates()
        return template

    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """获取模板"""
        return self.templates.get(name)

    def get_all_templates(self) -> List[Dict]:
        """获取所有模板"""
        return [
            {
                "name": name,
                "description": template.description,
                "method_count": len(template.methods),
                "created_at": template.created_at,
                "last_modified": template.last_modified,
            }
            for name, template in self.templates.items()
        ]

    def delete_template(self, name: str):
        """删除模板"""
        if name in self.templates:
            del self.templates[name]
            self._save_templates()

    def rename_template(self, old_name: str, new_name: str):
        """重命名模板"""
        if old_name in self.templates:
            template = self.templates.pop(old_name)
            unique_name = self._generate_unique_name(new_name)
            template.name = unique_name
            self.templates[unique_name] = template
            self._save_templates()

    def duplicate_template(
        self, source_name: str, new_name: str
    ) -> Optional[WorkflowTemplate]:
        """复制模板"""
        source = self.templates.get(source_name)
        if not source:
            return None

        # 创建副本
        unique_name = self._generate_unique_name(new_name)
        new_template = WorkflowTemplate(
            name=unique_name,
            description=source.description,
            methods=[m.copy() for m in source.methods],
        )

        self.templates[unique_name] = new_template
        self._save_templates()
        return new_template

    def add_method_to_template(self, template_name: str, method_id: str, params: dict):
        """向模板添加方法"""
        template = self.templates.get(template_name)
        if template:
            template.add_method(method_id, params)
            self._save_templates()

    def remove_method_from_template(self, template_name: str, index: int):
        """从模板移除方法"""
        template = self.templates.get(template_name)
        if template:
            template.remove_method(index)
            self._save_templates()

    def get_template_methods(self, template_name: str) -> List[Dict]:
        """获取模板中的方法列表"""
        template = self.templates.get(template_name)
        if not template:
            return []

        return [
            {
                "method_id": m["method_id"],
                "method_name": PROCESSING_METHODS.get(m["method_id"], {}).get(
                    "name", m["method_id"]
                ),
                "params": m["params"],
                "order": m["order"],
            }
            for m in template.methods
        ]

    def export_template(self, name: str, filepath: str):
        """导出模板到文件"""
        template = self.templates.get(name)
        if not template:
            return

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(template.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info("模板已导出: %s", filepath)
        except Exception as e:
            logger.warning("导出模板失败: %s", e)

    def import_template(self, filepath: str) -> Optional[WorkflowTemplate]:
        """从文件导入模板"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            template = WorkflowTemplate.from_dict(data)

            template.name = self._generate_unique_name(template.name)

            self.templates[template.name] = template
            self._save_templates()
            logger.info("模板已导入: %s", filepath)
            return template
        except Exception as e:
            logger.warning("导入模板失败: %s", e)
            return None

    def get_preset_templates(self) -> List[Dict]:
        """获取预设模板"""
        return [
            {
                "name": "稳健成像",
                "description": "标准GPR数据处理流程",
                "methods": [
                    {"method_id": "set_zero_time", "params": {"new_zero_time": 5.0}},
                    {"method_id": "dewow", "params": {"window": 41}},
                    {
                        "method_id": "fk_filter",
                        "params": {"angle_low": 12, "angle_high": 55},
                    },
                    {"method_id": "subtracting_average_2D", "params": {}},
                    {
                        "method_id": "sec_gain",
                        "params": {"gain_min": 1.0, "gain_max": 4.5},
                    },
                ],
            },
            {
                "name": "高聚焦",
                "description": "完整流程，包含迁移",
                "methods": [
                    {"method_id": "set_zero_time", "params": {"new_zero_time": 5.0}},
                    {"method_id": "dewow", "params": {"window": 41}},
                    {
                        "method_id": "fk_filter",
                        "params": {"angle_low": 12, "angle_high": 55},
                    },
                    {"method_id": "svd_bg", "params": {"rank": 1}},
                    {
                        "method_id": "sec_gain",
                        "params": {"gain_min": 1.0, "gain_max": 4.5},
                    },
                    {
                        "method_id": "hankel_svd",
                        "params": {"window_length": 0, "rank": 0},
                    },
                    {"method_id": "stolt_migration", "params": {"dx": 0.05, "dt": 0.1}},
                ],
            },
        ]
