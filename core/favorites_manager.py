#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收藏管理器 - 保存和加载收藏的参数组"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.app_paths import get_favorites_dir
from core.methods_registry import PROCESSING_METHODS

logger = logging.getLogger(__name__)


class FavoritesManager:
    """收藏管理器"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = get_favorites_dir()

        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

        self.favorites_file = os.path.join(config_dir, "favorites.json")
        self.favorites = self._load_favorites()

    def _load_favorites(self) -> Dict[str, Any]:
        """加载收藏"""
        if not os.path.exists(self.favorites_file):
            return {"methods": {}, "last_updated": None}

        try:
            with open(self.favorites_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("加载收藏失败: %s", e)
            return {"methods": {}, "last_updated": None}

    def _save_favorites(self):
        """保存收藏"""
        try:
            with open(self.favorites_file, "w", encoding="utf-8") as f:
                json.dump(self.favorites, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存收藏失败: %s", e)

    def _generate_unique_name(self, method_id: str, base_name: str) -> str:
        """为同一方法下的收藏生成唯一名称。"""
        existing = {
            fav.get("name", "") for fav in self.favorites["methods"].get(method_id, [])
        }
        if base_name not in existing:
            return base_name
        index = 2
        while f"{base_name} ({index})" in existing:
            index += 1
        return f"{base_name} ({index})"

    def add_favorite(self, method_id: str, params: dict, name: Optional[str] = None):
        """添加收藏

        Args:
            method_id: 方法ID
            params: 参数字典
            name: 收藏名称（可选）
        """
        if method_id not in self.favorites["methods"]:
            self.favorites["methods"][method_id] = []

        # 检查是否已存在相同参数的收藏
        for fav in self.favorites["methods"][method_id]:
            if fav["params"] == params:
                # 已存在，更新名称
                if name:
                    fav["name"] = name
                self._save_favorites()
                return

        # 添加新收藏
        fav_id = f"{method_id}_{len(self.favorites['methods'][method_id])}"
        fav_name = self._generate_unique_name(
            method_id, name or f"收藏 {len(self.favorites['methods'][method_id]) + 1}"
        )
        favorite = {
            "id": fav_id,
            "name": fav_name,
            "params": params,
            "created_at": datetime.now().isoformat(),
            "used_count": 0,
        }

        self.favorites["methods"][method_id].append(favorite)
        self.favorites["last_updated"] = datetime.now().isoformat()
        self._save_favorites()

    def get_favorites(self, method_id: Optional[str] = None) -> List[Dict]:
        """获取收藏列表

        Args:
            method_id: 方法ID，如果为 None 则返回所有收藏
        """
        if method_id:
            return self.favorites["methods"].get(method_id, [])
        else:
            # 返回所有收藏
            all_favorites = []
            for method_id, favorites in self.favorites["methods"].items():
                for fav in favorites:
                    all_favorites.append({"method_id": method_id, **fav})
            return all_favorites

    def remove_favorite(self, method_id: str, fav_id: str):
        """删除收藏"""
        if method_id in self.favorites["methods"]:
            self.favorites["methods"][method_id] = [
                f for f in self.favorites["methods"][method_id] if f["id"] != fav_id
            ]

            # 如果该方法没有收藏了，删除该方法
            if not self.favorites["methods"][method_id]:
                del self.favorites["methods"][method_id]

            self.favorites["last_updated"] = datetime.now().isoformat()
            self._save_favorites()

    def mark_used(self, method_id: str, fav_id: str):
        """标记收藏被使用"""
        if method_id in self.favorites["methods"]:
            for fav in self.favorites["methods"][method_id]:
                if fav["id"] == fav_id:
                    fav["used_count"] = fav.get("used_count", 0) + 1
                    fav["last_used"] = datetime.now().isoformat()
                    self._save_favorites()
                    break

    def get_recently_used(self, limit: int = 5) -> List[Dict]:
        """获取最近使用的收藏"""
        all_favorites = self.get_favorites()

        # 按最后使用时间排序
        sorted_favorites = sorted(
            all_favorites, key=lambda x: x.get("last_used", ""), reverse=True
        )

        return sorted_favorites[:limit]

    def get_most_used(self, limit: int = 5) -> List[Dict]:
        """获取最常使用的收藏"""
        all_favorites = self.get_favorites()

        # 按使用次数排序
        sorted_favorites = sorted(
            all_favorites, key=lambda x: x.get("used_count", 0), reverse=True
        )

        return sorted_favorites[:limit]

    def clear_all(self):
        """清空所有收藏"""
        self.favorites = {"methods": {}, "last_updated": None}
        self._save_favorites()

    def export_favorites(self, filepath: str):
        """导出收藏到文件"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.favorites, f, ensure_ascii=False, indent=2)
            logger.info("收藏已导出: %s", filepath)
        except Exception as e:
            logger.warning("导出收藏失败: %s", e)

    def import_favorites(self, filepath: str):
        """从文件导入收藏"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                imported = json.load(f)

            # 合并导入的收藏
            for method_id, favorites in imported.get("methods", {}).items():
                if method_id not in self.favorites["methods"]:
                    self.favorites["methods"][method_id] = []

                for fav in favorites:
                    # 检查是否已存在
                    exists = any(
                        f["params"] == fav["params"]
                        for f in self.favorites["methods"][method_id]
                    )
                    if not exists:
                        fav = dict(fav)
                        fav["name"] = self._generate_unique_name(
                            method_id, fav.get("name", "导入收藏")
                        )
                        self.favorites["methods"][method_id].append(fav)

            self.favorites["last_updated"] = datetime.now().isoformat()
            self._save_favorites()
            logger.info("收藏已导入: %s", filepath)
        except Exception as e:
            logger.warning("导入收藏失败: %s", e)
