# -*- coding: utf-8 -*-
"""JSON 设置持久化（复刻 style_spec §5.3）。

健壮性设计：
- ``load()`` 文件缺失 → 用默认值；
- JSON 损坏 → 回退默认并把旧文件改名 ``.bak``；
- 加载后与 ``DEFAULT_SETTINGS`` 合并（新增键不丢）；
- ``save()`` 用「临时文件 ``.tmp`` + ``os.replace`` 原子替换」防写一半损坏。
"""
import json
import logging
import os

from ui import constants

logger = logging.getLogger(__name__)

# 全量默认键（MyGPR 版）
DEFAULT_SETTINGS = {
    'theme': constants.THEME_LIGHT,                     # '浅色主题' / '深色主题'
    'default_dielectric': constants.DEFAULT_DIELECTRIC,  # 9.0（导入表单默认值）
    'max_workers': constants.MAX_WORKERS,                # 2（后端并行线程数，重启生效）
    'project_root': constants.DEFAULT_PROJECT_ROOT,      # ~/Documents/MyGPRProjects
    'recent_projects': [],                               # 最多 10 条
    'processing_left_collapsed': False,                  # 处理页左栏折叠状态
    'processing_right_collapsed': False,                 # 处理页右栏折叠状态
    'log_panel_collapsed': True,                         # 全局日志面板默认折叠
    'spatial_local_dem': '',                             # 空间页本地 DEM 文件路径（'' = 在线下载）
    'auto_prefetch_basemap': True,                       # 空间页加载轨迹后自动预下载底图
}


class SettingsManager:
    """JSON 文件设置管理（非 QSettings）。构造时自动 ``load()``。"""

    def __init__(self, settings_file: str = constants.SETTINGS_FILE):
        self.settings_file = os.path.abspath(os.path.expanduser(settings_file))
        self._settings = dict(DEFAULT_SETTINGS)
        self.load()

    # ------------------------------------------------------------ 持久化
    def load(self) -> None:
        """加载设置：缺失用默认；损坏回退默认并改名 .bak；与 DEFAULT_SETTINGS 合并。"""
        if not os.path.exists(self.settings_file):
            self._settings = dict(DEFAULT_SETTINGS)
            return
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError('settings root is not a dict')
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning('设置文件损坏，回退默认值: %s (%s)', self.settings_file, e)
            try:
                os.replace(self.settings_file, self.settings_file + '.bak')
            except OSError as be:
                logger.warning('备份损坏设置文件失败: %s', be)
            self._settings = dict(DEFAULT_SETTINGS)
            return
        # 与 DEFAULT_SETTINGS 合并（新增键不丢）
        merged = dict(DEFAULT_SETTINGS)
        merged.update(data)
        merged['recent_projects'] = self._trim_recent(merged.get('recent_projects'))
        self._settings = merged

    def save(self) -> bool:
        """原子写：临时文件 .tmp + os.replace。"""
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            tmp_file = self.settings_file + '.tmp'
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, ensure_ascii=False, indent=4)
            os.replace(tmp_file, self.settings_file)
            return True
        except OSError as e:
            logger.error('保存设置失败: %s (%s)', self.settings_file, e)
            return False

    # ------------------------------------------------------------ 访问
    def get(self, key, default=None):
        return self._settings.get(key, default)

    def set(self, key, value) -> None:
        if key == 'recent_projects':
            value = self._trim_recent(value)
        self._settings[key] = value

    def get_all(self) -> dict:
        return dict(self._settings)

    def reset_to_defaults(self) -> None:
        self._settings = dict(DEFAULT_SETTINGS)

    # ------------------------------------------------------------ 最近项目
    def add_recent_project(self, path: str) -> None:
        """加入最近项目列表（去重置顶，最多 10 条）。"""
        path = os.path.abspath(os.path.expanduser(str(path)))
        recent = [p for p in self._settings.get('recent_projects', []) if p != path]
        recent.insert(0, path)
        self._settings['recent_projects'] = recent[:constants.RECENT_PROJECTS_MAX]

    @staticmethod
    def _trim_recent(value) -> list:
        if not isinstance(value, list):
            return []
        return [str(p) for p in value][:constants.RECENT_PROJECTS_MAX]
