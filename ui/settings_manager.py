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
import math
import os
from copy import deepcopy

from ui import constants

logger = logging.getLogger(__name__)

# 全量默认键（MyGPR 版）
DEFAULT_SETTINGS = {
    'theme': constants.THEME_LIGHT,                     # '浅色主题' / '深色主题'
    'default_dielectric': constants.DEFAULT_DIELECTRIC,  # 9.0（导入表单默认值）
    'max_workers': constants.MAX_WORKERS,                # 2（后端并行线程数，重启生效）
    'project_root': constants.DEFAULT_PROJECT_ROOT,      # ~/Documents/MyGPRProjects
    'recent_projects': [],                               # 最多 10 条
    'pinned_recent_projects': [],                         # 最近项目中的固定项
    'processing_left_collapsed': False,                  # 处理页左栏折叠状态
    'processing_right_collapsed': False,                 # 处理页右栏折叠状态
    'output_panel_open': False,                          # 底部输出面板默认收起（用者自开）
    'output_panel_active_tab': 'log',                    # 输出面板当前页签：log / jobs
    # B-Scan 显示比例：free 拉伸铺满（默认）/ square 数据盒正方形 / cell 数据格1:1
    'bscan_aspect_mode': 'free',                         # 用户切换后跨会话记住（见 BScanView）
    # B-Scan 轴单位：横轴 trace 道数（默认）/ distance 距离(m)；纵轴 sample 采样轴 / elevation 海拔(m)
    'bscan_x_axis': 'trace',
    'bscan_y_axis': 'sample',
    # B-Scan 预览布局已退役（2026-09-24 tab 模型：窗口数=打开的 tab 数，
    # 自动排布；旧设置文件里的 bscan_layout_mode/bscan_free_split 键
    # 读取后闲置，不再消费）
    # B-Scan 显示色阶（低/高百分位，只影响显示不影响数据）
    'bscan_contrast': 1.0,
    'bscan_p_low': 2.0,
    'bscan_p_high': 98.0,
    # B-Scan 右侧色标条显隐（隐藏后其宽度归还画布；右键菜单也可切）
    'bscan_colorbar_visible': True,
    # B-Scan 色标映射（设置页统一选；单视图微调走 B-Scan 右键色标子菜单）
    'bscan_colormap': 'seismic',
    # B-Scan 显示增益：off 关闭 / sec SEC 补偿（扩散+指数吸收）/ tvg TVG 补偿
    # （幂次曲线滑条调参，显示域不写数据）
    'bscan_gain_mode': 'off',
    # SEC 衰减补偿系数（dB/采样轴单位：深度轴为 dB/m，时间轴为 dB/ns）
    'bscan_gain_alpha': 0.2,
    # TVG 深端总增益（dB，浅端恒 0 增益）与曲线弯度幂次（1 线性，>1 集中深部）
    'bscan_tvg_db': 24.0,
    'bscan_tvg_power': 1.0,
    # B-Scan 全屏独立窗口几何（Qt saveGeometry 的 base64），空串=未存过
    'bscan_fullscreen_geometry': '',
    'spatial_basemap_source': 'gaode_img',               # 空间页底图源（与 map_tiles.DEFAULT_TILE_SOURCE 一致）
    'spatial_terrain_source': 'online',                  # 三维地形来源：online / estimated / local_dem
    'spatial_left_collapsed': False,                     # 空间页左栏折叠状态
    'spatial_right_collapsed': False,                    # 空间页右栏折叠状态
    'spatial_local_dem': '',                             # 空间页本地 DEM 文件路径（'' = 在线下载）
    'auto_prefetch_basemap': True,                       # 空间页加载轨迹后自动预下载底图
}


_NUMBER_RANGES = {
    'default_dielectric': (1.0, 81.0), 'max_workers': (1, 8),
    'bscan_contrast': (0.5, 3.0),
    'bscan_p_low': (0.0, 100.0), 'bscan_p_high': (0.0, 100.0),
    'bscan_gain_alpha': (0.0, 20.0), 'bscan_tvg_db': (0.0, 60.0),
    'bscan_tvg_power': (0.3, 3.0),
}
_CHOICES = {
    'theme': (constants.THEME_LIGHT, constants.THEME_DARK),
    'output_panel_active_tab': ('log', 'jobs'),
    'bscan_aspect_mode': ('free', 'square', 'cell'),
    'bscan_x_axis': ('trace', 'distance'), 'bscan_y_axis': ('sample', 'elevation'),
    'bscan_gain_mode': ('off', 'sec', 'tvg'), 'bscan_colormap': constants.COLORMAPS,
}


class SettingsManager:
    """JSON 文件设置管理（非 QSettings）。构造时自动 ``load()``。"""

    def __init__(self, settings_file: str | None = None):
        settings_file = (settings_file or os.environ.get('MYGPR_SETTINGS_FILE')
                         or constants.SETTINGS_FILE)
        self.settings_file = os.path.abspath(os.path.expanduser(settings_file))
        self._settings = deepcopy(DEFAULT_SETTINGS)
        self.load()

    # ------------------------------------------------------------ 持久化
    def load(self) -> None:
        """加载设置：缺失用默认；损坏回退默认并改名 .bak；与 DEFAULT_SETTINGS 合并。"""
        if not os.path.exists(self.settings_file):
            self._settings = deepcopy(DEFAULT_SETTINGS)
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
            self._settings = deepcopy(DEFAULT_SETTINGS)
            return
        # 与 DEFAULT_SETTINGS 合并（新增键不丢）
        merged = deepcopy(DEFAULT_SETTINGS)
        merged.update(data)
        for key, default in DEFAULT_SETTINGS.items():
            merged[key] = self._validated_value(key, merged[key], default)
        if not merged['bscan_p_low'] < merged['bscan_p_high']:
            for key in ('bscan_p_low', 'bscan_p_high'):
                merged[key] = DEFAULT_SETTINGS[key]
        self._settings = merged

    def save(self) -> bool:
        """使用独立隐藏临时文件原子写入，失败时保留上次有效设置。"""
        try:
            from ui.desktop_backend_facade import write_settings_json
            write_settings_json(self.settings_file, self._settings)
            return True
        except (OSError, TypeError, ValueError) as e:
            logger.error('保存设置失败: %s (%s)', self.settings_file, e)
            return False

    # ------------------------------------------------------------ 访问
    def get(self, key, default=None):
        return self._settings.get(key, default)

    def set(self, key, value) -> None:
        if key in DEFAULT_SETTINGS:
            value = self._validated_value(key, value, DEFAULT_SETTINGS[key])
        self._settings[key] = value

    def get_all(self) -> dict:
        return dict(self._settings)

    def reset_to_defaults(self) -> None:
        self._settings = deepcopy(DEFAULT_SETTINGS)

    @classmethod
    def _validated_value(cls, key, value, default):
        """恢复已知字段的类型/范围，保留未知扩展字段与其他有效偏好。"""
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            text = str(value).strip().lower()
            if text in ('true', '1', 'yes'):
                return True
            if text in ('false', '0', 'no'):
                return False
            return default
        if key in _NUMBER_RANGES:
            try:
                number = float(value)
                low, high = _NUMBER_RANGES[key]
                if isinstance(value, bool) or not math.isfinite(number) or not low <= number <= high:
                    return default
                return int(number) if isinstance(default, int) else number
            except (TypeError, ValueError, OverflowError):
                return default
        if isinstance(default, list):
            return cls._trim_recent(value)
        if not isinstance(value, str) or key in _CHOICES and value not in _CHOICES[key]:
            return default
        return value

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
        return [p for p in value if isinstance(p, str) and p.strip()][:constants.RECENT_PROJECTS_MAX]
