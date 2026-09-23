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
    'output_panel_open': False,                          # 底部输出面板默认收起（用者自开）
    'output_panel_active_tab': 'log',                    # 输出面板当前页签：log / jobs
    # B-Scan 显示比例：free 拉伸铺满（默认）/ square 数据盒正方形 / cell 数据格1:1
    'bscan_aspect_mode': 'free',                         # 用户切换后跨会话记住（见 BScanView）
    # B-Scan 轴单位：横轴 trace 道数（默认）/ distance 距离(m)；纵轴 sample 采样轴 / elevation 海拔(m)
    'bscan_x_axis': 'trace',
    'bscan_y_axis': 'sample',
    # B-Scan 预览布局（BScanContainer）：auto 面板数自动跟随数据（默认）/
    # single 单视图 / dual 双视图对比 / quad 四宫格 / free 自由分屏（占比可调）
    'bscan_layout_mode': 'auto',
    # B-Scan 自由分屏占比（千分比文本 '750,250'），空串=未存过
    'bscan_free_split': '',
    # B-Scan 显示色阶（低/高百分位，只影响显示不影响数据）
    'bscan_p_low': 2.0,
    'bscan_p_high': 98.0,
    # B-Scan 右侧色标条显隐（隐藏后其宽度归还画布；右键菜单也可切）
    'bscan_colorbar_visible': True,
    # B-Scan 色标映射（设置页统一选；单视图微调走 B-Scan 右键色标子菜单）
    'bscan_colormap': 'seismic',
    # B-Scan 全屏独立窗口几何（Qt saveGeometry 的 base64），空串=未存过
    'bscan_fullscreen_geometry': '',
    'spatial_basemap_source': 'gaode_img',               # 空间页底图源（与 map_tiles.DEFAULT_TILE_SOURCE 一致）
    'spatial_terrain_source': 'online',                  # 三维地形来源：online / estimated / local_dem
    'spatial_left_collapsed': False,                     # 空间页左栏折叠状态
    'spatial_right_collapsed': False,                    # 空间页右栏折叠状态
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
