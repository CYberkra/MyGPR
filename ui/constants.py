# -*- coding: utf-8 -*-
"""MyGPR Qt 前端 UI 常量（唯一来源，无运行时依赖）。

对应 SPEC §2 [A1]：窗口尺寸 / 默认项 / 状态色 / 徽章色对。
颜色值逐字复刻 style_spec §1.2。

路径类常量（LOG_DIR）委托 ``core.app_paths`` 统一来源（策略迁移豁免），
避免 Windows 下 GUI 日志（~/MyGPR/logs）与 core 事件/崩溃日志
（%LOCALAPPDATA%/MyGPR/output/logs）分家。
"""
import os

from core.app_paths import get_logs_dir as _core_get_logs_dir

# ---------------------------------------------------------------- 窗口
APP_NAME = 'MyGPR 探地雷达数据处理软件'
WINDOW_WIDTH = 1450
WINDOW_HEIGHT = 850
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 700
NAV_EXPAND_WIDTH = 200
SPLASH_DURATION_MS = 600

# 右侧全局折叠面板（style_spec §2.5）
PANEL_MAX_WIDTH = 380
PANEL_MIN_HEIGHT = 400
PANEL_ANIM_DURATION_MS = 220
FOLD_BUTTON_WIDTH = 18
FOLD_BUTTON_HEIGHT = 60

# ---------------------------------------------------------------- 字体 / 间距
FONT_FAMILY = 'Microsoft YaHei'
PAGE_SPACING = 15
PAGE_MARGINS = (20, 20, 20, 20)
CARD_SPACING = 10
CARD_MARGINS = (15, 15, 15, 15)
PANEL_SPACING = 6
PANEL_MARGINS = (6, 6, 6, 6)

# ---------------------------------------------------------------- 状态色（style_spec §1.2）
COLOR_SUCCESS = '#22c55e'
COLOR_WARNING = '#f59e0b'
COLOR_ERROR = '#ef4444'
COLOR_INFO = '#3b82f6'
COLOR_DISABLED = '#9ca3af'

# 徽章配色对（文字色, 底色），逐字复刻 style_spec §1.2
BADGE_COLOR_PAIRS = {
    '未定位': ('#9ca3af', '#f3f4f6'),
    '单点定位': ('#f59e0b', '#fffbeb'),
    'RTK浮点解': ('#3b82f6', '#eff6ff'),
    'RTK固定解': ('#22c55e', '#f0fdf4'),
}

# ---------------------------------------------------------------- 日志（style_spec §1.2/§2.5）
# 日志配色派生自语义色单轨（任务 F 候选 4：消除 Bootstrap/Tailwind 双轨）。
# 深底终端上 Tailwind 语义原值明度不足，故取同色相高亮度变体；
# 色相与语义一一对应，仅明度适配深底（#2b2b2b 系背景）。
LOG_COLOR_ERROR = '#ff5c5c'    # ← COLOR_ERROR #ef4444 深底增亮
LOG_COLOR_WARNING = '#ffb84d'  # ← COLOR_WARNING #f59e0b 深底增亮
LOG_COLOR_SUCCESS = '#34d97b'  # ← COLOR_SUCCESS #22c55e 深底增亮
LOG_COLOR_INFO = '#5b9dff'     # ← COLOR_INFO #3b82f6 深底增亮

# 日志面板 QSS 三套配色（bg / fg / border）
LOG_QSS_TERMINAL = ('#2b2b2b', '#e0e0e0', '#404040')   # 初始（浅色主题下也用深底）
LOG_QSS_DARK = ('#1e1e1e', '#e0e0e0', '#333')
LOG_QSS_LIGHT = ('#f5f5f5', '#333', '#ddd')

# ---------------------------------------------------------------- 图表/可视化（任务 F 候选 4：图表色板归一）
# 测线颜色循环：matplotlib tab10（数据系列用，与语义状态色分命名空间——
# 图表色编码的是"测线身份"而非"状态"，故独立成板）。
CHART_TRACK_COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
CHART_TRACK_DEFAULT = CHART_TRACK_COLORS[0]
# 解释标注高亮（SPEC §6.6：pick/overlay 琥珀 #fbbf24）
CHART_OVERLAY_COLOR = '#fbbf24'

# ---------------------------------------------------------------- 日志文件（style_spec §5.4）
DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024   # 10MB
DEFAULT_LOG_BACKUP_COUNT = 5
LOG_DIR = _core_get_logs_dir()   # 与 core.app_paths.get_logs_dir() 统一（单源）

# ---------------------------------------------------------------- 默认项
DEFAULT_DIELECTRIC = 9.0
PREVIEW_MAX_SAMPLES = 900
PREVIEW_MAX_TRACES = 1800
MAX_WORKERS = 2
DEFAULT_PROJECT_ROOT = os.path.join(os.path.expanduser('~'), 'Documents', 'MyGPRProjects')
RECENT_PROJECTS_MAX = 10

# B-Scan 颜色映射九项（SPEC §1 / style_spec §2.9），默认 seismic
COLORMAPS = ['seismic', 'hot', 'jet', 'gray', 'viridis', 'plasma',
             'inferno', 'magma', 'cividis']
DEFAULT_COLORMAP = 'seismic'

THEME_LIGHT = '浅色主题'
THEME_DARK = '深色主题'

# ---------------------------------------------------------------- 路径
UI_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(UI_DIR, 'resources')
APP_ICON_PATH = os.path.join(RESOURCES_DIR, 'mygpr_logo.png')
SETTINGS_DIR = os.path.join(os.path.expanduser('~'), 'MyGPR', 'config')
SETTINGS_FILE = os.path.join(SETTINGS_DIR, 'settings.json')
