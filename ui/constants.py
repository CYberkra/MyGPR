# -*- coding: utf-8 -*-
"""MyGPR Qt 前端 UI 常量（唯一来源，无运行时依赖）。

对应 SPEC §2 [A1]：窗口尺寸 / 默认项 / 状态色 / 徽章色对。
颜色值逐字复刻 style_spec §1.2。
"""
import os

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
LOG_COLOR_ERROR = '#dc3545'
LOG_COLOR_WARNING = '#ffc107'
LOG_COLOR_SUCCESS = '#28a745'
LOG_COLOR_INFO = '#17a2b8'

# 日志面板 QSS 三套配色（bg / fg / border）
LOG_QSS_TERMINAL = ('#2b2b2b', '#e0e0e0', '#404040')   # 初始（浅色主题下也用深底）
LOG_QSS_DARK = ('#1e1e1e', '#e0e0e0', '#333')
LOG_QSS_LIGHT = ('#f5f5f5', '#333', '#ddd')

# ---------------------------------------------------------------- 日志文件（style_spec §5.4）
DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024   # 10MB
DEFAULT_LOG_BACKUP_COUNT = 5
LOG_DIR = os.path.join(os.path.expanduser('~'), 'MyGPR', 'logs')

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
