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
# 顶部横排页签条（Pivot，替代竖导航；竖导航已退役，NAV_EXPAND_WIDTH 随之移除）
NAV_TOP_BAR_HEIGHT = 36
SPLASH_DURATION_MS = 600

# 右侧全局折叠面板（style_spec §2.5）
PANEL_MAX_WIDTH = 380
PANEL_MIN_HEIGHT = 400
PANEL_ANIM_DURATION_MS = 220
FOLD_BUTTON_WIDTH = 18
FOLD_BUTTON_HEIGHT = 60

# 左侧常驻测线树面板（主窗口 widgetLayout 内容区首位插入，顶部页签条之下全高）
LINE_TREE_PANEL_WIDTH = 232
LINE_TREE_HEADER_HEIGHT = 44

# 底部输出面板（OutputPanel：日志/任务，横贯页面区下方）
OUTPUT_PANEL_HEIGHT = 160
OUTPUT_PANEL_HEADER_HEIGHT = 34

# DockPanel 统一细条宽度（左右坞共用；LINE_TREE_COLLAPSED_WIDTH 保留别名）
DOCK_COLLAPSED_WIDTH = 18
LINE_TREE_COLLAPSED_WIDTH = DOCK_COLLAPSED_WIDTH

# ---------------------------------------------------------------- 字体 / 间距
FONT_FAMILY = 'Microsoft YaHei'
# 字号层级（原来 9/10/12 散落在 page_scaffold / separators / project_page 等处
# 各自字面量，收敛为三级：辅助说明 < 正文/标签 < 分区标题）。
FONT_SIZE_CAPTION = 9    # 辅助说明、图表轴刻度
FONT_SIZE_BODY = 10      # 正文与表单标签（全局默认，见 app_qt.py）
FONT_SIZE_SECTION = 12   # 卡片/分组标题（配 Bold）
PAGE_SPACING = 15
PAGE_MARGINS = (20, 20, 20, 20)
CARD_SPACING = 10
CARD_MARGINS = (15, 15, 15, 15)
PANEL_SPACING = 6
PANEL_MARGINS = (6, 6, 6, 6)

# ---------------------------------------------------------------- 页面骨架（布局统一轮）
# 侧栏宽度两档：工具栏（方法库/测线列表/标注工具）与表单栏（参数/导入/项目信息）。
# 历史上 320/340/400/460 各页各值，统一后一律经 CollapsiblePanel 可折叠。
SIDE_TOOL_WIDTH = 320
SIDE_FORM_WIDTH = 360
# 单列表单页（设置/交付）限宽居中列：宽屏下卡片不再全宽拉满留右侧空带
FORM_COLUMN_MAX_WIDTH = 760
# 预览区（B-Scan/地图等主视图）最小高度统一档；解释页主画布例外（工作区，更高）
PREVIEW_MIN_HEIGHT = 300

# ---------------------------------------------------------------- 状态色（style_spec §1.2）
# 浅色值 + 深色变体（(浅色, 深色) 成对）。深色变体沿用 LOG_COLOR_* 的
# "同色相高亮度"原则（#2b2b2b 系深底对比度）；新代码须经
# ui.theme_helpers.status_color()/badge_colors() 按主题查表，
# 以下裸常量仅作兼容别名保留。
STATUS_COLORS = {
    'success': ('#22c55e', '#34d97b'),
    'warning': ('#f59e0b', '#ffb84d'),
    'error': ('#ef4444', '#ff5c5c'),
    'info': ('#3b82f6', '#5b9dff'),
    'disabled': ('#9ca3af', '#9ca3af'),
}

COLOR_SUCCESS = STATUS_COLORS['success'][0]
COLOR_WARNING = STATUS_COLORS['warning'][0]
COLOR_ERROR = STATUS_COLORS['error'][0]
COLOR_INFO = STATUS_COLORS['info'][0]
COLOR_DISABLED = STATUS_COLORS['disabled'][0]

# 徽章配色对（文字色, 底色）：浅色 = 彩字淡底（style_spec §1.2 逐字值）；
# 深色 = 白字彩底（参照任务中心徽章的双主题安全做法，淡底在深底上刺眼）。
BADGE_COLOR_SETS = {
    'success': (('#22c55e', '#f0fdf4'), ('#ffffff', '#15803d')),
    'warning': (('#f59e0b', '#fffbeb'), ('#ffffff', '#b45309')),
    'info': (('#3b82f6', '#eff6ff'), ('#ffffff', '#1d4ed8')),
    'error': (('#ef4444', '#fef2f2'), ('#ffffff', '#b91c1c')),
    'neutral': (('#9ca3af', '#f3f4f6'), ('#e5e7eb', '#4b5563')),
}

# RTK 定位状态 → 徽章语义键
BADGE_STATUS_KEYS = {
    '未定位': 'neutral',
    '单点定位': 'warning',
    'RTK浮点解': 'info',
    'RTK固定解': 'success',
}

# 兼容别名：逐字保留 style_spec §1.2 浅色徽章配色对（按中文状态键）。
BADGE_COLOR_PAIRS = {
    status: BADGE_COLOR_SETS[key][0]
    for status, key in BADGE_STATUS_KEYS.items()
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

# pyqtgraph 轴刻度与网格（图表观感统一单源）
# 刻度字号比正文 10pt 略小是图表惯例；负值 tickLength 让刻度朝内、
# 配合 stopAxisAtTick 使轴线止于首尾刻度，不悬空。
CHART_TICK_FONT_SIZE = FONT_SIZE_CAPTION
CHART_TICK_LENGTH = -4
# 网格仅供曲线类视图（AScan / 高程剖面）开启；图像类（B-Scan / 深度切片）
# 套网格会盖住数据，由 style_plot_item(grid=) 控制。
CHART_GRID_ALPHA_LIGHT = 0.28
CHART_GRID_ALPHA_DARK = 0.18    # 深底网格更淡，避免抢过数据

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
