# -*- coding: utf-8 -*-
"""SpatialPage — 空间信息页（在线地图底图 + 高程剖面 + 三维轨迹）。

三栏 QHBoxLayout（模式照抄 processing_page）：
- 左栏 ScrollArea 固定 320px（可折叠）：卡片"测线"（多选勾选列表 + 颜色块）、
  卡片"底图"（瓦图源 ComboBox + 预下载按钮 + 进度）、卡片"投影信息"
- 中栏 stretch：SegmentedWidget 切换"平面地图 / 高程剖面 / 三维视图"
  + QStackedWidget（MapView / 高程剖面 PlotWidget / Trajectory3DView）
- 右栏 ScrollArea 固定 340px（可折叠）：卡片"测线详情" + "设为当前测线"

页面纯展示 + 发信号，不直接调 controller/backend。
"""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QListWidget,
                             QListWidgetItem, QStackedWidget, QVBoxLayout,
                             QWidget)
from qfluentwidgets import (CaptionLabel, CardWidget, ComboBox, DoubleSpinBox,
                            PrimaryPushButton, PushButton, ScrollArea,
                            SegmentedWidget, SubtitleLabel, SwitchButton)
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.settings_manager import SettingsManager
from ui.widgets.collapsible_panel import CollapsiblePanel
from ui.widgets.local_dem import load_xyz_grid
from ui.widgets.map_tiles import BASEMAP_LAYERS, DEFAULT_TILE_SOURCE
from ui.widgets.map_view import MapView
from ui.widgets.trajectory_3d_view import Trajectory3DView

# 中栏分段（SegmentedWidget routeKey）
_SEG_MAP = 'planMap'
_SEG_PROFILE = 'elevationProfile'
_SEG_3D = 'trajectory3d'

# 测线颜色循环（matplotlib tab10）
_TRACK_COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')


def _page_title(text: str) -> SubtitleLabel:
    """页面标题：SubtitleLabel 微软雅黑 12pt Bold 居中（SPEC §1）。"""
    label = SubtitleLabel(text)
    label.setFont(QFont(constants.FONT_FAMILY, 12, QFont.Weight.Bold))
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


def _card_title(text: str) -> SubtitleLabel:
    """卡片标题：SubtitleLabel 微软雅黑 10pt Bold（SPEC §1）。"""
    label = SubtitleLabel(text)
    label.setFont(QFont(constants.FONT_FAMILY, 10, QFont.Weight.Bold))
    return label


def _make_card(title: str) -> tuple:
    """卡片范式：CardWidget + QVBoxLayout，首行卡片标题。返回 (card, layout)。"""
    card = CardWidget()
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*constants.CARD_MARGINS)
    layout.setSpacing(constants.CARD_SPACING)
    layout.addWidget(_card_title(title))
    return card, layout


def _make_scroll_column(width: int) -> tuple:
    """固定宽滚动栏：ScrollArea(固定 width) + 内容 widget + QVBoxLayout。
    返回 (scroll_area, content_layout)。"""
    scroll = ScrollArea()
    scroll.setFixedWidth(width)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setStyleSheet(
        'QScrollArea { background-color: transparent; border: none; }')
    content = QWidget(scroll)
    content.setFixedWidth(width - 16)
    content.setObjectName('pageScrollContent')
    content.setStyleSheet(
        'QWidget#pageScrollContent { background-color: transparent; }')
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(constants.PAGE_SPACING)
    scroll.setWidget(content)
    return scroll, layout


def _color_icon(hex_color: str) -> QIcon:
    """12×12 纯色块图标（测线列表颜色标识）。"""
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(hex_color))
    return QIcon(pixmap)


class ElevationProfileView(pg.PlotWidget):
    """高程剖面：选中测线的里程-高程曲线（里程由相邻点距离累积）。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._plot_item = self.getPlotItem()
        self._plot_item.setLabel('bottom', '里程', units='m')
        self._plot_item.setLabel('left', '高程', units='m')
        self._plot_item.showGrid(x=True, y=True, alpha=0.3)
        from qfluentwidgets import isDarkTheme
        self.apply_theme(isDarkTheme())

    def set_tracks(self, tracks, colors: dict) -> None:
        """重绘选中测线的里程-高程曲线。"""
        self._plot_item.clear()
        legend = self._plot_item.legend
        if legend is not None:
            legend.clear()
        else:
            legend = self._plot_item.addLegend(
                offset=(8, 8),
                labelTextColor='w' if self._dark else 'k')
        colors = dict(colors or {})
        for track in tracks or []:
            points = list(getattr(track, 'points', ()) or ())
            if len(points) < 2:
                continue
            xs = np.asarray([float(getattr(p, 'x', 0.0)) for p in points])
            ys = np.asarray([float(getattr(p, 'y', 0.0)) for p in points])
            zs = np.asarray([float(getattr(p, 'elevation_m', 0.0)) for p in points])
            finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
            if np.count_nonzero(finite) < 2:
                continue
            steps = np.hypot(np.diff(xs[finite]), np.diff(ys[finite]))
            mileage = np.concatenate(([0.0], np.cumsum(steps)))
            line_id = str(getattr(track, 'line_id', '') or '')
            name = str(getattr(track, 'name', '') or line_id)
            pen = pg.mkPen(QColor(colors.get(line_id, '#1f77b4')), width=2)
            self._plot_item.plot(mileage, zs[finite], pen=pen, name=name)

    def apply_theme(self, dark: bool) -> None:
        """深色 bg 'k'/文字 'w'；浅色 bg 'w'/文字 'k'；轴 pen/textPen/标签/图例同步。"""
        self._dark = bool(dark)
        bg = 'k' if dark else 'w'
        fg = 'w' if dark else 'k'
        self.setBackground(bg)
        # 不能用 QColor(fg)：Qt 颜色名不含 'w'/'k'，非法色会变黑导致深色下轴字不可见
        pen = pg.mkPen(fg)
        for name in ('bottom', 'left'):
            axis = self._plot_item.getAxis(name)
            axis.setPen(pen)
            axis.setTextPen(pen)
            # 轴标题（里程/高程）是独立 label，不随 textPen 变色，需显式同步
            axis.setLabel(text=axis.labelText, color=fg)
        # 已有图例的条目文字颜色不随主题更新，逐条同步
        legend = self._plot_item.legend
        if legend is not None:
            for _sample, label in legend.items:
                label.setText(label.text, color=fg)


class SpatialPage(QWidget):
    """空间信息页面。"""

    current_line_requested = pyqtSignal(str)    # 设为当前测线（line_id）
    basemap_prefetch_requested = pyqtSignal()   # 预下载当前区域（页面内部已处理，供外部观测）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks = []            # 全量 SpatialTrack
        self._tracks_by_id = {}
        self._lines_by_id = {}       # 测线记录（rtk_status 等）
        self._colors = {}            # line_id -> '#rrggbb'
        self._restoring_basemap = False
        self._last_auto_prefetch_key = None   # 自动预下载去重（包围盒+瓦图源）
        self._auto_prefetch_enabled = True     # 默认启用（受设置页开关控制，P2-4）
        self._dem_base_text = ''              # 本地 DEM 标签基础文本（覆盖提示拼接用）
        self._terrain_mode = 'online'         # 三维地形来源：online / estimated / local_dem
        self._restoring_terrain = False       # 恢复/程序化设置地形来源下拉时屏蔽信号

        self._build_ui()
        self._connect_internal()
        self._restore_state()

    # ============================================================ 面板状态
    def panel_states(self) -> dict:
        return {
            'left': self._left_panel.is_collapsed(),
            'right': self._right_panel.is_collapsed(),
        }

    def set_panel_collapsed(self, *, left: bool = None, right: bool = None,
                            animate: bool = True) -> None:
        if left is not None:
            self._left_panel.set_collapsed(bool(left), animate=animate)
        if right is not None:
            self._right_panel.set_collapsed(bool(right), animate=animate)

    def _restore_state(self) -> None:
        """恢复折叠状态 + 底图源选择。"""
        sm = SettingsManager()
        self._left_panel.set_collapsed(
            bool(sm.get('spatial_left_collapsed', False)), animate=False)
        self._right_panel.set_collapsed(
            bool(sm.get('spatial_right_collapsed', False)), animate=False)
        source = str(sm.get('spatial_basemap_source', DEFAULT_TILE_SOURCE))
        if source not in BASEMAP_LAYERS:
            source = DEFAULT_TILE_SOURCE
        self._restoring_basemap = True
        try:
            index = self._basemap_combo.findData(source)
            if index >= 0:
                self._basemap_combo.setCurrentIndex(index)
            self._map_view.set_source(source)
        finally:
            self._restoring_basemap = False
        self._auto_load_dem(sm)
        mode = str(sm.get('spatial_terrain_source', 'online') or 'online')
        if mode not in ('online', 'estimated', 'local_dem'):
            mode = 'online'
        if mode == 'local_dem' and not self._3d_dem_clear_btn.isEnabled():
            mode = 'online'   # 上次导入的 DEM 文件已丢失，回退在线下载
        self._set_terrain_mode(mode, persist=False)

    def _set_terrain_mode(self, mode: str, *, persist: bool = True) -> None:
        """设置三维地形来源（更新下拉 + 三维视图 + 可选持久化）。"""
        self._terrain_mode = mode
        self._restoring_terrain = True
        try:
            index = self._3d_terrain_combo.findData(mode)
            if index >= 0:
                self._3d_terrain_combo.setCurrentIndex(index)
        finally:
            self._restoring_terrain = False
        self._3d_view.set_terrain_source(mode)
        if persist:
            sm = SettingsManager()
            sm.set('spatial_terrain_source', mode)
            sm.save()

    def _auto_load_dem(self, sm: SettingsManager) -> None:
        """启动自动加载上次导入的本地 DEM（默认在线下载，无需任何操作）。

        文件已被移动/删除或解析失败时清除记录，静默回退在线下载。
        """
        path = str(sm.get('spatial_local_dem', '') or '')
        if not path:
            return
        dem = None
        if os.path.isfile(path):
            try:
                dem = load_xyz_grid(path)
            except (OSError, ValueError):
                dem = None
        if dem is None:
            sm.set('spatial_local_dem', '')
            sm.save()
            return
        self._apply_local_dem(dem, path)

    def _apply_local_dem(self, dem: dict, path: str) -> None:
        """应用本地 DEM 到三维视图并更新卡片显示。"""
        self._3d_view.set_local_dem(dem)
        rows, cols = dem['elev'].shape
        self._dem_base_text = f'{os.path.basename(path)}（{cols}×{rows}）'
        self._3d_dem_label.setText(self._dem_base_text)
        self._3d_dem_label.setToolTip(path)
        self._3d_dem_clear_btn.setEnabled(True)

    def _on_dem_notice(self, text: str) -> None:
        """三维视图的地形提示（DEM 覆盖 / 估算回退等）：拼到标签基础文本后。"""
        base = self._dem_base_text or '未导入本地 DEM'
        self._3d_dem_label.setText(f'{base}；{text}' if text else base)

    def _save_panel_state(self) -> None:
        sm = SettingsManager()
        sm.set('spatial_left_collapsed', self._left_panel.is_collapsed())
        sm.set('spatial_right_collapsed', self._right_panel.is_collapsed())
        sm.save()

    # ============================================================ UI 构建
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)
        root.addWidget(_page_title('空间信息'))

        columns = QHBoxLayout()
        columns.setSpacing(constants.PAGE_SPACING)
        root.addLayout(columns, 1)

        # ---------------- 左栏（展开 320px，可折叠）
        left_scroll, left_layout = _make_scroll_column(320)
        left_panel = CollapsiblePanel(
            'left', expand_width=320, collapse_width=40, parent=self)
        left_panel.set_content_widget(left_scroll)
        columns.addWidget(left_panel)
        self._left_panel = left_panel

        lines_card, lines_layout = _make_card('测线')
        self._line_list = QListWidget(lines_card)
        self._line_list.setMinimumHeight(180)
        lines_layout.addWidget(self._line_list, 1)
        left_layout.addWidget(lines_card, 1)

        basemap_card, basemap_layout = _make_card('底图')
        source_row = QHBoxLayout()
        source_row.setSpacing(constants.CARD_SPACING)
        source_label = CaptionLabel('来源:', basemap_card)
        source_label.setMinimumWidth(40)
        source_row.addWidget(source_label)
        self._basemap_combo = ComboBox(basemap_card)
        for key, (display, _base, _overlay) in BASEMAP_LAYERS.items():
            self._basemap_combo.addItem(display, userData=key)
        source_row.addWidget(self._basemap_combo, 1)
        basemap_layout.addLayout(source_row)
        self._prefetch_btn = PushButton('预下载测线区域', basemap_card, FIF.DOWNLOAD)
        basemap_layout.addWidget(self._prefetch_btn)
        self._prefetch_label = CaptionLabel('', basemap_card)
        self._prefetch_label.setWordWrap(True)
        basemap_layout.addWidget(self._prefetch_label)
        left_layout.addWidget(basemap_card)

        crs_card, crs_layout = _make_card('投影信息')
        self._crs_label = CaptionLabel('暂无轨迹数据', crs_card)
        self._crs_label.setWordWrap(True)
        crs_layout.addWidget(self._crs_label)
        left_layout.addWidget(crs_card)

        view3d_card, view3d_layout = _make_card('三维显示')
        exag_row = QHBoxLayout()
        exag_row.setSpacing(constants.CARD_SPACING)
        exag_label = CaptionLabel('垂直夸张:', view3d_card)
        exag_label.setMinimumWidth(60)
        exag_row.addWidget(exag_label)
        self._3d_exag_spin = DoubleSpinBox(view3d_card)
        self._3d_exag_spin.setRange(0.5, 5.0)
        self._3d_exag_spin.setSingleStep(0.5)
        self._3d_exag_spin.setValue(1.0)
        self._3d_exag_spin.setSuffix(' x')
        exag_row.addWidget(self._3d_exag_spin, 1)
        view3d_layout.addLayout(exag_row)
        drape_row = QHBoxLayout()
        drape_row.setSpacing(constants.CARD_SPACING)
        drape_label = CaptionLabel('测线贴地:', view3d_card)
        drape_label.setMinimumWidth(60)
        drape_label.setToolTip('开：测线高程从地形采样（抑制 RTK 高程噪声）；关：原始 GPS 高程')
        drape_row.addWidget(drape_label)
        self._3d_drape_switch = SwitchButton(view3d_card)
        drape_row.addWidget(self._3d_drape_switch, 1)
        view3d_layout.addLayout(drape_row)
        imagery_row = QHBoxLayout()
        imagery_row.setSpacing(constants.CARD_SPACING)
        imagery_label = CaptionLabel('影像贴图:', view3d_card)
        imagery_label.setMinimumWidth(60)
        imagery_label.setToolTip('开：地形表面贴卫星影像；关：高程色表')
        imagery_row.addWidget(imagery_label)
        self._3d_imagery_switch = SwitchButton(view3d_card)
        self._3d_imagery_switch.setChecked(True)
        imagery_row.addWidget(self._3d_imagery_switch, 1)
        view3d_layout.addLayout(imagery_row)
        terrain_row = QHBoxLayout()
        terrain_row.setSpacing(constants.CARD_SPACING)
        terrain_label = CaptionLabel('地形来源:', view3d_card)
        terrain_label.setMinimumWidth(60)
        terrain_label.setToolTip(
            '在线下载：联网获取全球高程瓦片（默认）；\n'
            '测线数据估算：地表高程 = 轨迹海拔 − 离地高度，无需联网，仅为近似；\n'
            '本地 DEM：使用导入的 Global Mapper 等 XYZ 格网')
        terrain_row.addWidget(terrain_label)
        self._3d_terrain_combo = ComboBox(view3d_card)
        self._3d_terrain_combo.addItem('在线下载', userData='online')
        self._3d_terrain_combo.addItem('测线数据估算', userData='estimated')
        self._3d_terrain_combo.addItem('本地 DEM', userData='local_dem')
        terrain_row.addWidget(self._3d_terrain_combo, 1)
        view3d_layout.addLayout(terrain_row)
        dem_row = QHBoxLayout()
        dem_row.setSpacing(constants.CARD_SPACING)
        dem_label = CaptionLabel('本地 DEM:', view3d_card)
        dem_label.setMinimumWidth(60)
        dem_label.setToolTip(
            '导入 Global Mapper 等导出的 WGS84 经纬度 XYZ 格网，\n'
            '三维地形优先使用本地数据，不再在线下载高程')
        dem_row.addWidget(dem_label)
        self._3d_dem_btn = PushButton('导入…', view3d_card, FIF.FOLDER)
        dem_row.addWidget(self._3d_dem_btn)
        self._3d_dem_clear_btn = PushButton('清除', view3d_card, FIF.CLOSE)
        self._3d_dem_clear_btn.setEnabled(False)
        dem_row.addWidget(self._3d_dem_clear_btn)
        dem_row.addStretch(1)
        view3d_layout.addLayout(dem_row)
        self._3d_dem_label = CaptionLabel('未导入（在线下载高程）', view3d_card)
        self._3d_dem_label.setWordWrap(True)
        view3d_layout.addWidget(self._3d_dem_label)
        left_layout.addWidget(view3d_card)
        left_layout.addStretch(1)

        # ---------------- 中栏（stretch）
        middle = QWidget(self)
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(constants.PAGE_SPACING)
        columns.addWidget(middle, 1)

        view_card, view_layout = _make_card('空间视图')
        seg_row = QHBoxLayout()
        seg_row.setSpacing(constants.CARD_SPACING)
        self._view_segment = SegmentedWidget(view_card)
        self._view_segment.addItem(
            _SEG_MAP, '平面地图', onClick=lambda: self._switch_view(_SEG_MAP))
        self._view_segment.addItem(
            _SEG_PROFILE, '高程剖面', onClick=lambda: self._switch_view(_SEG_PROFILE))
        self._view_segment.addItem(
            _SEG_3D, '三维视图', onClick=lambda: self._switch_view(_SEG_3D))
        self._view_segment.setCurrentItem(_SEG_MAP)
        seg_row.addWidget(self._view_segment)
        seg_row.addStretch(1)
        view_layout.addLayout(seg_row)

        self._view_stack = QStackedWidget(view_card)
        self._map_view = MapView(view_card)
        self._map_view.setMinimumHeight(300)
        self._profile_view = ElevationProfileView(view_card)
        self._profile_view.setMinimumHeight(300)
        self._3d_view = Trajectory3DView(view_card)
        self._3d_view.setMinimumHeight(300)
        self._view_stack.addWidget(self._map_view)
        self._view_stack.addWidget(self._profile_view)
        self._view_stack.addWidget(self._3d_view)
        view_layout.addWidget(self._view_stack, 1)
        middle_layout.addWidget(view_card, 1)

        # ---------------- 右栏（展开 340px，可折叠）
        right_scroll, right_layout = _make_scroll_column(340)
        right_panel = CollapsiblePanel(
            'right', expand_width=340, collapse_width=40, parent=self)
        right_panel.set_content_widget(right_scroll)
        columns.addWidget(right_panel)
        self._right_panel = right_panel

        detail_card, detail_layout = _make_card('测线详情')
        self._detail_labels = {}
        for key, title in (('name', '名称'), ('traces', '道数'),
                           ('elevation', '高程范围'), ('rtk', 'RTK状态'),
                           ('crs', '坐标系')):
            row = QHBoxLayout()
            row.setSpacing(constants.CARD_SPACING)
            title_label = CaptionLabel(f'{title}:', detail_card)
            title_label.setMinimumWidth(70)
            row.addWidget(title_label)
            value_label = CaptionLabel('--', detail_card)
            value_label.setWordWrap(True)
            row.addWidget(value_label, 1)
            detail_layout.addLayout(row)
            self._detail_labels[key] = value_label
        self._set_current_btn = PrimaryPushButton(
            '设为当前测线', detail_card, FIF.ACCEPT)
        self._set_current_btn.setEnabled(False)
        detail_layout.addWidget(self._set_current_btn)
        right_layout.addWidget(detail_card)
        right_layout.addStretch(1)

    # ============================================================ 内部接线
    def _connect_internal(self) -> None:
        self._line_list.itemChanged.connect(self._on_line_check_changed)
        self._line_list.currentItemChanged.connect(self._on_line_selected)
        self._basemap_combo.currentIndexChanged.connect(self._on_basemap_changed)
        self._prefetch_btn.clicked.connect(self._on_prefetch_clicked)
        self._map_view.prefetch_progress.connect(self._on_prefetch_progress)
        self._set_current_btn.clicked.connect(self._on_set_current_clicked)
        self._3d_exag_spin.valueChanged.connect(
            self._3d_view.set_vertical_exaggeration)
        self._3d_drape_switch.checkedChanged.connect(
            self._3d_view.set_track_drape)
        self._3d_imagery_switch.checkedChanged.connect(
            self._3d_view.set_imagery_enabled)
        self._3d_dem_btn.clicked.connect(self._on_import_dem_clicked)
        self._3d_dem_clear_btn.clicked.connect(self._on_clear_dem_clicked)
        self._3d_terrain_combo.currentIndexChanged.connect(
            self._on_terrain_source_changed)
        self._3d_view.local_dem_notice.connect(self._on_dem_notice)
        self._left_panel.sig_collapsed.connect(self._save_panel_state)
        self._right_panel.sig_collapsed.connect(self._save_panel_state)

    # ============================================================ 公共接口（供主窗口接线）
    def set_tracks(self, tracks: list) -> None:
        """空间轨迹列表（SpatialTrack，鸭子类型取属性）。"""
        self._tracks = list(tracks or [])
        self._tracks_by_id = {}
        self._colors = {}
        for index, track in enumerate(self._tracks):
            line_id = str(getattr(track, 'line_id', '') or '')
            if not line_id:
                continue
            self._tracks_by_id[line_id] = track
            self._colors[line_id] = _TRACK_COLORS[index % len(_TRACK_COLORS)]

        # 重建勾选列表（默认全选，保持已有勾选状态）
        previous_checked = {}
        for row in range(self._line_list.count()):
            item = self._line_list.item(row)
            previous_checked[str(item.data(Qt.ItemDataRole.UserRole) or '')] = (
                item.checkState() == Qt.CheckState.Checked)
        self._line_list.blockSignals(True)
        self._line_list.clear()
        for track in self._tracks:
            line_id = str(getattr(track, 'line_id', '') or '')
            if not line_id:
                continue
            name = str(getattr(track, 'name', '') or line_id)
            item = QListWidgetItem(_color_icon(self._colors[line_id]), name)
            item.setData(Qt.ItemDataRole.UserRole, line_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = previous_checked.get(line_id, True)
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self._line_list.addItem(item)
        self._line_list.blockSignals(False)

        self._refresh_views()
        self._refresh_crs_card()
        self._refresh_detail()
        # 轨迹就绪后自动预下载测线所在地理区域（延迟执行，等视图 fit 完成）
        QTimer.singleShot(800, self._auto_prefetch_tracks)

    def set_lines(self, lines: list) -> None:
        """测线记录列表（取 line_id/name/rtk_status 等，鸭子类型）。"""
        self._lines_by_id = {}
        for line in lines or []:
            line_id = str(getattr(line, 'line_id', '') or '')
            if line_id:
                self._lines_by_id[line_id] = line
        self._refresh_detail()

    def apply_theme(self, dark: bool) -> None:
        """主题切换转发：地图 / 剖面 / 三维视图。"""
        self._map_view.apply_theme(dark)
        self._profile_view.apply_theme(dark)
        self._3d_view.apply_theme(dark)

    # ============================================================ 内部逻辑
    def _selected_line_id(self) -> str:
        item = self._line_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole) or '') if item else ''

    def _checked_tracks(self) -> list:
        """勾选中的测线轨迹（保持 set_tracks 顺序）。"""
        checked = set()
        for row in range(self._line_list.count()):
            item = self._line_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                checked.add(str(item.data(Qt.ItemDataRole.UserRole) or ''))
        return [track for track in self._tracks
                if str(getattr(track, 'line_id', '') or '') in checked]

    def _switch_view(self, route_key: str) -> None:
        widget = {_SEG_MAP: self._map_view, _SEG_PROFILE: self._profile_view,
                  _SEG_3D: self._3d_view}.get(route_key, self._map_view)
        self._view_stack.setCurrentWidget(widget)

    def _refresh_views(self) -> None:
        """勾选集合变化 → 三个视图同步重绘。"""
        tracks = self._checked_tracks()
        self._map_view.set_tracks(tracks, self._colors)
        self._profile_view.set_tracks(tracks, self._colors)
        self._3d_view.set_tracks(tracks, self._colors)

    def _refresh_crs_card(self) -> None:
        """投影信息卡：坐标系 / EPSG / 数据来源（按轨迹摘要汇总）。"""
        summaries = self._map_view.track_summaries()
        if not summaries:
            self._crs_label.setText('暂无轨迹数据')
            return
        lines = []
        for info in summaries:
            crs = info.get('crs') or '未标注'
            epsg = info.get('epsg')
            epsg_text = f'EPSG:{epsg}' if epsg is not None else 'EPSG 未识别'
            mapped_text = '已配准底图' if info.get('mapped') else '原始坐标（未配准底图）'
            source = info.get('source') or '未知来源'
            lines.append(
                f"{info.get('name') or info.get('line_id')}: {crs}（{epsg_text}，"
                f'{mapped_text}，{source}）')
        self._crs_label.setText('\n'.join(lines))

    def _refresh_detail(self) -> None:
        """测线详情卡：名称 / 道数 / 高程范围 / RTK状态 / 坐标系。"""
        line_id = self._selected_line_id()
        track = self._tracks_by_id.get(line_id)
        line = self._lines_by_id.get(line_id)
        labels = self._detail_labels
        if track is None and line is None:
            for label in labels.values():
                label.setText('--')
            self._set_current_btn.setEnabled(False)
            return
        name = str(getattr(track, 'name', '') or getattr(line, 'name', '') or line_id)
        points = list(getattr(track, 'points', ()) or ()) if track is not None else []
        trace_count = int(getattr(line, 'trace_count', 0) or 0) if line is not None else 0
        traces_text = str(trace_count or len(points) or '--')
        elevations = [float(getattr(p, 'elevation_m', 0.0))
                      for p in points if np.isfinite(float(getattr(p, 'elevation_m', 0.0)))]
        if elevations:
            elevation_text = f'{min(elevations):.2f} ~ {max(elevations):.2f} m'
        else:
            elevation_text = '--'
        rtk = str(getattr(line, 'rtk_status', '') or '--') if line is not None else '--'
        crs = str(getattr(track, 'coordinate_system', '') or '--') if track is not None else '--'
        labels['name'].setText(name)
        labels['traces'].setText(traces_text)
        labels['elevation'].setText(elevation_text)
        labels['rtk'].setText(rtk)
        labels['crs'].setText(crs)
        self._set_current_btn.setEnabled(bool(line_id))

    # ---------------- 槽
    def _on_line_check_changed(self, _item) -> None:
        self._refresh_views()

    def _on_line_selected(self, _current, _previous=None) -> None:
        self._refresh_detail()

    def _on_basemap_changed(self, index: int) -> None:
        if self._restoring_basemap:
            return
        source = str(self._basemap_combo.itemData(index) or DEFAULT_TILE_SOURCE)
        self._map_view.set_source(source)
        sm = SettingsManager()
        sm.set('spatial_basemap_source', source)
        sm.save()

    def _on_prefetch_clicked(self) -> None:
        # 优先按测线包围盒下载对应地理区域；无已配准轨迹回退当前视野
        queued = self._map_view.prefetch_tracks()
        if queued == 0 and self._map_view.tracks_bbox_lonlat() is None:
            queued = self._map_view.prefetch_current_view(max_extra_zoom=2)
        if queued > 0:
            self._prefetch_label.setText(f'正在下载 {queued} 张瓦片…')
        else:
            self._prefetch_label.setText('测线区域瓦片已全部缓存')
        self.basemap_prefetch_requested.emit()

    def _on_terrain_source_changed(self, index: int) -> None:
        """地形来源下拉：本地 DEM 尚未导入时先引导导入，取消则回退原选择。"""
        if self._restoring_terrain:
            return
        mode = str(self._3d_terrain_combo.itemData(index) or 'online')
        if mode == 'local_dem' and not self._3d_dem_clear_btn.isEnabled():
            self._on_import_dem_clicked()   # 成功路径内部会切到 local_dem
            if self._terrain_mode != 'local_dem':
                # 用户取消导入 → 下拉回退到原来源
                self._set_terrain_mode(self._terrain_mode, persist=False)
            return
        self._set_terrain_mode(mode)

    def _on_import_dem_clicked(self) -> None:
        """导入本地 DEM（XYZ 格网）：三维地形改用本地高程，免在线下载。

        路径记入设置，之后启动自动加载，无需重复导入。
        """
        path, _selected = QFileDialog.getOpenFileName(
            self, '选择本地 DEM 格网文件', '',
            'DEM 格网 (*.xyz *.csv *.txt);;所有文件 (*)')
        if not path:
            return
        try:
            dem = load_xyz_grid(path)
        except (OSError, ValueError) as exc:
            self._3d_dem_label.setText(f'导入失败：{exc}')
            return
        self._apply_local_dem(dem, path)
        self._set_terrain_mode('local_dem')
        sm = SettingsManager()
        sm.set('spatial_local_dem', path)
        sm.save()

    def _on_clear_dem_clicked(self) -> None:
        """清除本地 DEM：三维地形回退在线高程瓦片，并删除设置记录。"""
        self._3d_view.set_local_dem(None)
        self._dem_base_text = ''
        self._3d_dem_label.setText('未导入（在线下载高程）')
        self._3d_dem_label.setToolTip('')
        self._3d_dem_clear_btn.setEnabled(False)
        if self._terrain_mode == 'local_dem':
            self._set_terrain_mode('online')
        sm = SettingsManager()
        sm.set('spatial_local_dem', '')
        sm.save()

    def _auto_prefetch_tracks(self) -> None:
        """轨迹加载后自动下载测线所在地理区域（按 包围盒+瓦图源 去重）。

        只处理已配准轨迹；未配准（原始坐标）测线不触发下载。
        受设置项 auto_prefetch_basemap 控制（P2-4）。
        """
        if not self._auto_prefetch_enabled:
            return
        bbox = self._map_view.tracks_bbox_lonlat()
        if bbox is None:
            return
        key = (tuple(round(v, 5) for v in bbox), self._map_view.source_key())
        if key == self._last_auto_prefetch_key:
            return
        self._last_auto_prefetch_key = key
        queued = self._map_view.prefetch_tracks()
        if queued > 0:
            self._prefetch_label.setText(f'自动下载测线区域 {queued} 张瓦片…')

    def set_auto_prefetch_enabled(self, enabled: bool) -> None:
        """设置页开关：自动预下载底图是否启用（P2-4）。"""
        self._auto_prefetch_enabled = bool(enabled)

    def _on_prefetch_progress(self, done: int, total: int) -> None:
        if total <= 0:
            return
        done = min(int(done), int(total))
        if done >= total:
            self._prefetch_label.setText(f'预下载完成：{total} 张瓦片')
        else:
            self._prefetch_label.setText(f'正在下载 {done}/{total} 张瓦片…')

    def _on_set_current_clicked(self) -> None:
        line_id = self._selected_line_id()
        if line_id:
            self.current_line_requested.emit(line_id)
