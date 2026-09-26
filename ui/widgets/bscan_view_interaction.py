"""BScanView 交互 mixin——右键菜单 / 拾取与十字光标 / 坐标映射。

2026-09-24 组件化拆分（自 bscan_view.py 机械搬迁，方法体零改动）：
BScanView 一个类曾达 1646 行/95 方法，交互与坐标映射是最自洽的一块。
状态仍在 ``self``（mixin 不持有状态），对外 API 与信号零变化——
``BScanView(BScanViewInteractionMixin, GraphicsViewBase, QWidget)``。
``BScanDisplayMode`` 枚举随迁至此（菜单与 wiggle 共用），bscan_view
re-export 保持既有导入路径（tests 直接 ``from ui.widgets.bscan_view
import BScanDisplayMode`` 依然成立）。
"""

from enum import Enum

import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.widgets.bscan_axes import sample_unit_label
from ui.widgets.context_menus import (RoundMenu, add_action,
                                      add_checkable_submenu, make_menu)


class BScanDisplayMode(Enum):
    """B-scan 显示模式（Phase1 1.2）：灰度 / 变面积 / 波形叠加。"""

    GRAYSCALE = 'grayscale'
    WIGGLE = 'wiggle'          # 变面积：正半轴填充
    WAVEFORM = 'waveform'      # 波形叠加：正负对称双线


def _downsample_map_index(view_index: int, view_count: int,
                          data_count: int) -> int:
    """显示坐标索引 → 原始数据索引（strided 降采样近似线性映射）。

    读数"原始约 N"标注与 pick/overlay 坐标换算共用本函数，避免双实现漂移。
    """
    return int(round(view_index * (data_count - 1)
                     / max(view_count - 1, 1)))


def _downsample_map_index_inverse(data_index, view_count: int,
                                  data_count: int):
    """原始数据索引 → 显示坐标（_downsample_map_index 的逆映射）。"""
    return (data_index * max(view_count - 1, 1)
            / max(data_count - 1, 1))


def format_crosshair_readout(trace: int, sample: int, shape: tuple,
                             amplitude: float, *,
                             trace_axis_m=None, sample_axis=None,
                             sample_axis_label: str = '',
                             trace_count: int = 0,
                             sample_count: int = 0) -> str:
    """十字光标读数文本（纯函数，便于测试）。

    trace/sample 为显示坐标（0 基）；shape=(n_traces, n_display_rows)。
    trace_axis_m/sample_axis 为与显示矩阵等长的物理轴（None 跳过）——
    海拔模式下调用方传共享海拔轴 + '海拔 (m)' 标签，纵轴读数即为海拔。
    trace_count/sample_count 为原始（未降采样）数量，与显示数不同
    时追加"（原始约 N）"（strided 降采样近似线性映射）。
    """
    n_traces, n_rows = shape
    lines = [f'道 {trace + 1}']
    if trace_count and trace_count != n_traces:
        approx = _downsample_map_index(trace, n_traces, trace_count)
        lines[0] += f'（原始约 {approx + 1}）'
    if trace_axis_m is not None and 0 <= trace < len(trace_axis_m):
        lines.append(f'距起点 {float(trace_axis_m[trace]):.3g} m')
    if sample_axis is not None and 0 <= sample < len(sample_axis):
        label = sample_axis_label or '纵轴'
        lines.append(f'{label} {float(sample_axis[sample]):.4g}')
    else:
        text = f'采样 {sample + 1}'
        if sample_count and sample_count != n_rows:
            approx = _downsample_map_index(sample, n_rows, sample_count)
            text += f'（原始约 {approx + 1}）'
        lines.append(text)
    if amplitude is None or not math.isfinite(float(amplitude)):
        lines.append('幅值 —')       # 海拔模式地表以上是 NaN（留白区）
    else:
        lines.append(f'幅值 {float(amplitude):.4g}')
    return '\n'.join(lines)




class BScanViewInteractionMixin:

    # ------------------------------------------------------------------ 右键菜单
    def _show_context_menu(self, event) -> None:
        menu = make_menu(self)
        self._add_menu_zoom(menu)
        menu.addSeparator()
        self._add_menu_axes(menu)
        menu.addSeparator()
        add_checkable_submenu(menu, '色标', constants.COLORMAPS,
                              self._cmap_name, self._choose_colormap)
        add_action(menu, None, '色阶设置…', self._edit_levels,
                   enabled=self._image_shape is not None)
        self._add_menu_toggles(menu)
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制图像', self._copy_image,
                   enabled=self._image_shape is not None)
        add_action(menu, FIF.SAVE, '导出 PNG…',
                   lambda: self.export_png(
                       title=self._export_title, prefix='bscan'),
                   enabled=self._image_shape is not None)
        menu.exec(event.screenPos().toPoint())

    def _add_menu_zoom(self, menu) -> None:
        """缩放与视野组：缩放 / 回到全览 / 比例 / 全屏 / 铺满。

        「铺满」（收起页面侧栏）只在宿主页接管后出现——原工具条按钮
        退役后，这里是除快捷语义外的唯一入口。
        """
        add_action(menu, FIF.ZOOM_IN, '放大', self.zoom_in)
        add_action(menu, FIF.ZOOM_OUT, '缩小', self.zoom_out)
        add_action(menu, FIF.FIT_PAGE, '回到全览（铺满窗口）', self.fit_to_data)
        add_action(menu, None, '方形显示', self.fit_square)
        add_action(menu, None, '1:1（数据格等比）', self.reset_1to1)
        add_action(menu, FIF.FULL_SCREEN, '全屏浏览', self.enter_fullscreen)
        if self._expand_enabled:
            add_action(menu, FIF.VIEW, '铺满（收起页面侧栏）',
                       self.request_expand)

    def _add_menu_axes(self, menu) -> None:
        """轴单位组：工具条精简后这里是切换主入口（另一处是设置页）。"""
        x_menu = RoundMenu('横轴单位', menu)
        for mode, label in (('trace', '道数'), ('distance', '距离 (m)')):
            x_menu.addAction(self._axis_menu_action(
                label, self._x_axis == mode,
                mode != 'distance' or self.distance_available(),
                lambda m=mode: self.set_x_axis_mode(m, notify=True)))
        menu.addMenu(x_menu)
        y_menu = RoundMenu('纵轴单位', menu)
        y_menu.addAction(self._axis_menu_action(
            sample_unit_label(self._sample_axis_label),
            self.effective_y_axis_mode() == 'sample', self._image_shape is not None,
            lambda: self.set_y_axis_mode('sample', notify=True)))
        y_menu.addAction(self._axis_menu_action(
            '海拔 (m)', self.effective_y_axis_mode() == 'elevation',
            self.elevation_available(),
            lambda: self.set_y_axis_mode('elevation', notify=True)))
        menu.addMenu(y_menu)

    @staticmethod
    def _axis_menu_action(label: str, checked: bool, enabled: bool, slot):
        from qfluentwidgets import Action
        act = Action(label)
        act.setCheckable(True)
        act.setChecked(bool(checked))
        act.setEnabled(bool(enabled))
        act.triggered.connect(lambda _checked=False: slot())
        return act

    def _add_menu_toggles(self, menu) -> None:
        """交互开关组：十字光标 / A-scan 跟随 / 色标显隐 / 显示模式。"""
        from qfluentwidgets import Action
        crosshair_action = Action('十字光标读数')
        crosshair_action.setCheckable(True)
        crosshair_action.setChecked(self._crosshair_on)
        crosshair_action.triggered.connect(self._toggle_crosshair)
        menu.addAction(crosshair_action)
        ascan_action = Action('A-scan 波形跟随')
        ascan_action.setCheckable(True)
        ascan_action.setChecked(self._ascan_follow)
        ascan_action.triggered.connect(self.set_ascan_follow)
        menu.addAction(ascan_action)
        if self._colorbar is not None:
            colorbar_action = Action('显示色标')
            colorbar_action.setCheckable(True)
            colorbar_action.setChecked(self._colorbar_visible)
            colorbar_action.triggered.connect(
                lambda checked=False: self.set_colorbar_visible(
                    checked, notify=True))
            menu.addAction(colorbar_action)
        mode_submenu = RoundMenu('显示模式', menu)
        for mode in BScanDisplayMode:
            mode_label = {
                BScanDisplayMode.GRAYSCALE: '灰度',
                BScanDisplayMode.WIGGLE: '变面积（Wiggle）',
                BScanDisplayMode.WAVEFORM: '波形叠加',
            }[mode]
            act = Action(mode_label)
            act.setCheckable(True)
            act.setChecked(self.display_mode is mode)
            act.triggered.connect(
                lambda _checked=False, m=mode: self.set_display_mode(m))
            mode_submenu.addAction(act)
        menu.addMenu(mode_submenu)
        gain_submenu = RoundMenu('显示增益', menu)
        for mode, label in (('off', '关闭'), ('sec', 'SEC 补偿'),
                            ('tvg', 'TVG 补偿（滑条调参）')):
            act = Action(label)
            act.setCheckable(True)
            act.setChecked(self._gain_mode == mode)
            act.triggered.connect(
                lambda _checked=False, m=mode: self._on_gain_menu(m))
            gain_submenu.addAction(act)
        menu.addMenu(gain_submenu)

    # ------------------------------------------------------------------ 交互
    def set_pick_enabled(self, enabled: bool) -> None:
        """开启后鼠标点击把图像坐标换算成 (trace, sample) 发 sig_point_picked。"""
        self._pick_enabled = bool(enabled)

    def _toggle_crosshair(self, checked: bool) -> None:
        self._crosshair_on = bool(checked)
        if not self._crosshair_on:
            self._hide_crosshair()

    def set_display_mode(self, mode) -> None:
        """切换灰度/变面积/波形叠加三态；模式切换时重置视野（用户主动切换）。"""
        if isinstance(mode, str):
            mode = BScanDisplayMode(mode)
        if not isinstance(mode, BScanDisplayMode):
            raise ValueError(f'未知显示模式: {mode!r}')
        self.display_mode = mode
        self._apply_display_mode(mode, reset_view=True)

    def _apply_display_mode(self, mode, *, reset_view: bool) -> None:
        """应用显示模式；reset_view 仅用户切换模式时为 True。

        set_matrix 新数据到达时以 reset_view=False 重渲染波形，保持用户
        当前缩放/视野（原实现无条件 autoRange，新数据会冲掉手动缩放）。
        波形数据源固定用未变形的 ``_matrix``（套显示增益，与灰度图一致）：
        海拔模式下 ImageItem 里是 warp 后的矩阵，直接取列会把波形纵向
        重采样；正确做法是值取原矩阵（增益后）、纵坐标按海拔映射
        （见 _render_wiggle 的 y_context）。
        """
        img = self._image_item.image
        if img is None or self._matrix is None:
            return
        if mode is BScanDisplayMode.GRAYSCALE:
            if self._wiggle_item is not None:
                self._wiggle_item.hide()
            self._image_item.show()
            return
        if mode is BScanDisplayMode.WIGGLE:
            self._image_item.hide()
        else:
            self._image_item.show()
        self._render_wiggle(self._gain_applied(self._matrix),
                            filled=(mode is BScanDisplayMode.WIGGLE),
                            symmetric=(mode is BScanDisplayMode.WAVEFORM),
                            reset_view=reset_view,
                            y_context=self._wiggle_y_context())

    def _wiggle_y_context(self):
        """海拔模式下波形的纵坐标映射参数；采样轴模式返回 None。

        返回 ``(elev_axis, ground, depth)``：波形每一点的纵坐标 =
        (最高海拔 − (该道地面高程 − 深度)) / 行距，即海拔网格行号——
        每道波形从自己的地面高程处起画，随地形起伏。
        """
        if not self._showing_elevation or self._elev_axis is None:
            return None
        return self._elev_axis, self._ground_elevation_m, self._depth_axis_m

    def _render_wiggle(self, data, *, filled: bool, symmetric: bool,
                       reset_view: bool = False, y_context=None) -> None:
        """变面积/波形叠加：pg.arrayToQPath C 层批量构建（向量化，非逐道循环）。

        填充（变面积）用每道"上升沿正包络+基线回程"闭合；波形叠加为
        全波形双线。connect 数组在每道末尾断开，避免跨道连线。

        :param y_context: None → 纵坐标 = 采样点行号（采样轴模式）；
            (elev_axis, ground, depth) → 纵坐标 = 海拔网格行号（海拔模式）。
        """
        from PyQt6.QtGui import QColor

        data = np.asarray(data, dtype=np.float64)
        n_samples, n_traces = data.shape
        levels = self._image_item.levels
        if levels is None or len(levels) < 2 or levels[0] == levels[1]:
            levels = (-1.0, 1.0)
        vmax = max(abs(float(levels[0])), abs(float(levels[1]))) or 1.0
        # 显示层降采样：波形形态用 ≤500 点/道已足够（峰值包络不失真），
        # 全样本会让 QPainterPath 剖析+绘制逼近秒级（实测 2000×1000 全点
        # ~1.1s，500 点 ~40ms）。
        step = max(1, n_samples // 500)
        if step > 1:
            data = data[::step, :]
        n_samples_ds = data.shape[0]
        # 振幅归一：每道正峰占相邻道间距的 0.45（Wiggle 惯例不重叠）
        gain = 0.45 / vmax
        bases = np.arange(n_traces, dtype=np.float64) + 0.5  # 每道基线

        amp = data * gain                      # (samples_ds, traces)
        if filled:
            xs = bases[None, :] + np.maximum(amp, 0.0)
        else:
            xs = bases[None, :] + amp
        ys_col, ys_bottom = self._wiggle_y_arrays(
            n_samples, n_samples_ds, step, n_traces, y_context)
        # 坐标序列：道1全部点 → 道1基线回程点 → 道2... （每道 2 段闭合）
        xs_t = np.concatenate([xs, bases[None, :]], axis=0)      # (n_ds+1, traces)
        ys_t = np.concatenate([ys_col, ys_bottom[None, :]], axis=0)
        # 列优先展开为点序列，connect=0 断开道间
        x_flat = xs_t.T.ravel()
        y_flat = ys_t.T.ravel()
        connect = np.ones(x_flat.size, dtype=np.int32)
        connect[np.arange(n_traces) * (n_samples_ds + 1) + n_samples_ds] = 0
        connect[-1] = 0

        path = pg.arrayToQPath(x_flat, y_flat, connect)
        path.setFillRule(Qt.FillRule.OddEvenFill if filled
                         else Qt.FillRule.WindingFill)

        from pyqtgraph import GraphicsItem  # noqa: F401 - 确保 pg 图层初始化
        from PyQt6.QtWidgets import QGraphicsPathItem
        if self._wiggle_item is None:
            self._wiggle_item = QGraphicsPathItem()
            self._plot.addItem(self._wiggle_item)
        self._wiggle_item.setPath(path)
        # 填充色取当前色表正端；描边为其深色
        lut = (self._cmap.getLookupTable(0.0, 1.0, 2)
               if self._cmap is not None else None)
        rgb = (int(lut[1][0]), int(lut[1][1]), int(lut[1][2])) \
            if lut is not None else (30, 30, 30)
        fill = QColor(*rgb, 160)
        self._wiggle_item.setPen(pg.mkPen(QColor(*rgb).darker(140), width=1))
        self._wiggle_item.setBrush(fill if filled else pg.mkBrush(None))
        self._wiggle_item.show()
        if reset_view:
            self._plot.getViewBox().autoRange()

    def _wiggle_y_arrays(self, n_samples: int, n_samples_ds: int,
                         step: int, n_traces: int, y_context):
        """波形的纵坐标数组（降采样后每点一行号 + 每道基线回程行号）。

        采样轴模式：行号 = 降采样索引 × 步距，基线回程 = 最深采样行。
        海拔模式：行号 = (最高海拔 − (地面高程 − 深度)) / 行距，每道
        各自从自己的地面高程起画（地形起伏直接体现在波形起点上）。
        """

        if y_context is None:
            ys_col = np.repeat(
                np.arange(n_samples_ds, dtype=np.float64)[:, None] * step,
                n_traces, axis=1)
            ys_bottom = np.full(n_traces, float(n_samples - 1))
            return ys_col, ys_bottom
        elev_axis, ground, depth = y_context
        top = float(elev_axis[0])
        step_m = (top - float(elev_axis[-1])) / max(len(elev_axis) - 1, 1)
        if step_m <= 0:
            ys_col = np.zeros((n_samples_ds, n_traces), dtype=np.float64)
            return ys_col, np.zeros(n_traces, dtype=np.float64)
        sample_idx = np.clip(
            np.arange(n_samples_ds, dtype=np.float64) * step,
            0, len(depth) - 1)
        d = np.asarray(depth, dtype=np.float64)[sample_idx.astype(np.int64)]
        g = np.asarray(ground, dtype=np.float64)[:n_traces]
        # 行号 = (top − (ground − depth)) / step_m；depth ≥ 0 → 行号 ≥ 0，
        # 且 top = 最高地面 → 任何道的最浅点行号也不会越出网格顶。
        ys_col = (top - g[None, :] + d[:, None]) / step_m
        ys_bottom = (top - g + float(depth[-1])) / step_m
        return ys_col, ys_bottom

    def set_ascan_follow(self, enabled: bool) -> None:
        """开关"A-scan 波形跟随"：懒创建浮窗并同步 pick 模式（Phase1 1.1）。"""
        from ui.widgets.ascan_popup import AScanPopup

        self._ascan_follow = bool(enabled)
        if enabled:
            if self._ascan_popup is None:
                self._ascan_popup = AScanPopup(self.window())
                self._ascan_popup.closed.connect(self._on_ascan_popup_closed)
            self._ascan_popup.show()
            self.set_pick_enabled(True)
        elif self._ascan_popup is not None:
            self._ascan_popup.hide()

    def _on_ascan_popup_closed(self) -> None:
        """浮窗被用户关闭（仅隐藏、实例复用）：回落跟随标志。

        与菜单取消勾选同效——不动 pick 模式（set_ascan_follow(False)
        同样只隐藏浮窗），仅让下次右键菜单的勾选态与浮窗实际状态一致。
        """
        self._ascan_follow = False

    def _on_mouse_moved(self, pos) -> None:
        """鼠标在图像区移动：十字线跟手 + 左下角读数浮层（文本 30ms 合并刷新）。"""
        if not self._crosshair_on or self._image_shape is None:
            self._hide_crosshair()
            return
        if not self._plot.sceneBoundingRect().contains(pos):
            self._hide_crosshair()
            return
        view_point = self._plot.vb.mapSceneToView(pos)
        trace, sample = int(view_point.x()), int(view_point.y())
        n_traces, n_samples = self._image_shape
        if not (0 <= trace < n_traces and 0 <= sample < n_samples):
            self._hide_crosshair()
            return
        # ImageItem 像素中心在半整数处，十字线对到像素中心
        self._vline.setPos(trace + 0.5)
        self._hline.setPos(sample + 0.5)
        self._vline.setVisible(True)
        self._hline.setVisible(True)
        self._pending_readout = (trace, sample,
                                 float(self._image_item.image[sample, trace]))
        if self._readout.isVisible():
            if not self._readout_timer.isActive():
                self._readout_timer.start()
        else:
            self._flush_crosshair_readout()   # 首次出现即时，不等节流
        self._readout.setVisible(True)

    def _flush_crosshair_readout(self) -> None:
        """节流刷新读数文本 + 浮层定位（_readout_timer 与首次出现共用）。"""
        pending = self._pending_readout
        if pending is None:
            return
        self._pending_readout = None
        trace, sample, amplitude = pending
        # 海拔模式下纵轴读数 = 行号直查共享海拔轴（显示坐标已是行号）
        showing_elevation = self._showing_elevation
        self._readout.setText(format_crosshair_readout(
            trace, sample, self._image_shape, amplitude,
            trace_axis_m=self._trace_axis_m,
            sample_axis=(self._elev_axis if showing_elevation
                         else self._sample_axis),
            sample_axis_label=('海拔 (m)' if showing_elevation
                               else self._sample_axis_label),
            trace_count=self._trace_count,
            sample_count=self._sample_count))
        self._position_readout()

    def _hide_crosshair(self) -> None:
        self._readout_timer.stop()
        self._pending_readout = None
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._readout.hide()

    def _position_readout(self) -> None:
        self._readout.adjustSize()
        margin = 10
        self._readout.move(
            margin, self.height() - self._readout.height() - margin)

    def leaveEvent(self, event) -> None:
        self._hide_crosshair()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._readout.isVisible():
            self._position_readout()

    def set_overlay_points(self, points, color: str = constants.CHART_OVERLAY_COLOR) -> None:
        """解释页标注散点：points 为 [(trace, sample), ...]（原始数据坐标系）。

        预览矩阵可能经 strided 降采样（>900×1800），此处把原始坐标
        映射回显示坐标再绘制，保证标注落在正确的图像位置上。
        """
        spots = [{'pos': self._data_to_view(t, s), 'brush': pg.mkBrush(color)}
                 for t, s in (points or [])]
        self._scatter.setData(spots)

    # ------------------------------------------------------------------ 坐标映射
    def _view_to_data(self, trace: int, sample: int) -> tuple:
        """显示坐标 → 原始数据坐标（strided 降采样近似线性映射）。

        采样轴模式：y 显示坐标 = 采样点索引（降采样时线性映射回原始）。
        海拔模式：y 显示坐标 = 海拔网格行号 → 该道地面高程处的深度 →
        深度轴查表得采样点（行号越界夹取；地表以上/最深处以下夹到端点）。
        无物理轴元数据（直接 set_matrix）或未降采样时为恒等映射。
        """
        if self._image_shape is None:
            return int(trace), int(sample)
        n_traces, n_rows = self._image_shape
        t, r = int(trace), int(sample)
        r = max(0, min(n_rows - 1, r))
        if self._showing_elevation:
            s = self._row_to_sample(t, r)
        else:
            s = r
            if self._sample_count and self._sample_count != n_rows:
                s = _downsample_map_index(s, n_rows, self._sample_count)
        if self._trace_count and self._trace_count != n_traces:
            t = _downsample_map_index(t, n_traces, self._trace_count)
        return t, s

    def _row_to_sample(self, trace_display: int, row: int) -> int:
        """海拔网格行 → 采样点索引（该道地面高程 − 行海拔 = 深度 → 查深度轴）。"""

        ground, depth = self._ground_elevation_m, self._depth_axis_m
        if ground is None or depth is None or self._elev_axis is None:
            return 0
        if not len(depth) or not len(ground):
            return 0
        ti = max(0, min(len(ground) - 1, int(trace_display)))
        ri = max(0, min(len(self._elev_axis) - 1, int(row)))
        depth_at_row = float(ground[ti]) - float(self._elev_axis[ri])
        idx = int(np.searchsorted(np.asarray(depth, dtype=np.float64),
                                  depth_at_row))
        return max(0, min(len(depth) - 1, idx))

    def _data_to_view(self, trace, sample) -> tuple:
        """原始数据坐标 → 显示坐标（_view_to_data 的逆映射）。

        海拔模式下返回行号（浮点，overlay 散点可落在行间）。
        """
        if self._image_shape is None:
            return float(trace), float(sample)
        n_traces, n_rows = self._image_shape
        t, s = float(trace), float(sample)
        if self._trace_count and self._trace_count != n_traces:
            t = _downsample_map_index_inverse(t, n_traces, self._trace_count)
        if self._showing_elevation:
            r = self._sample_to_row(t, s)
        else:
            r = s
            if self._sample_count and self._sample_count != n_rows:
                r = _downsample_map_index_inverse(s, n_rows, self._sample_count)
        return t, r

    def _sample_to_row(self, trace_display: float, sample_display: float) -> float:
        """采样点 → 海拔网格行（浮点）：行 = (最高海拔 − 该点海拔) / 行距。"""
        ground, depth, elev = (self._ground_elevation_m,
                               self._depth_axis_m, self._elev_axis)
        if ground is None or depth is None or elev is None or len(elev) < 2:
            return float(sample_display)
        ti = max(0, min(len(ground) - 1, int(round(trace_display))))
        si = max(0, min(len(depth) - 1, int(round(sample_display))))
        elevation = float(ground[ti]) - float(depth[si])
        top = float(elev[0])
        step = (top - float(elev[-1])) / (len(elev) - 1)
        if step <= 0:
            return 0.0
        return (top - elevation) / step

    def _on_mouse_clicked(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            if self._plot.sceneBoundingRect().contains(event.scenePos()):
                self._show_context_menu(event)
            return
        if not self._pick_enabled or self._image_shape is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        view_point = self._plot.vb.mapSceneToView(pos)
        trace, sample = int(view_point.x()), int(view_point.y())
        n_traces, n_samples = self._image_shape
        if 0 <= trace < n_traces and 0 <= sample < n_samples:
            # sig_point_picked 契约是原始数据坐标：预览可能降采样，
            # 后端会话（界面标注/双曲线拾取）全部在原始坐标系工作。
            data_trace, data_sample = self._view_to_data(trace, sample)
            self._emit_point_picked(data_trace, data_sample, view_trace=trace)

    def _emit_point_picked(self, trace: int, sample: int, *, view_trace: int | None = None) -> None:
        """统一 pick 发射口：跟随浮窗消费 + 原信号照常发出。

        trace/sample 为原始数据坐标（信号契约）；view_trace 是点击处的
        显示坐标道号——跟随浮窗画的波形取自当前预览矩阵，必须用显示
        索引取列，否则降采样预览下波形与十字线错位。
        """
        if (self._ascan_follow and self._ascan_popup is not None
                and self._image_shape is not None):
            import numpy as np

            # 波形取自未变形的预览矩阵：海拔模式下 ImageItem 里是 warp 后
            # 的列（纵向重采样过），不是原始 A-scan，弹窗必须用 _matrix。
            image = self._matrix
            if image is None:
                self.sig_point_picked.emit(trace, sample)
                return
            image = np.asarray(image)
            display_trace = self._image_shape[0] - 1 if view_trace is None else int(view_trace)
            if 0 <= display_trace < image.shape[1]:
                dist = (self._trace_axis_m[display_trace]
                        if self._trace_axis_m is not None
                        and display_trace < len(self._trace_axis_m) else None)
                self._ascan_popup.show_trace(
                    image[:, display_trace], trace_index=display_trace, distance_m=dist)
        self.sig_point_picked.emit(trace, sample)

