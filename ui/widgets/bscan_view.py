"""BScanView — B-Scan 图像显示控件（SPEC §5.1）。

性能纪律（style_spec §1 / §3.3）：
- 预分配 ImageItem 复用，setImage 不重建
- matrix 约定为 (samples, traces)，row-major 语义下零拷贝直接显示
  （x=道/列、y=采样/行；不再 .T，见 set_matrix 注释）
- ImageItem(axisOrder='row-major') + invertY(True)
- autoLevels=False + 显式 levels=(vmin, vmax)
- ColorBarItem 随行同步

十字光标读数：鼠标在图像区移动时显示十字线 + 左下角读数浮层
（道号/距起点/纵轴物理值/幅值；PreviewBundle 带 trace_axis_m /
sample_axis 时显示物理量，降采样数据附"原始约 N"），右键菜单可开关。

右键菜单（RoundMenu）：缩放组 / 色标子菜单 / 十字光标开关 /
复制图像 / 导出 PNG。
接入时已 vb.setMenuEnabled(False) 关闭 pyqtgraph 原生英文菜单
（代价：右键拖拽框选缩放失效，由菜单缩放项补偿）。
"""

from PyQt6.QtCore import Qt, QDateTime, pyqtSignal
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget

import pyqtgraph as pg
from PyQt6.QtWidgets import QLabel
from qfluentwidgets import FluentIcon as FIF, PushButton

from ui import constants, file_dialogs
from ui.widgets.context_menus import (add_action, add_checkable_submenu,
                                      make_menu)


def format_crosshair_readout(trace: int, sample: int, shape: tuple,
                             amplitude: float, *,
                             trace_axis_m=None, sample_axis=None,
                             sample_axis_label: str = '',
                             trace_count: int = 0,
                             sample_count: int = 0) -> str:
    """十字光标读数文本（纯函数，便于测试）。

    trace/sample 为显示坐标（0 基）；shape=(n_traces, n_samples)。
    trace_axis_m/sample_axis 为与显示矩阵等长的物理轴（None 跳过）。
    trace_count/sample_count 为原始（未降采样）数量，与显示数不同
    时追加"（原始约 N）"（strided 降采样近似线性映射）。
    """
    n_traces, n_samples = shape
    lines = [f'道 {trace + 1}']
    if trace_count and trace_count != n_traces:
        approx = int(round(trace * (trace_count - 1) / max(n_traces - 1, 1)))
        lines[0] += f'（原始约 {approx + 1}）'
    if trace_axis_m is not None and 0 <= trace < len(trace_axis_m):
        lines.append(f'距起点 {float(trace_axis_m[trace]):.3g} m')
    if sample_axis is not None and 0 <= sample < len(sample_axis):
        label = sample_axis_label or '纵轴'
        lines.append(f'{label} {float(sample_axis[sample]):.4g}')
    else:
        text = f'采样 {sample + 1}'
        if sample_count and sample_count != n_samples:
            approx = int(round(sample * (sample_count - 1)
                               / max(n_samples - 1, 1)))
            text += f'（原始约 {approx + 1}）'
        lines.append(text)
    lines.append(f'幅值 {amplitude:.4g}')
    return '\n'.join(lines)


class BScanView(QWidget):
    """B-Scan 剖面图像视图。

    信号:
        sig_point_picked(int, int): pick 模式下鼠标点击发射 (trace_index, sample_index)，
            统一为原始数据坐标（预览降采样时已换算）。
        sig_colormap_changed(str): 右键菜单切换色标时发射（页面同步 ComboBox 用）。
    """

    sig_point_picked = pyqtSignal(int, int)
    sig_colormap_changed = pyqtSignal(str)

    def __init__(self, parent=None, *, with_colorbar: bool = True,
                 default_aspect: str = 'square'):
        super().__init__(parent)
        self._pick_enabled = False
        self._image_shape = None  # (traces, samples) 显示坐标系尺寸
        # 显示比例策略：'square' 近似方形（默认，B-Scan 习惯比例）/
        # 'free' 拉伸铺满 / 'cell' 数据格 1:1
        self._aspect_mode = default_aspect if default_aspect in (
            'square', 'free', 'cell') else 'square'
        self._cmap_name = constants.DEFAULT_COLORMAP
        # 十字光标读数状态（PreviewBundle 物理轴元数据）
        self._crosshair_on = True
        self._trace_axis_m = None
        self._sample_axis = None
        self._sample_axis_label = ''
        self._trace_count = 0
        self._sample_count = 0

        self._glw = pg.GraphicsLayoutWidget(self)
        self._plot = self._glw.addPlot(row=0, col=0, title='B-Scan图像')
        self._plot.setLabel('bottom', '道数')
        self._plot.setLabel('left', '采样点')
        self._plot.invertY(True)
        self._plot.setMouseEnabled(x=True, y=True)
        # 关闭 pyqtgraph 原生英文右键菜单（右拖框选缩放随之失效，
        # 缩放操作由自定义 RoundMenu 提供）
        self._plot.vb.setMenuEnabled(False)

        # 预分配并复用
        self._image_item = pg.ImageItem(axisOrder='row-major')
        self._image_item.setLevels((0.0, 1.0))
        self._plot.addItem(self._image_item)

        # 十字光标线（默认隐藏，鼠标进入图像区显示）
        self._vline = pg.InfiniteLine(angle=90, movable=False)
        self._hline = pg.InfiniteLine(angle=0, movable=False)
        for line in (self._vline, self._hline):
            line.setVisible(False)
            self._plot.addItem(line, ignoreBounds=True)

        self._colorbar = None
        if with_colorbar:
            self._colorbar = pg.ColorBarItem(label='幅度', interactive=False)
            self._colorbar.setImageItem(self._image_item,
                                        insert_in=self._plot)

        # overlay 标注散点（解释页）
        self._scatter = pg.ScatterPlotItem(size=8, pen=None,
                                           brush=pg.mkBrush(constants.CHART_OVERLAY_COLOR))
        self._plot.addItem(self._scatter)

        self._cmap = None
        self.set_colormap('seismic')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._toolbar = QWidget(self)
        self._toolbar.setObjectName('bscanToolbar')
        self._toolbar.setLayout(self._build_toolbar())
        layout.addWidget(self._toolbar)
        layout.addWidget(self._glw, 1)

        # 十字光标读数浮层（左下角，半透明底白字，深浅主题通用）
        self._readout = QLabel(self)
        self._readout.setStyleSheet(
            'QLabel { background-color: rgba(0, 0, 0, 150); '
            'color: white; border-radius: 4px; padding: 6px 10px; '
            'font-size: 12px; line-height: 1.5; }')
        self._readout.setWordWrap(False)
        self._readout.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._readout.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._readout.hide()

        self._glw.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._glw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        from qfluentwidgets import isDarkTheme
        self.apply_theme(isDarkTheme())

    def _build_toolbar(self) -> QHBoxLayout:
        """缩放/平移/自适应工具条。"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(4)

        self._zoom_in_btn = PushButton('+', self)
        self._zoom_in_btn.setToolTip('放大')
        self._zoom_in_btn.setFixedSize(28, 24)
        self._zoom_in_btn.clicked.connect(self.zoom_in)
        layout.addWidget(self._zoom_in_btn)

        self._zoom_out_btn = PushButton('-', self)
        self._zoom_out_btn.setToolTip('缩小')
        self._zoom_out_btn.setFixedSize(28, 24)
        self._zoom_out_btn.clicked.connect(self.zoom_out)
        layout.addWidget(self._zoom_out_btn)

        self._fit_btn = PushButton('自适应', self)
        self._fit_btn.setToolTip('拉伸铺满窗口')
        self._fit_btn.setMinimumHeight(24)
        self._fit_btn.clicked.connect(self.fit_to_data)
        layout.addWidget(self._fit_btn)

        self._square_btn = PushButton('方形', self)
        self._square_btn.setToolTip('图像整体按近似正方形显示（B-Scan 常用比例）')
        self._square_btn.setMinimumHeight(24)
        self._square_btn.clicked.connect(self.fit_square)
        layout.addWidget(self._square_btn)

        self._one_to_one_btn = PushButton('1:1', self)
        self._one_to_one_btn.setToolTip('像素 1:1 显示（一道 = 一采样点等宽等高）')
        self._one_to_one_btn.setMinimumHeight(24)
        self._one_to_one_btn.clicked.connect(self.reset_1to1)
        layout.addWidget(self._one_to_one_btn)

        self._toolbar_buttons = (self._zoom_in_btn, self._zoom_out_btn,
                                 self._fit_btn, self._square_btn,
                                 self._one_to_one_btn)

        layout.addStretch(1)
        return layout

    # ------------------------------------------------------------------ 缩放
    def zoom_in(self) -> None:
        """放大 20%。"""
        self._plot.vb.scaleBy((1.2, 1.2))

    def zoom_out(self) -> None:
        """缩小 20%。"""
        self._plot.vb.scaleBy((1.0 / 1.2, 1.0 / 1.2))

    def fit_to_data(self) -> None:
        """自适应窗口：显示全部数据（解除纵横锁定，拉伸铺满）。"""
        self._aspect_mode = 'free'
        self._plot.vb.setAspectLocked(False)
        self._plot.vb.autoRange()

    def fit_square(self) -> None:
        """近似方形显示：锁定纵横比，使数据包围盒在屏幕上接近正方形。

        pyqtgraph setAspectLocked 的 ratio 是「一个 x 单位在屏幕上的宽度 /
        一个 y 单位在屏幕上的高度」。数据盒宽 n_traces、高 n_samples，
        要让它显示为正方形：n_traces * ratio = n_samples → ratio = 高/宽。
        """
        self._aspect_mode = 'square'
        if self._image_shape is None:
            self._plot.vb.setAspectLocked(False)
            self._plot.vb.autoRange()
            return
        n_traces, n_samples = self._image_shape
        vb = self._plot.vb
        vb.setAspectLocked(False)
        vb.setRange(xRange=(0, n_traces), yRange=(0, n_samples), padding=0.02)
        vb.setAspectLocked(True, ratio=float(n_samples) / max(float(n_traces), 1.0))

    def reset_1to1(self) -> None:
        """恢复像素 1:1 显示（x/y 轴等比例）。"""
        self._aspect_mode = 'cell'
        if self._image_shape is None:
            self._plot.vb.setAspectLocked(False)
            self._plot.vb.autoRange()
            return
        n_traces, n_samples = self._image_shape
        self._plot.vb.setRange(
            xRange=(0, n_traces), yRange=(0, n_samples), padding=0.0)
        self._plot.vb.setAspectLocked(True, ratio=1.0)

    def _fit_current_mode(self) -> None:
        """按当前比例策略铺满视野（新数据到达时调用，不重置用户选择）。"""
        if self._aspect_mode == 'free':
            self._plot.vb.setAspectLocked(False)
            self._plot.vb.autoRange()
        elif self._aspect_mode == 'cell':
            self.reset_1to1()
        else:
            self.fit_square()

    # ------------------------------------------------------------------ 数据
    def set_bundle(self, bundle) -> None:
        """接收 PreviewBundle（鸭子类型，不 import core.gui_rendering）。"""
        self.set_matrix(
            bundle.matrix, bundle.vmin, bundle.vmax,
            title=getattr(bundle, 'title', '') or 'B-Scan图像',
            x_label=getattr(bundle, 'x_label', '道数'),
            y_label=getattr(bundle, 'y_label', '采样点'),
        )
        # 十字光标读数用物理轴元数据（可选；与显示矩阵同降采样）
        self._trace_axis_m = getattr(bundle, 'trace_axis_m', None)
        self._sample_axis = getattr(bundle, 'sample_axis', None)
        self._sample_axis_label = str(
            getattr(bundle, 'sample_axis_label', '') or '')
        self._trace_count = int(getattr(bundle, 'trace_count', 0) or 0)
        self._sample_count = int(getattr(bundle, 'sample_count', 0) or 0)

    def set_matrix(self, matrix, vmin, vmax, *,
                   title="", x_label="道数", y_label="采样点") -> None:
        """matrix 为 (samples, traces)，零拷贝视图直接显示。

        row-major 语义下 array[r, c] → (x=c, y=r)，故 (samples, traces)
        矩阵本身就是显示布局：x=道数（列）、y=采样点（行），无需转置拷贝。
        （SPEC §5.1 字面要求"显示 .T 转置视图"系沿袭师兄缓冲 (traces,
        samples) 约定；本控件输入约定为 (samples, traces)，若再 .T 会
        使 x/y 轴互换，与 x_label=道数 / y_label=采样点 及
        sig_point_picked(trace, sample) 契约矛盾，故按契约语义实现。）
        """
        import numpy as np

        mat = np.asarray(matrix)
        if mat.ndim != 2:
            raise ValueError('B-Scan 矩阵必须是二维 (samples, traces)')
        view = mat  # 零拷贝；row-major 下 y=采样(行)、x=道(列)
        new_shape = (view.shape[1], view.shape[0])  # (traces, samples)
        shape_changed = new_shape != self._image_shape
        self._image_shape = new_shape
        # 直接 set_matrix 的调用方没有物理轴元数据，读数退回索引显示
        self._trace_axis_m = None
        self._sample_axis = None
        self._sample_axis_label = ''
        self._trace_count = 0
        self._sample_count = 0
        self._image_item.setImage(view, autoLevels=False,
                                  levels=(float(vmin), float(vmax)))
        if shape_changed:
            # 新数据尺寸变化时按当前比例策略铺满视野，避免换测线后图像跑出可视区
            self._fit_current_mode()
        if self._colorbar is not None:
            self._colorbar.setLevels((float(vmin), float(vmax)))
        self._plot.setTitle(title or 'B-Scan图像')
        self._plot.setLabel('bottom', x_label)
        self._plot.setLabel('left', y_label)

    def set_colormap(self, name: str) -> None:
        """按 matplotlib 名取 LUT（九项见 SPEC §1，默认 seismic）。"""
        self._cmap_name = str(name)
        self._cmap = pg.colormap.getFromMatplotlib(name)
        self._image_item.setColorMap(self._cmap)
        if self._colorbar is not None:
            self._colorbar.setColorMap(self._cmap)

    def _choose_colormap(self, name: str) -> None:
        """右键菜单选色标：应用到本视图并发信号让页面同步控件。"""
        self.set_colormap(name)
        self.sig_colormap_changed.emit(name)

    # ------------------------------------------------------------------ 右键菜单
    def _show_context_menu(self, event) -> None:
        from qfluentwidgets import Action
        menu = make_menu(self)
        add_action(menu, FIF.ZOOM_IN, '放大', self.zoom_in)
        add_action(menu, FIF.ZOOM_OUT, '缩小', self.zoom_out)
        add_action(menu, FIF.FIT_PAGE, '自适应（铺满）', self.fit_to_data)
        add_action(menu, None, '方形显示', self.fit_square)
        add_action(menu, None, '1:1（数据格等比）', self.reset_1to1)
        menu.addSeparator()
        add_checkable_submenu(menu, '色标', constants.COLORMAPS,
                              self._cmap_name, self._choose_colormap)
        crosshair_action = Action('十字光标读数')
        crosshair_action.setCheckable(True)
        crosshair_action.setChecked(self._crosshair_on)
        crosshair_action.triggered.connect(self._toggle_crosshair)
        menu.addAction(crosshair_action)
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制图像', self._copy_image,
                   enabled=self._image_shape is not None)
        add_action(menu, FIF.SAVE, '导出 PNG…', self._export_png,
                   enabled=self._image_shape is not None)
        menu.exec(event.screenPos().toPoint())

    def _copy_image(self) -> None:
        """视图内容复制到剪贴板。"""
        QApplication.clipboard().setPixmap(self._glw.grab())

    def _export_png(self) -> None:
        """视图内容导出 PNG（默认文件名含标题与时间戳）。"""
        import re
        title = re.sub(r'[\\/:*?"<>|\s]+', '_', self._plot.titleLabel.text)
        stamp = QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')
        path, _selected = file_dialogs.getSaveFileName(
            self, '导出 B-Scan 图像', f'bscan_{title}_{stamp}.png',
            'PNG 图片 (*.png)')
        if path:
            self._glw.grab().save(path, 'PNG')

    # ------------------------------------------------------------------ 交互
    def set_pick_enabled(self, enabled: bool) -> None:
        """开启后鼠标点击把图像坐标换算成 (trace, sample) 发 sig_point_picked。"""
        self._pick_enabled = bool(enabled)

    def _toggle_crosshair(self, checked: bool) -> None:
        self._crosshair_on = bool(checked)
        if not self._crosshair_on:
            self._hide_crosshair()

    def _on_mouse_moved(self, pos) -> None:
        """鼠标在图像区移动：十字线跟手 + 左下角读数浮层。"""
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
        amplitude = float(self._image_item.image[sample, trace])
        self._readout.setText(format_crosshair_readout(
            trace, sample, self._image_shape, amplitude,
            trace_axis_m=self._trace_axis_m,
            sample_axis=self._sample_axis,
            sample_axis_label=self._sample_axis_label,
            trace_count=self._trace_count,
            sample_count=self._sample_count))
        self._readout.setVisible(True)
        self._position_readout()

    def _hide_crosshair(self) -> None:
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

        无物理轴元数据（直接 set_matrix）或未降采样时为恒等映射。
        """
        if self._image_shape is None:
            return int(trace), int(sample)
        n_traces, n_samples = self._image_shape
        t, s = int(trace), int(sample)
        if self._trace_count and self._trace_count != n_traces:
            t = int(round(t * (self._trace_count - 1) / max(n_traces - 1, 1)))
        if self._sample_count and self._sample_count != n_samples:
            s = int(round(s * (self._sample_count - 1) / max(n_samples - 1, 1)))
        return t, s

    def _data_to_view(self, trace, sample) -> tuple:
        """原始数据坐标 → 显示坐标（_view_to_data 的逆映射）。"""
        if self._image_shape is None:
            return float(trace), float(sample)
        n_traces, n_samples = self._image_shape
        t, s = float(trace), float(sample)
        if self._trace_count and self._trace_count != n_traces:
            t = t * max(n_traces - 1, 1) / max(self._trace_count - 1, 1)
        if self._sample_count and self._sample_count != n_samples:
            s = s * max(n_samples - 1, 1) / max(self._sample_count - 1, 1)
        return t, s

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
            # 统一发射原始数据坐标：预览可能降采样，后端会话在原始坐标系工作
            self.sig_point_picked.emit(*self._view_to_data(trace, sample))

    # ------------------------------------------------------------------ 其它
    def clear(self) -> None:
        self._image_item.clear()
        self._scatter.setData([])
        self._image_shape = None
        self._hide_crosshair()
        # P2-5：空态引导文案，替代只剩坐标轴
        self._plot.setTitle('暂无数据 — 请先在项目页导入测线')

    def apply_theme(self, dark: bool) -> None:
        """深色 bg 'k'/文字 'w'；浅色 bg 'w'/文字 'k'；轴 pen/textPen/标签同步。"""
        bg = 'k' if dark else 'w'
        fg = 'w' if dark else 'k'
        self._glw.setBackground(bg)
        surface = '#000000' if dark else '#ffffff'
        self._toolbar.setStyleSheet(
            f'QWidget#bscanToolbar {{ background-color: {surface}; }}')
        # 工具条按钮：紧凑尺寸保留，颜色随主题（硬编码浅色会在深色下突兀）
        border = '#5a5a5a' if dark else '#d9d9d9'
        hover = '#3d3d3d' if dark else '#f0f0f0'
        button_bg = '#2d2d2d' if dark else '#ffffff'
        button_text = '#f0f0f0' if dark else '#202020'
        btn_qss = (
            f'PushButton {{ background-color: {button_bg}; color: {button_text}; '
            f'border: 1px solid {border}; border-radius: 4px; '
            f'padding: 2px 8px; font-size: 11px; }}'
            f'PushButton:hover {{ background-color: {hover}; }}'
        )
        for btn in getattr(self, '_toolbar_buttons', ()):
            btn.setStyleSheet(btn_qss)
        # 注意不能用 QColor(fg)：Qt 颜色名不含 'w'/'k'，QColor('w') 非法会变黑，
        # 深色主题下轴刻度黑底黑字不可见；pg.mkPen 支持 'w'/'k' 简写
        pen = pg.mkPen(fg)
        for name in ('bottom', 'left'):
            axis = self._plot.getAxis(name)
            axis.setPen(pen)
            axis.setTextPen(pen)
            # 刻度文字由 textPen 控制，但轴标题（道数/采样点）是独立 label，
            # 不随 textPen 变色，需显式同步，否则深色主题下标题隐身
            axis.setLabel(text=axis.labelText, color=fg)
        title_item = self._plot.titleLabel
        title_item.setText(title_item.text, color=fg)
        if self._colorbar is not None:
            caxis = self._colorbar.axis
            caxis.setPen(pen)
            caxis.setTextPen(pen)
            if getattr(caxis, 'labelText', ''):
                caxis.setLabel(text=caxis.labelText, color=fg)
        # 十字光标：深色主题黄 / 浅色主题深红（图像与白底上均醒目）
        crosshair_pen = pg.mkPen(
            '#ffe135' if dark else '#c8000a',
            style=Qt.PenStyle.DashLine, width=1)
        self._vline.setPen(crosshair_pen)
        self._hline.setPen(crosshair_pen)
