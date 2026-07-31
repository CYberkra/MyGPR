"""BScanView — B-Scan 图像显示控件（SPEC §5.1）。

性能纪律（style_spec §1 / §3.3）：
- 预分配 ImageItem 复用，setImage 不重建
- matrix 约定为 (samples, traces)，row-major 语义下零拷贝直接显示
  （x=道/列、y=采样/行；不再 .T，见 set_matrix 注释）
- ImageItem(axisOrder='row-major') + invertY(True)
- autoLevels=False + 显式 levels=(vmin, vmax)
- ColorBarItem 随行同步

右键菜单（RoundMenu）：缩放组 / 色标子菜单 / 复制图像 / 导出 PNG。
接入时已 vb.setMenuEnabled(False) 关闭 pyqtgraph 原生英文菜单
（代价：右键拖拽框选缩放失效，由菜单缩放项补偿）。
"""

from PyQt6.QtCore import Qt, QDateTime, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QFileDialog,
                             QVBoxLayout, QWidget)

import pyqtgraph as pg
from qfluentwidgets import FluentIcon as FIF, PushButton

from ui import constants
from ui.widgets.context_menus import (add_action, add_checkable_submenu,
                                      make_menu)


class BScanView(QWidget):
    """B-Scan 剖面图像视图。

    信号:
        sig_point_picked(int, int): pick 模式下鼠标点击发射 (trace_index, sample_index)。
        sig_colormap_changed(str): 右键菜单切换色标时发射（页面同步 ComboBox 用）。
    """

    sig_point_picked = pyqtSignal(int, int)
    sig_colormap_changed = pyqtSignal(str)

    def __init__(self, parent=None, *, with_colorbar: bool = True):
        super().__init__(parent)
        self._pick_enabled = False
        self._image_shape = None  # (traces, samples) 显示坐标系尺寸
        self._cmap_name = constants.DEFAULT_COLORMAP

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

        self._colorbar = None
        if with_colorbar:
            self._colorbar = pg.ColorBarItem(label='幅度', interactive=False)
            self._colorbar.setImageItem(self._image_item,
                                        insert_in=self._plot)

        # overlay 标注散点（解释页）
        self._scatter = pg.ScatterPlotItem(size=8, pen=None,
                                           brush=pg.mkBrush('#fbbf24'))
        self._plot.addItem(self._scatter)

        self._cmap = None
        self.set_colormap('seismic')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._build_toolbar())
        layout.addWidget(self._glw, 1)

        self._glw.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        from qfluentwidgets import isDarkTheme
        self.apply_theme(isDarkTheme())

    def _build_toolbar(self) -> QHBoxLayout:
        """缩放/平移/自适应工具条。"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(4)

        btn_style = (
            'PushButton { border: 1px solid #ddd; border-radius: 4px; '
            'padding: 2px 8px; font-size: 11px; }'
            'PushButton:hover { background-color: #f0f0f0; }'
        )

        self._zoom_in_btn = PushButton('+', self)
        self._zoom_in_btn.setToolTip('放大')
        self._zoom_in_btn.setStyleSheet(btn_style)
        self._zoom_in_btn.setFixedSize(28, 24)
        self._zoom_in_btn.clicked.connect(self.zoom_in)
        layout.addWidget(self._zoom_in_btn)

        self._zoom_out_btn = PushButton('-', self)
        self._zoom_out_btn.setToolTip('缩小')
        self._zoom_out_btn.setStyleSheet(btn_style)
        self._zoom_out_btn.setFixedSize(28, 24)
        self._zoom_out_btn.clicked.connect(self.zoom_out)
        layout.addWidget(self._zoom_out_btn)

        self._fit_btn = PushButton('自适应', self)
        self._fit_btn.setToolTip('适应窗口大小')
        self._fit_btn.setStyleSheet(btn_style)
        self._fit_btn.setMinimumHeight(24)
        self._fit_btn.clicked.connect(self.fit_to_data)
        layout.addWidget(self._fit_btn)

        self._one_to_one_btn = PushButton('1:1', self)
        self._one_to_one_btn.setToolTip('像素 1:1 显示')
        self._one_to_one_btn.setStyleSheet(btn_style)
        self._one_to_one_btn.setMinimumHeight(24)
        self._one_to_one_btn.clicked.connect(self.reset_1to1)
        layout.addWidget(self._one_to_one_btn)

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
        """自适应窗口：显示全部数据（解除 1:1 的纵横锁定）。"""
        self._plot.vb.setAspectLocked(False)
        self._plot.vb.autoRange()

    def reset_1to1(self) -> None:
        """恢复像素 1:1 显示（x/y 轴等比例）。"""
        if self._image_shape is None:
            self.fit_to_data()
            return
        n_traces, n_samples = self._image_shape
        self._plot.vb.setRange(
            xRange=(0, n_traces), yRange=(0, n_samples), padding=0.0)
        self._plot.vb.setAspectLocked(True, ratio=1.0)

    # ------------------------------------------------------------------ 数据
    def set_bundle(self, bundle) -> None:
        """接收 PreviewBundle（鸭子类型，不 import core.gui_rendering）。"""
        self.set_matrix(
            bundle.matrix, bundle.vmin, bundle.vmax,
            title=getattr(bundle, 'title', '') or 'B-Scan图像',
            x_label=getattr(bundle, 'x_label', '道数'),
            y_label=getattr(bundle, 'y_label', '采样点'),
        )

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
        self._image_item.setImage(view, autoLevels=False,
                                  levels=(float(vmin), float(vmax)))
        if shape_changed:
            # 新数据尺寸变化时自动铺满视野，避免换测线后图像跑出可视区
            self.fit_to_data()
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
        menu = make_menu(self)
        add_action(menu, FIF.ZOOM_IN, '放大', self.zoom_in)
        add_action(menu, FIF.ZOOM_OUT, '缩小', self.zoom_out)
        add_action(menu, FIF.FIT_PAGE, '自适应', self.fit_to_data)
        add_action(menu, None, '1:1', self.reset_1to1)
        menu.addSeparator()
        add_checkable_submenu(menu, '色标', constants.COLORMAPS,
                              self._cmap_name, self._choose_colormap)
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
        path, _selected = QFileDialog.getSaveFileName(
            self, '导出 B-Scan 图像', f'bscan_{title}_{stamp}.png',
            'PNG 图片 (*.png)')
        if path:
            self._glw.grab().save(path, 'PNG')

    # ------------------------------------------------------------------ 交互
    def set_pick_enabled(self, enabled: bool) -> None:
        """开启后鼠标点击把图像坐标换算成 (trace, sample) 发 sig_point_picked。"""
        self._pick_enabled = bool(enabled)

    def set_overlay_points(self, points, color: str = '#fbbf24') -> None:
        """解释页标注散点：points 为 [(trace, sample), ...]（图像坐标系）。"""
        spots = [{'pos': (float(t), float(s)), 'brush': pg.mkBrush(color)}
                 for t, s in (points or [])]
        self._scatter.setData(spots)

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
            self.sig_point_picked.emit(trace, sample)

    # ------------------------------------------------------------------ 其它
    def clear(self) -> None:
        self._image_item.clear()
        self._scatter.setData([])
        self._image_shape = None
        self._plot.setTitle('B-Scan图像')

    def apply_theme(self, dark: bool) -> None:
        """深色 bg 'k'/文字 'w'；浅色 bg 'w'/文字 'k'；轴 pen/textPen/标签同步。"""
        bg = 'k' if dark else 'w'
        fg = 'w' if dark else 'k'
        self._glw.setBackground(bg)
        pen = pg.mkPen(QColor(fg))
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
