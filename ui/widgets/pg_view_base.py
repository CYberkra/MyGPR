# -*- coding: utf-8 -*-
"""pyqtgraph 视图基类与轴主题帮助（UI 收敛轮）。

收敛 4 份逐字雷同的轴主题循环（bscan/ascan/depth_slice/spatial 高程剖面）到
:func:`style_plot_item`；:class:`GraphicsViewBase` mixin 统一缩放步长
（原 bscan 1.2 / map 0.75 → 1.25）、PNG 导出（时间戳文件名 + getSaveFileName
+ grab）、主题骨架（轴样式 + 控件调色板钩子）与右键菜单标准项。

mixin 设计：各视图的 Qt 基类不同（QWidget / GraphicsLayoutWidget /
PlotWidget），故以多继承 mixin 提供，约定子类持有 ``self._plot_item``
（pg.PlotItem）供缩放与轴样式，必要时覆写 ``_fit_view()`` /
``_export_grab_target()`` / ``_refresh_control_palette(dark)``。
"""
from __future__ import annotations

import re

import pyqtgraph as pg
from PyQt6.QtCore import QDateTime
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon as FIF

from ui import constants, file_dialogs
from ui.theme_helpers import ui_font
from ui.widgets.context_menus import add_action

__all__ = ['ZOOM_STEP', 'style_plot_item', 'GraphicsViewBase']

# 统一缩放步长：zoom_in 视野收窄 1/1.25（"放大 25%"），zoom_out 反之。
ZOOM_STEP = 1.25


def style_plot_item(plot_item: pg.PlotItem, dark: bool, *,
                    colorbar_axis=None, grid: bool = False) -> str:
    """统一轴主题循环：pen/textPen/轴标签/标题同步深浅色，返回前景色名。

    深色 'k' 底 'w' 字 / 浅色 'w' 底 'k' 字（SPEC §1）。轴标题（如
    道数/采样点）是独立 label，不随 textPen 变色，需显式同步；标题
    （PlotItem titleLabel）同理。

    pyqtgraph 默认 top/right 轴本就隐藏，这里不动；统一的是刻度字体
    （默认走 Qt 通用字体，中英文轴标签与界面字体不一致）、刻度朝内
    且轴线止于首尾刻度（默认朝外且两端悬空）。

    :param colorbar_axis: BScanView 色标轴（ColorBarItem.axis）一并同步。
    :param grid: 是否叠加淡网格。仅曲线类视图（AScan / 高程剖面）适用；
        图像类（B-Scan / 深度切片）套网格会盖住数据，保持默认 False。
    :return: 前景色名 'w'/'k'，供调用方同步曲线/图例等颜色。

    注意不能用 ``QColor(fg)``：Qt 颜色名不含 'w'/'k'，非法色会变黑，
    深色主题下轴字不可见；pg.mkPen 支持 'w'/'k' 简写。
    """
    fg = 'w' if dark else 'k'
    pen = pg.mkPen(fg)
    tick_font = ui_font(constants.CHART_TICK_FONT_SIZE)
    for name in ('bottom', 'left'):
        axis = plot_item.getAxis(name)
        axis.setPen(pen)
        axis.setTextPen(pen)
        axis.setTickFont(tick_font)
        axis.setStyle(tickLength=constants.CHART_TICK_LENGTH,
                      stopAxisAtTick=(True, True))
        axis.setLabel(text=axis.labelText, color=fg)
    if grid:
        plot_item.showGrid(
            x=True, y=True,
            alpha=(constants.CHART_GRID_ALPHA_DARK if dark
                   else constants.CHART_GRID_ALPHA_LIGHT))
    title_item = plot_item.titleLabel
    if title_item is not None:
        title_item.setText(title_item.text, color=fg)
    if colorbar_axis is not None:
        colorbar_axis.setPen(pen)
        colorbar_axis.setTextPen(pen)
        if getattr(colorbar_axis, 'labelText', ''):
            colorbar_axis.setLabel(text=colorbar_axis.labelText, color=fg)
    return fg


class GraphicsViewBase:
    """pyqtgraph 视图通用行为 mixin（缩放 / PNG 导出 / 主题 / 右键菜单骨架）。

    子类约定：
    - ``self._plot_item``：pg.PlotItem（缩放与轴样式用）；
    - 可选覆写 ``_fit_view()`` 自适应实现（默认 ViewBox.autoRange）；
    - 可选覆写 ``_export_grab_target()`` 指定 PNG 抓取目标（默认 self）；
    - 可选覆写 ``_refresh_control_palette(dark)`` 刷工具条等控件配色
      （由 :meth:`apply_theme` 骨架调用）。
    """

    # ---------------------------------------------------------------- 缩放
    def zoom_in(self) -> None:
        """放大（统一步长 1.25）。"""
        self._plot_item.vb.scaleBy((ZOOM_STEP, ZOOM_STEP))

    def zoom_out(self) -> None:
        """缩小（统一步长 1.25）。"""
        inverse = 1.0 / ZOOM_STEP
        self._plot_item.vb.scaleBy((inverse, inverse))

    def zoom_fit(self) -> None:
        """自适应视野（各视图覆写 _fit_view 实现专属语义）。"""
        self._fit_view()

    def _fit_view(self) -> None:
        self._plot_item.vb.autoRange()

    # ---------------------------------------------------------------- 导出
    def _export_grab_target(self):
        """PNG 抓取目标（3D 视图等可覆写；grabFramebuffer 特例留在其自身类）。"""
        return self

    def export_png(self, *, title: str = '', prefix: str = 'view') -> None:
        """视图内容导出 PNG：时间戳默认文件名 + getSaveFileName + grab。"""
        stamp = QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')
        safe = re.sub(r'[\\/:*?"<>|\s]+', '_', str(title or '')).strip('_')
        default = (f'{prefix}_{safe}_{stamp}.png' if safe
                   else f'{prefix}_{stamp}.png')
        path, _selected = file_dialogs.getSaveFileName(
            self, '导出 PNG 图像', default, 'PNG 图片 (*.png)')
        if path:
            self._export_grab_target().grab().save(path, 'PNG')

    def _copy_image(self) -> None:
        """视图内容复制到剪贴板。"""
        QApplication.clipboard().setPixmap(
            self._export_grab_target().grab())

    # ---------------------------------------------------------------- 主题
    def apply_theme(self, dark: bool) -> None:
        """主题骨架：轴样式 + 控件调色板钩子；子类按需扩展（背景/曲线等）。"""
        self._dark = bool(dark)
        style_plot_item(self._plot_item, dark)
        refresh = getattr(self, '_refresh_control_palette', None)
        if callable(refresh):
            refresh(dark)

    # ---------------------------------------------------------------- 右键菜单
    def _add_standard_menu_actions(self, menu, *, can_export: bool = True,
                                   fit_text: str = '自适应',
                                   export_prefix: str = 'view') -> None:
        """右键菜单标准项：自适应 / 复制图像 / 导出 PNG…（各子类补充专属项）。"""
        add_action(menu, FIF.FIT_PAGE, fit_text, self.zoom_fit)
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制图像', self._copy_image,
                   enabled=can_export)
        add_action(menu, FIF.SAVE, '导出 PNG…',
                   lambda: self.export_png(prefix=export_prefix),
                   enabled=can_export)
