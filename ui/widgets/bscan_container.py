# -*- coding: utf-8 -*-
"""BScanContainer — B-Scan 多视图容器（Phase 2）。

在同一个预览卡里按布局模式摆放 1 / 2 / 4 个 :class:`BScanView`：

- ``auto``：**默认**，面板数量自动跟随数据（1 份 bundle → 单视图，2 份 →
  左右对比；由宿主页面经 :meth:`resolve_auto` 喂当前应有面板数）；
- ``single``：固定单视图（等价于历史上的单 BScanView）；
- ``dual``：固定左右并排双视图（处理页「原始 | 成果」同屏对比）；
- ``quad``：固定 2×2 四宫格（0/1 号位与 dual 相同，2/3 号位留空占位，
  后续接入历史成果对比；auto 模式因数据源不足不会自动触发）。

职责边界（哑组件）：

- 只管「有几个面板、怎么摆、切换时发信号」；**数据路由**（哪个 bundle 进
  哪个面板）由宿主页面实现——容器不认识 PreviewBundle；
- 每个面板的轴模式 / 显示比例 / 色阶 / 全屏几何等偏好持久化，由主窗口
  ``_wire_bscan_preferences`` 经 ``findChildren(BScanView)`` 统一覆盖，
  容器不重复做；主题同理（BScanView 构造自刷 + 全局换肤覆盖）。
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QStackedWidget,
                             QVBoxLayout, QWidget)

from ui import constants
from ui.widgets.bscan_view import BScanView

LAYOUT_AUTO = 'auto'
LAYOUT_SINGLE = 'single'
LAYOUT_DUAL = 'dual'
LAYOUT_QUAD = 'quad'
# auto 是"策略"（面板数随数据解析），single/dual/quad 是"实体布局"；
# 设置层五选一（含 auto），容器实际摆哪一页由 _effective 决定。
LAYOUT_MODES = (LAYOUT_AUTO, LAYOUT_SINGLE, LAYOUT_DUAL, LAYOUT_QUAD)
LAYOUT_ENTITY_MODES = (LAYOUT_SINGLE, LAYOUT_DUAL, LAYOUT_QUAD)

_MODE_INDEX = {LAYOUT_SINGLE: 0, LAYOUT_DUAL: 1, LAYOUT_QUAD: 2}


class BScanContainer(QWidget):
    """B-Scan 多面板容器：布局切换 + 面板访问，不做数据路由。"""

    sig_layout_changed = pyqtSignal(str)     # 用户/设置切换布局模式

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = LAYOUT_AUTO             # 用户/设置层的偏好值
        self._effective = LAYOUT_SINGLE      # auto 解析后实际摆的页
        self._pages = {}                     # entity mode -> list[BScanView]

        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._build_flow_page(LAYOUT_SINGLE, 1))
        self._stack.addWidget(self._build_flow_page(LAYOUT_DUAL, 2))
        self._stack.addWidget(self._build_quad_page())
        self._stack.setCurrentIndex(_MODE_INDEX[self._effective])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

    # ------------------------------------------------------------ 页面构建
    def _build_flow_page(self, mode: str, count: int) -> QWidget:
        """single（纵向 1 个）与 dual（横向 2 个）共用：顺序摆放。"""
        page = QWidget(self)
        flow = (QHBoxLayout(page) if mode == LAYOUT_DUAL
                else QVBoxLayout(page))
        flow.setContentsMargins(0, 0, 0, 0)
        flow.setSpacing(constants.PANEL_SPACING)
        views = []
        for _ in range(count):
            view = BScanView(page)
            flow.addWidget(view, 1)
            views.append(view)
        self._pages[mode] = views
        return page

    def _build_quad_page(self) -> QWidget:
        """四宫格 2×2。"""
        page = QWidget(self)
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(constants.PANEL_SPACING)
        views = []
        for row in range(2):
            for col in range(2):
                view = BScanView(page)
                grid.addWidget(view, row, col)
                views.append(view)
        self._pages[LAYOUT_QUAD] = views
        return page

    # ------------------------------------------------------------ 布局模式
    def layout_mode(self) -> str:
        """设置层的偏好值（可能是 auto；实体布局见 effective_mode）。"""
        return self._mode

    def effective_mode(self) -> str:
        """当前实际摆的实体布局页（auto 解析后的结果）。"""
        return self._effective

    def set_layout_mode(self, mode: str, *, notify: bool = True) -> None:
        """切换布局模式；非法值回落 auto（坏设置不该让预览空白）。

        auto 本身不决定摆哪页（保持当前 _effective，等宿主 resolve_auto）；
        实体模式直接切换。持久化镜像走 sig_layout_changed——注意 auto 的
        resolve_auto **不发**本信号：数据驱动的重排不是用户偏好变化，
        若发出去会被主窗口把 'dual' 之类写回设置、破坏 auto 偏好。
        """
        mode = str(mode or LAYOUT_AUTO)
        if mode not in LAYOUT_MODES:
            mode = LAYOUT_AUTO
        if mode == self._mode:
            return
        self._mode = mode
        if mode in _MODE_INDEX:
            self._effective = mode
            self._stack.setCurrentIndex(_MODE_INDEX[mode])
        if notify:
            self.sig_layout_changed.emit(mode)

    def resolve_auto(self, panel_count: int) -> bool:
        """auto 模式下按数据量解析实际布局；返回是否换了页。

        1 → single；2 → dual；3 及以上 → quad（当前数据源最多 2 份，
        quad 仅留接口）。非 auto 模式调用是无害空操作。
        """
        if self._mode != LAYOUT_AUTO:
            return False
        if panel_count <= 1:
            wanted = LAYOUT_SINGLE
        elif panel_count == 2:
            wanted = LAYOUT_DUAL
        else:
            wanted = LAYOUT_QUAD
        if wanted == self._effective:
            return False
        self._effective = wanted
        self._stack.setCurrentIndex(_MODE_INDEX[wanted])
        return True

    # ------------------------------------------------------------ 面板访问
    def all_views(self) -> list:
        """全部面板（含当前未显示布局页里的——信号接线用，跨页常驻）。"""
        out = []
        for views in self._pages.values():
            out.extend(views)
        return out

    def views(self) -> list:
        """当前实际布局下的面板列表（新列表，调用方可放心改）。"""
        return list(self._pages[self._effective])

    def view_at(self, index: int) -> BScanView:
        """第 index 个面板；越界回落主面板（只读容错，写路径禁用）。"""
        pages = self._pages[self._effective]
        return pages[index] if 0 <= index < len(pages) else pages[0]

    def primary_view(self) -> BScanView:
        """主面板（0 号位）——single 模式下即唯一面板。"""
        return self._pages[self._effective][0]
