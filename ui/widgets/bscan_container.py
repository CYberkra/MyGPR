# -*- coding: utf-8 -*-
"""BScanContainer — B-Scan 多视图容器（Phase 2）。

在同一个预览卡里按面板数摆放 1 / 2 / 3 / 4 个 :class:`BScanView`：

- 面板数由宿主页面经 :meth:`resolve_auto` 喂入（tab 模型：打开的数据源
  数 = 面板数，1→单窗、2→左右、3-4→2×2；>4 主区留前 4，其余进总览墙）；
  quad 页在 n=3 时隐藏 4 号面板。容器保持哑组件，只认喂进来的面板数；
- ``single``/``dual``/``quad``：预建实体页（历史上由布局档位手动选择，
  2026-09-24 起 auto=唯一策略，实体页仅作内部实现细节保留）。

职责边界（哑组件）：

- 只管「有几个面板、怎么摆」；**数据路由**（哪个 bundle 进哪个面板）由
  宿主页面实现——容器不认识 PreviewBundle；
- 每个面板的轴模式 / 显示比例 / 色阶 / 全屏几何等偏好持久化，由主窗口
  ``_wire_bscan_preferences`` 经 ``findChildren(BScanView)`` 统一覆盖，
  容器不重复做；主题同理（BScanView 构造自刷 + 全局换肤覆盖）。

历史注记：``free`` 自由分屏（QSplitter 拖占比）于 2026-09-24 随 tab 模型
上线退役——窗口数自动排布后占比手动调整失去对象。
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from ui import constants
from ui.widgets.bscan_view import BScanView

LAYOUT_AUTO = 'auto'
LAYOUT_SINGLE = 'single'
LAYOUT_DUAL = 'dual'
LAYOUT_QUAD = 'quad'
# auto 是唯一策略（面板数随 tab 数解析），single/dual/quad 是内部实体页。
LAYOUT_MODES = (LAYOUT_AUTO, LAYOUT_SINGLE, LAYOUT_DUAL, LAYOUT_QUAD)

_MODE_INDEX = {LAYOUT_SINGLE: 0, LAYOUT_DUAL: 1, LAYOUT_QUAD: 2}
MAX_PANELS = 4


class BScanContainer(QWidget):
    """B-Scan 多面板容器：面板数切换 + 面板访问，不做数据路由。"""

    sig_layout_changed = pyqtSignal(str)     # 面板数变化时发射（实体页名）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._effective = LAYOUT_SINGLE      # 当前实际摆的页
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

    # （free 自由分屏已于 2026-09-24 随 tab 模型退役：窗口数自动排布后，
    #  拖占比失去对象——详见模块 docstring 历史注记。）

    # ------------------------------------------------------------ 面板数
    def effective_mode(self) -> str:
        """当前实际摆的实体布局页（面板数解析后的结果）。"""
        return self._effective

    def resolve_auto(self, panel_count: int) -> bool:
        """按打开的数据源数解析实际布局；返回是否换了页/可见性。

        1 → single；2 → dual；3-4 → quad（quad 在 n=3 时隐藏 4 号面板）。
        panel_count 由宿主按 tab 数喂入，夹取到 [1, MAX_PANELS]。
        """
        n = max(1, min(int(panel_count), MAX_PANELS))
        if n <= 1:
            wanted = LAYOUT_SINGLE
        elif n == 2:
            wanted = LAYOUT_DUAL
        else:
            wanted = LAYOUT_QUAD
        changed = wanted != self._effective
        if changed:
            self._effective = wanted
            self._stack.setCurrentIndex(_MODE_INDEX[wanted])
        # quad 页可见面板数（n=3 → 隐藏 4 号；n=4 → 全显）
        if wanted == LAYOUT_QUAD:
            quad_views = self._pages[LAYOUT_QUAD]
            for index, view in enumerate(quad_views):
                view.setVisible(index < n)
        return changed

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
