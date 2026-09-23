# -*- coding: utf-8 -*-
"""BScanContainer — B-Scan 多视图容器（Phase 2）。

在同一个预览卡里按布局模式摆放 1 / 2 / 4 个 :class:`BScanView`：

- ``auto``：**默认**，面板数量自动跟随数据（1 份 bundle → 单视图，2 份 →
  左右对比；由宿主页面经 :meth:`resolve_auto` 喂当前应有面板数）。布局
  稳定优先：粘性策略（对比长出后不随成果清空收回）由宿主实现，见
  ``ProcessingPage._sync_auto_layout``——容器保持哑组件，只认喂进来的
  面板数；
- ``single``：固定单视图（等价于历史上的单 BScanView）；
- ``dual``：固定左右并排双视图（处理页「原始 | 成果」同屏对比）；
- ``quad``：固定 2×2 四宫格（0/1 号位与 dual 相同，2/3 号位留空占位，
  后续接入历史成果对比；auto 模式因数据源不足不会自动触发）；
- ``free``：**自由分屏**（QSplitter，2026-09-23 由 QMdiArea 迁移而来）——
  两个画布占满全部空间、拖中间分割条调占比，右键可重置；占比跨会话
  记忆（宿主经 :meth:`set_split_state_store` 注入读写回调）。格位固定
  0=原始数据、1=处理结果；单图放大走 BScanView 已有的独立全屏。手动
  模式，不参与 auto 解析。选型理由：MDI 子窗标题栏/边框每窗吃 ~24px
  且拖出一套关闭拦截/平铺时机的防御逻辑，而"叠放窗口"对剖面对比无
  实际意义（叠住的图不可读）。

职责边界（哑组件）：

- 只管「有几个面板、怎么摆、切换时发信号」；**数据路由**（哪个 bundle 进
  哪个面板）由宿主页面实现——容器不认识 PreviewBundle；
- 每个面板的轴模式 / 显示比例 / 色阶 / 全屏几何等偏好持久化，由主窗口
  ``_wire_bscan_preferences`` 经 ``findChildren(BScanView)`` 统一覆盖，
  容器不重复做；主题同理（BScanView 构造自刷 + 全局换肤覆盖）。
"""

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QMenu, QSplitter,
                             QStackedWidget, QVBoxLayout, QWidget)

from ui import constants
from ui.widgets.bscan_view import BScanView

LAYOUT_AUTO = 'auto'
LAYOUT_SINGLE = 'single'
LAYOUT_DUAL = 'dual'
LAYOUT_QUAD = 'quad'
LAYOUT_FREE = 'free'
# auto 是"策略"（面板数随数据解析），single/dual/quad/free 是"实体布局"；
# 设置层六选一（含 auto），容器实际摆哪一页由 _effective 决定。
LAYOUT_MODES = (LAYOUT_AUTO, LAYOUT_SINGLE, LAYOUT_DUAL, LAYOUT_QUAD,
                LAYOUT_FREE)
LAYOUT_ENTITY_MODES = (LAYOUT_SINGLE, LAYOUT_DUAL, LAYOUT_QUAD, LAYOUT_FREE)

_MODE_INDEX = {LAYOUT_SINGLE: 0, LAYOUT_DUAL: 1, LAYOUT_QUAD: 2, LAYOUT_FREE: 3}


class BScanContainer(QWidget):
    """B-Scan 多面板容器：布局切换 + 面板访问，不做数据路由。"""

    sig_layout_changed = pyqtSignal(str)     # 用户/设置切换布局模式

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = LAYOUT_AUTO             # 用户/设置层的偏好值
        self._effective = LAYOUT_SINGLE      # auto 解析后实际摆的页
        self._pages = {}                     # entity mode -> list[BScanView]
        # 分屏占比持久化回调（主窗口经 set_split_state_store 注入；
        # 提前初始化防接线前 splitterMoved 落到未定义属性）。
        self._split_loader = None
        self._split_saver = None
        # 当前认可的分屏比例（千分比整数，[750, 250] = 3:1）。QSplitter
        # 自身 resize 时按 stretch 因子重排（默认全 0 → 均分），会丢掉
        # 用户拖出的占比——由 eventFilter 在每次 resize 时重放本值。
        self._split_ratio = [1, 1]

        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._build_flow_page(LAYOUT_SINGLE, 1))
        self._stack.addWidget(self._build_flow_page(LAYOUT_DUAL, 2))
        self._stack.addWidget(self._build_quad_page())
        self._stack.addWidget(self._build_free_page())
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

    def _build_free_page(self) -> QWidget:
        """自由分屏页：QSplitter 承载两个常驻画布，拖分割条调占比。

        格位固定 0=原始数据、1=处理结果（与 dual 分发语义一致，views()
        契约依赖该顺序）；childrenCollapsible(False) 保证拖到头也不把
        一格挤没。旧 QMdiArea 方案（标题栏/边框 + 关闭拦截 + 平铺时机
        守卫）已退役——单图放大由 BScanView 的独立全屏承担。
        """
        page = QWidget(self)
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        self._splitter = QSplitter(Qt.Orientation.Horizontal, page)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._splitter.customContextMenuRequested.connect(
            self._show_free_menu)
        self._splitter.installEventFilter(self)   # Resize → 重放认可比例
        views = []
        for _ in range(2):
            view = BScanView(self._splitter)
            self._splitter.addWidget(view)
            views.append(view)
        self._splitter.setSizes([1, 1])    # setSizes 是相对权重：即均分
        self._splitter.splitterMoved.connect(self._on_split_moved)
        outer.addWidget(self._splitter)
        self._pages[LAYOUT_FREE] = views
        return page

    def eventFilter(self, obj, event) -> bool:
        """分屏区真实宽度变化 → 重放认可比例（见 __init__ 的 _split_ratio）。

        QSplitter resize 默认按 stretch 因子重排（全 0 → 均分），restoreState
        的绝对尺寸同样会被重排冲掉（offscreen 实测）。每次 resize 重放当前
        认可比例：用户拖动更新认可值，程序性 setSizes 不发 splitterMoved，
        无回环。
        """
        if obj is self._splitter and event.type() == QEvent.Type.Resize:
            self._apply_split_ratio()
        return super().eventFilter(obj, event)

    # ------------------------------------------------ 自由分屏占比控制
    def _show_free_menu(self, pos) -> None:
        """自由分屏区右键菜单：重置占比（回到均分）。"""
        menu = QMenu(self._splitter)
        menu.addAction('重置分屏占比', self.reset_free_split)
        menu.exec(self._splitter.mapToGlobal(pos))

    def reset_free_split(self) -> None:
        """重置分屏占比：两格均分（与新占比一同持久化）。"""
        self._split_ratio = [1, 1]
        self._apply_split_ratio()
        self._save_split_ratio()

    def _apply_split_ratio(self) -> None:
        """按当前分屏区宽度把认可比例换算成绝对尺寸下发。

        QSplitter.setSizes 的值是 **sizeHint** 语义（qGeomCalc）：sum <
        实际长度时多余空间被 stretch（全 0 → 均分）吃掉——实测下发自
        [3, 1] 会得到均分；sum ≥ 长度才按 hint 比例分配。故必须按当前
        长度换算后再下发。
        """
        sp = self._splitter
        length = (sp.width() if sp.orientation() == Qt.Orientation.Horizontal
                  else sp.height())
        ratio = self._split_ratio
        denom = sum(ratio) or 1
        sp.setSizes([max(1, round(w * length / denom)) for w in ratio])

    def set_split_state_store(self, loader=None, saver=None) -> None:
        """注入分屏占比持久化回调（主窗口接线；容器不碰文件/设置）。

        占比格式为千分比整数文本（``'750,250'`` = 3:1）。

        :param loader: ``() -> str | None``，回放历史占比；
        :param saver:  ``(str) -> None``，占比变化（拖动/重置）时保存。
        """
        self._split_loader = loader
        self._split_saver = saver

    def restore_free_split(self) -> bool:
        """回放历史分屏占比（主窗口接线时调用一次）。是否恢复成功。

        恢复只更新认可比例：分屏区可见时立即生效；启动期藏在
        QStackedWidget 里则由首次真实 resize 经 eventFilter 重放——
        无需专门的显示时机守卫。
        """
        loader = self._split_loader
        if loader is None:
            return False
        try:
            raw = loader()
        except Exception:  # noqa: BLE001 - 坏设置不让预览空白
            return False
        parts = str(raw or '').replace(' ', '').split(',')
        if len(parts) != 2:
            return False
        try:
            w0, w1 = int(float(parts[0])), int(float(parts[1]))
        except (ValueError, OverflowError):
            return False
        if w0 <= 0 or w1 <= 0:
            return False
        self._split_ratio = [w0, w1]
        self._apply_split_ratio()
        return True

    def _on_split_moved(self, _pos: int, _index: int) -> None:
        """用户拖动分割条（程序性 setSizes/重放不发此信号）。"""
        sizes = self._splitter.sizes()
        total = sum(sizes)
        if len(sizes) != 2 or total <= 0:
            return
        self._split_ratio = [max(1, round(sizes[0] * 1000 / total)),
                             max(1, round(sizes[1] * 1000 / total))]
        self._save_split_ratio()

    def _save_split_ratio(self) -> None:
        if self._split_saver is None:
            return
        try:
            ratio = self._split_ratio
            self._split_saver(f'{ratio[0]},{ratio[1]}')
        except Exception:  # noqa: BLE001 - 存占比失败不该报错给用户
            pass

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
