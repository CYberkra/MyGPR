# -*- coding: utf-8 -*-
"""ResultGrid — 结果网格（处理页 v2 主区下半）。

设计（2026-09-27 v2.3，参考用户设计稿定稿）：
- **两列大图**：``N=1 → 全幅``，``N≥2 → 2 列``；
- **三种视图模式**（结果区头部切换，选中卡驱动）：
  - ``all`` 全部步骤：现有网格；
  - ``compare`` 前后对比：相邻两张（选中步与其前一步，选中输入则为
    输入+第 1 步）并排大图；
  - ``single`` 单步结果：只显示选中卡全幅大图；
- 网格卡**无色标**（``BScanView(with_colorbar=False)``）；
- **统一色标**开关：开启时全组结果共用同一 [vmin, vmax]（经
  ``BScanView.set_levels_override``，display 层覆盖，raw 不动）——
  增益前后横向可比；
- **全部步骤**开关：关闭时只铺「输入 + 最终结果」（P3 设置的就地版），
  经 :attr:`sig_expand_all_changed` 通知宿主过滤槽位；
- 单元间距 16（8pt 栅格），单元最小高 320（single/compare 大图档 460）；
- 卡头极简：序号 + 算法名 + 幅值范围，⤢ hover 才显；
- **选中同步高亮**：点卡 → :attr:`sig_card_selected`；宿主调
  :meth:`set_selected`（与链条 chip 双向同步）；
- 禁用步骤 → 虚线占位卡（保留 1:1 对应，不占画布）。

网格只认「槽位」（key/title/enabled），数据由宿主页按 key 回填。
"""
import math

from PyQt6.QtCore import (QPropertyAnimation, Qt, pyqtSignal)
from PyQt6.QtWidgets import (QFrame, QGraphicsOpacityEffect,
                             QGridLayout, QHBoxLayout, QLabel, QScrollArea,
                             QSizePolicy, QVBoxLayout, QWidget)
from qfluentwidgets import (CaptionLabel, FluentIcon as FIF, SegmentedWidget,
                            SwitchButton, ToolButton)

from ui.widgets.bscan_view import BScanView
from ui.widgets.empty_state import EmptyStateOverlay

_CELL_MIN_HEIGHT = 320
_CELL_MIN_HEIGHT_LARGE = 460
_GRID_SPACING = 16
_MAX_COLUMNS = 2

VIEW_ALL = 'all'
VIEW_COMPARE = 'compare'
VIEW_SINGLE = 'single'


class _Skeleton(QWidget):
    """结果卡骨架：预览未回填时的微光占位（动效移植①）。

    移植 Transitions.dev 的「Skeleton loader and reveal」思路——Qt 侧用
    不透明度脉冲 + 横向扫光条（QPropertyAnimation）近似 shimmer，不引入
    新依赖；出图后与画布做交叉淡入（skeleton 淡出 / 画布淡入）。
    """

    def __init__(self, host=None):
        super().__init__(host)
        from PyQt6.QtCore import QPropertyAnimation, QRect
        from PyQt6.QtGui import QColor, QPainter
        self._QPropertyAnimation = QPropertyAnimation
        self._QRect = QRect
        self._QColor = QColor
        self._QPainter = QPainter
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._shimmer_x = -0.35
        self._anim = QPropertyAnimation(self, b'geometry')
        self._anim.setDuration(1400)
        self._anim.setLoopCount(-1)
        self._anim.setStartValue(0)
        self._anim.setEndValue(1000)
        self._anim.valueChanged.connect(lambda _v: self.update())

    def showEvent(self, event) -> None:
        if self._anim.state() != self._anim.State.Running:
            self._anim.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._anim.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:
        painter = self._QPainter(self)
        painter.setRenderHint(self._QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, self._QColor(150, 150, 150, 26))
        # 扫光条：按动画进度在卡宽上平移
        progress = (self._anim.currentValue() or 0) / 1000.0
        width = max(rect.width() * 0.35, 40)
        x = rect.left() + (rect.width() + width) * progress - width
        band = self._QRect(int(x), rect.top(), int(width), rect.height())
        painter.fillRect(band, self._QColor(200, 200, 200, 34))
        painter.end()


class _ResultCard(QFrame):
    """单张结果：极简卡头（序号 + 算法名）+ B-Scan + hover 操作钮。"""

    sig_expand_requested = pyqtSignal(str)   # slot key
    sig_compare_requested = pyqtSignal(str)  # slot key（P2 接线）
    sig_clicked = pyqtSignal(str)            # 点卡 → 宿主选中对应步骤

    _SEL_QSS = ('#resultCard{border:2px solid rgba(90,156,216,0.85);'
                'border-radius:6px}')
    _NORMAL_QSS = '#resultCard{border:2px solid transparent;border-radius:6px}'

    def __init__(self, key: str, title: str, *, placeholder: bool = False,
                 parent=None):
        super().__init__(parent)
        self.key = key
        self._placeholder = placeholder
        self.setObjectName('resultCard')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(self._NORMAL_QSS)
        self.setMinimumHeight(_CELL_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            '#resultCard{background:transparent}' if not placeholder else
            '#resultCard{border:1px dashed rgba(140,140,140,0.55);'
            'border-radius:6px}')

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        self.title_label = QLabel(title, self)
        self.title_label.setStyleSheet(
            'color:#8A8A85' if placeholder else '')
        head.addWidget(self.title_label)
        self.range_label = QLabel('', self)
        self.range_label.setStyleSheet('color:#8A8A85')
        self.range_label.setVisible(False)
        head.addWidget(self.range_label)
        head.addStretch(1)
        self.expand_btn = ToolButton(FIF.FULL_SCREEN, self)
        self.expand_btn.setFixedSize(20, 20)
        self.expand_btn.setToolTip('放大 / 全屏浏览该结果')
        self.expand_btn.clicked.connect(
            lambda: self.sig_expand_requested.emit(self.key))
        head.addWidget(self.expand_btn)
        self.compare_btn = ToolButton(FIF.VIEW, self)
        self.compare_btn.setFixedSize(20, 20)
        self.compare_btn.setToolTip('与另一张结果并排对比（P2 接线）')
        self.compare_btn.clicked.connect(
            lambda: self.sig_compare_requested.emit(self.key))
        head.addWidget(self.compare_btn)
        self.expand_btn.setVisible(False)
        self.compare_btn.setVisible(False)

        body = QVBoxLayout(self)
        body.setContentsMargins(4, 2, 4, 4)
        body.setSpacing(4)
        body.addLayout(head)
        if placeholder:
            self.view = None
            hint = QLabel('已禁用（未参与本次运行）', self)
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setStyleSheet('color:#8A8A85')
            body.addWidget(hint, 1)
        else:
            # 网格卡无色标：坐标/色标不挤占绘图区（看色标走放大/全屏）
            self.view = BScanView(self, with_colorbar=False)
            self._skeleton = _Skeleton(self.view)
            self._skeleton.setGeometry(self.view.rect())
            self._skeleton.show()          # 建卡即占位：等预览回填
            # 画布透明度效果（出图时淡入，与骨架交叉）
            self._view_effect = QGraphicsOpacityEffect(self.view)
            self._view_effect.setOpacity(0.0)
            self.view.setGraphicsEffect(self._view_effect)
            body.addWidget(self.view, 1)
            self._fade_out = QPropertyAnimation(self._skeleton, b'windowOpacity')
            self._fade_out.setDuration(240)
            self._fade_out.setStartValue(1.0)
            self._fade_out.setEndValue(0.0)
            self._fade_out.finished.connect(self._skeleton.hide)
            self._fade_in = QPropertyAnimation(self._view_effect, b'opacity')
            self._fade_in.setDuration(240)
            self._fade_in.setStartValue(0.0)
            self._fade_in.setEndValue(1.0)

    def set_bundle(self, bundle) -> None:
        if self.view is None:
            return
        if bundle is None:
            self.view.clear()       # 尚无输入数据 → 空态（骨架继续占位）
            self.range_label.setVisible(False)
            return
        self.view.set_bundle(bundle)
        vmin = float(getattr(bundle, 'vmin', 0.0) or 0.0)
        vmax = float(getattr(bundle, 'vmax', 0.0) or 0.0)
        self.range_label.setText(f'{vmin:.3g} ~ {vmax:.3g}')
        self.range_label.setVisible(True)
        self._reveal()

    def _reveal(self) -> None:
        """出图：骨架淡出 + 画布淡入（交叉淡入，避免「空白→突然有图」）。"""
        if self._skeleton.isHidden():
            return
        self._fade_out.start()
        self._fade_in.start()

    def clear(self) -> None:
        if self.view is not None:
            self.view.clear()

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def set_selected(self, selected: bool) -> None:
        """选中高亮：与链条 chip 同步（描边 2px 主题蓝）。"""
        self.setStyleSheet(self._SEL_QSS if selected else self._NORMAL_QSS)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.sig_clicked.emit(self.key)
        super().mousePressEvent(event)

    def enterEvent(self, event) -> None:
        if not self._placeholder:
            self.expand_btn.setVisible(True)
            self.compare_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.expand_btn.setVisible(False)
        self.compare_btn.setVisible(False)
        super().leaveEvent(event)


class ResultGrid(QWidget):
    """结果网格：按槽位铺卡片，列数自适应，纵向可滚；支持三种视图模式。"""

    sig_card_selected = pyqtSignal(str)      # 点卡 → 宿主选中对应步骤
    sig_expand_all_changed = pyqtSignal(bool)  # 「全部步骤」开关 → 宿主过滤槽位

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards = []                 # 顺序 = 槽位顺序
        self._keys = {}                  # key → card
        self._bundles = {}               # key → 原始 bundle（统一色标的全局范围原料）
        self._view_mode = VIEW_ALL
        self._selected_key = None
        self._shared_scale = False
        self._body = QWidget(self)
        self._grid = QGridLayout(self._body)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(_GRID_SPACING)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # ---- 结果区头部：幅数 + 视图模式 + 统一色标/全部步骤开关 ----
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(10)
        self._count_label = CaptionLabel('结果', self)
        head.addWidget(self._count_label)
        self._mode_seg = SegmentedWidget(self)
        for key, text in ((VIEW_ALL, '全部步骤'), (VIEW_COMPARE, '前后对比'),
                          (VIEW_SINGLE, '单步结果')):
            self._mode_seg.addItem(routeKey=key, text=text)
        self._mode_seg.setCurrentItem(VIEW_ALL)
        self._mode_seg.currentItemChanged.connect(self.set_view_mode)
        head.addWidget(self._mode_seg)
        head.addStretch(1)
        self._scale_switch = SwitchButton('统一色标', self)
        self._scale_switch.setChecked(False)
        self._scale_switch.checkedChanged.connect(self.set_shared_scale)
        head.addWidget(self._scale_switch)
        self._expand_switch = SwitchButton('全部步骤', self)
        self._expand_switch.setChecked(True)
        self._expand_switch.checkedChanged.connect(self.sig_expand_all_changed)
        head.addWidget(self._expand_switch)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addLayout(head)
        outer.addWidget(self._scroll)
        self._scroll.setWidget(self._body)

        # 运行前不摆空画布：只给一句引导（有数据才建 B-Scan 视图）
        self._empty = EmptyStateOverlay(
            self, icon=None, title='暂无结果',
            hint='选择测线后显示输入数据；点「运行」后按步骤显示各步结果')
        self._empty.setVisible(True)

    # ---------------------------------------------------------------- 槽位
    def set_slots(self, slots) -> None:
        """slots: [{key, title, enabled}]（宿主页为数据源；这里只铺卡）。

        **差量更新**：key 与占位态都未变的卡**原地复用**（bundle 与视图
        状态保留）——整排重建会让已有画面清空→骨架重现，用户可感为
        「原始 B-Scan 消失几秒」（真机反馈，2026-09-26）。
        """
        new_cards: list = []
        reused = set()
        for spec in (slots or []):
            key = spec['key']
            placeholder = not bool(spec.get('enabled', True))
            card = self._keys.get(key)
            if card is not None and card._placeholder == placeholder:
                card.set_title(spec.get('title', ''))
                reused.add(id(card))
            else:
                card = _ResultCard(
                    key, spec.get('title', ''), placeholder=placeholder,
                    parent=self._body)
                card.sig_clicked.connect(self.sig_card_selected)
            new_cards.append(card)
        # 移除不再存在的槽位卡
        keep = {id(c) for c in new_cards}
        for old in self._cards:
            if id(old) not in keep:
                self._grid.removeWidget(old)
                old.setParent(None)
                old.deleteLater()
        self._cards = new_cards
        self._keys = {c.key: c for c in new_cards}
        # 同步 bundle 缓存：移除消失槽位（统一色标的全局范围只算现存的）
        live = set(self._keys)
        for stale in set(self._bundles) - live:
            del self._bundles[stale]
        self._count_label.setText(f'结果 {len(new_cards)} 幅')
        self._empty.setVisible(not self._cards)
        self._apply_view_mode()

    # ------------------------------------------------------------ 视图模式
    def set_view_mode(self, mode: str) -> None:
        """切换视图模式（all / compare / single）；重排可见卡。"""
        if mode not in (VIEW_ALL, VIEW_COMPARE, VIEW_SINGLE):
            return
        if mode == self._view_mode:
            return
        self._view_mode = mode
        # 程序化切换同步 segmented 选中态（已是当前项时信号不重发，无回环）
        self._mode_seg.setCurrentItem(mode)
        self._apply_view_mode()

    def view_mode(self) -> str:
        return self._view_mode

    def _visible_cards(self) -> list:
        """当前模式下应显示的卡（有序）；隐藏其余。"""
        cards = self._cards
        if self._view_mode == VIEW_SINGLE:
            sel = self._keys.get(self._selected_key)
            chosen = [sel] if sel is not None else cards[-1:]
        elif self._view_mode == VIEW_COMPARE:
            if len(cards) >= 2:
                index = self._index_of(self._selected_key)
                if index <= 0:            # 未选 / 选中输入 → 输入+第 1 步
                    pair = [cards[0], cards[1]]
                else:
                    pair = [cards[index - 1], cards[index]]
                chosen = pair
            else:
                chosen = list(cards)
        else:
            chosen = cards
        for card in cards:
            card.setVisible(card in chosen)
        return chosen

    def _index_of(self, key) -> int:
        if key is None:
            return -1
        for index, card in enumerate(self._cards):
            if card.key == key:
                return index
        return -1

    def _apply_view_mode(self) -> None:
        """按当前模式重排可见卡（single/compare 用大图档高度）。"""
        visible = self._visible_cards()
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    self._grid.removeWidget(widget)
        n = len(visible)
        # 列拉伸先全清：QGridLayout 的 setColumnStretch 会**跨重排残留**
        # （从两列切单步，空列仍按旧 stretch 分走一半宽——真机反馈
        # 「单步结果只是半个窗口」）
        for col in range(_MAX_COLUMNS + 1):
            self._grid.setColumnStretch(col, 0)
        if n == 0:
            return
        if self._view_mode == VIEW_ALL:
            cols = n if n <= _MAX_COLUMNS else min(
                int(math.ceil(math.sqrt(n))), _MAX_COLUMNS)
            min_h = _CELL_MIN_HEIGHT
        elif self._view_mode == VIEW_COMPARE:
            cols, min_h = 2, _CELL_MIN_HEIGHT_LARGE
        else:                                 # single：全幅
            cols, min_h = 1, _CELL_MIN_HEIGHT_LARGE
        for index, card in enumerate(visible):
            card.setMinimumHeight(min_h)
            if self._view_mode == VIEW_SINGLE:
                # 全幅：跨满最大列数——QGridLayout 的空列不会自动回收
                # （即使 stretch 清零仍参与分配，卡只占半窗，真机反馈）
                self._grid.addWidget(card, 0, 0, 1, _MAX_COLUMNS)
            else:
                self._grid.addWidget(card, index // cols, index % cols)
        # 列均分：QGridLayout 默认按 sizeHint 分配，compare 左卡会被挤扁
        for col in range(cols):
            self._grid.setColumnStretch(col, 1)

    def _reflow(self) -> None:
        """兼容旧入口：差量更新后统一走视图模式重排。"""
        self._apply_view_mode()

    # ------------------------------------------------------------ 统一色标
    def set_shared_scale(self, shared: bool) -> None:
        """统一色标开关：全组卡共用全局 [vmin, vmax]（display 层覆盖）。"""
        self._shared_scale = bool(shared)
        for card in self._cards:
            self._apply_scale_to_card(card)

    def shared_scale(self) -> bool:
        return self._shared_scale

    def _apply_scale_to_card(self, card) -> None:
        if card.view is None:
            return
        if not self._shared_scale:
            card.view.set_levels_override(None)
            return
        glo = self._global_range()
        if glo is not None:
            card.view.set_levels_override(glo)

    def _global_range(self):
        """现存 bundle 的全局 [vmin, vmax]；无可算数据返回 None。"""
        los, his = [], []
        for bundle in self._bundles.values():
            if bundle is None or getattr(bundle, 'matrix', None) is None:
                continue
            los.append(float(getattr(bundle, 'vmin', 0.0) or 0.0))
            his.append(float(getattr(bundle, 'vmax', 0.0) or 0.0))
        if not los:
            return None
        lo, hi = min(los), max(his)
        if hi <= lo:
            hi = lo + 1e-12
        return (lo, hi)

    def set_selected(self, key: str) -> None:
        """高亮指定槽位卡；single/compare 模式下选中驱动可见卡。"""
        self._selected_key = key
        for card_key, card in self._keys.items():
            card.set_selected(card_key == key)
        if self._view_mode != VIEW_ALL:
            self._apply_view_mode()

    def set_bundle(self, key: str, bundle) -> None:
        self._bundles[key] = bundle
        card = self._keys.get(key)
        if card is not None:
            card.set_bundle(bundle)
        if self._shared_scale:
            # 新数据可能扩大全局范围 → 全组重推（保持横向可比）
            for other in self._cards:
                self._apply_scale_to_card(other)

    def clear_all(self) -> None:
        for card in self._cards:
            card.clear()

    def card(self, key: str):
        return self._keys.get(key)

    def cards(self) -> list:
        return list(self._cards)
