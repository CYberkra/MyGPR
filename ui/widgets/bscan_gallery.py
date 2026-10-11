# -*- coding: utf-8 -*-
"""BScanGallery — 总览墙（P1-4 重写，2026-10-11）。

定位是**总览工具而非精读工具**：当前上下文数据源（原始 + 本次 run_group
各步，与结果网格一致）的卡片墙——真实 BScanView 缩略图 + 步序徽章 +
名称；动态列数随源数 1→2→3→4 档增长；当前主区源 2px 描边 + 「当前」
角标；hover 仅淡描边（无悬浮按钮）；单击整卡 = 设为当前并关闭；空状态
居中；打开错峰淡入；Esc 关闭。

单实例语义由宿主页托管（processing_page._open_gallery）：存活且可见 →
set_sources 增量更新 + activateWindow；closeEvent 后经 finished 信号
置页面引用 None（修旧版每次 new 实例覆盖引用、窗口堆叠的 bug）。

数据源由宿主**显式传入**（原始 + 当前 run_group 各步，与结果网格一致）
——不遍历 page._preview_sources 全量（历史积累会把几天前的成果混进
当次结果，真机截图反馈）。

网格重排纪律（复刻 bscan_result_grid 教训）：重排前列 stretch 全清，
否则空列残留分走一半宽度。
"""
from PyQt6.QtCore import QPropertyAnimation, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QGraphicsOpacityEffect, QGridLayout,
                             QLabel, QScrollArea, QSizePolicy, QVBoxLayout,
                             QWidget)
from qfluentwidgets import CaptionLabel, FluentIcon as FIF

from ui.design_tokens import color, radius, rgba
from ui.widgets.bscan_view import BScanView
from ui.widgets.empty_state import EmptyStateOverlay

_MIN_CARD_H = 200           # 卡最小高（总览工具低于结果网格 320，守可读下限）
_MIN_CARD_W = 240           # 列宽最小值（Q3 视觉走查下限）
_MAX_COLUMNS = 4            # 列数上限（1/2/3/4 档）
_FADE_MS = 240              # 单卡淡入时长（复用结果网格规格）
_STAGGER_MS = 60            # 错峰步进（index * 60ms）
_PLACEHOLDER_TEXT = '预览尚未生成'   # 预览异步回填未到的占位文案（测试有断言）


class _GalleryCard(QWidget):
    """单张总览卡：真实 BScanView 缩略图 + 步序徽章 + 名称 + 当前角标。

    整卡可点（mousePressEvent 左键 → sig_clicked(key)）；hover 仅淡描边；
    当前卡 2px border_focus 描边 + 右上「当前」角标（令牌与结果网格
    _sel_qss 同规格）。
    """

    sig_clicked = pyqtSignal(str)          # 卡 key（= 数据源 key）

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._key = ''
        self._current = False
        self.setObjectName('galleryCard')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumSize(_MIN_CARD_W, _MIN_CARD_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_qss()

        body = QVBoxLayout(self)
        body.setContentsMargins(8, 8, 8, 8)
        body.setSpacing(6)
        # 图区：BScanView + 浮层（徽章 / 角标 / 占位提示）
        self._canvas = QWidget(self)
        self._canvas.setMinimumHeight(120)
        canvas_layout = QVBoxLayout(self._canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._view = BScanView(self._canvas, with_colorbar=False)
        self._view.set_title_visible(False)     # 卡头已是唯一信息源
        canvas_layout.addWidget(self._view, 1)
        body.addWidget(self._canvas, 1)
        # 名称（卡底，常显）
        self._name = CaptionLabel('', self)
        body.addWidget(self._name)
        # 浮层：步序徽章（左上）+「当前」角标（右上）+ 占位提示（铺满）
        self._badge = QLabel(self._canvas)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._corner = QLabel('当前', self._canvas)
        self._corner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder = QLabel(_PLACEHOLDER_TEXT, self._canvas)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._style_overlays()

    # ------------------------------------------------------------ 样式
    def _apply_qss(self) -> None:
        """常态/hover/当前三态描边：2px 常宽（transparent 占位防布局跳）。"""
        r = radius('md')
        if self._current:
            self.setStyleSheet(
                f'#galleryCard{{border:2px solid {color("border_focus")};'
                f'border-radius:{r}px}}')
        else:
            self.setStyleSheet(
                f'#galleryCard{{border:2px solid transparent;'
                f'border-radius:{r}px}}'
                f'#galleryCard:hover{{border:2px solid '
                f'{rgba("primary", 0.35)};}}')

    def _style_overlays(self) -> None:
        """徽章/角标/占位提示的令牌色（主题切换经 apply_theme 重刷）。"""
        pill = radius('pill')
        self._badge.setStyleSheet(
            f'QLabel{{color:{color("text_inverse")};'
            f'background:{color("text_muted")};'
            f'border-radius:{pill}px;padding:1px 8px;}}')
        self._corner.setStyleSheet(
            f'QLabel{{color:{color("text_inverse")};'
            f'background:{color("primary")};'
            f'border-radius:{pill}px;padding:1px 8px;}}')
        self._placeholder.setStyleSheet(
            f'color:{color("text_muted")};'
            f'border:1px dashed {color("border_default")};'
            f'border-radius:{radius("sm")}px;')
        self._name.setStyleSheet(f'color:{color("text_secondary")};')

    # ------------------------------------------------------------ 数据
    def set_source(self, source: dict, index: int) -> None:
        """灌入数据源：key/标题/步序徽章/bundle（None → 占位提示）。"""
        source = dict(source or {})
        self._key = str(source.get('key', ''))
        title = str(source.get('title', ''))
        self._name.setText(title)
        self.setToolTip(title)
        # 步序徽章：原始数据槽位 = 「输入」，其余按步序 1..n
        self._badge.setText('输入' if index == 0 else str(index))
        bundle = source.get('bundle')
        if bundle is not None:
            self._view.set_bundle(bundle)
            self._placeholder.setVisible(False)
        else:
            # 预览异步回填未到（自动弹画廊与预览回填有时序窗）：
            # 空画布无骨架无提示像故障 → muted 虚线占位（如实告知）
            self._view.clear()
            self._placeholder.setVisible(True)

    def set_current(self, current: bool) -> None:
        """当前主区源：2px 描边 + 「当前」角标。"""
        self._current = bool(current)
        self._corner.setVisible(self._current)
        self._apply_qss()

    # ------------------------------------------------------------ 事件
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.button() == Qt.MouseButton.LeftButton and self._key:
            self.sig_clicked.emit(self._key)
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().resizeEvent(event)
        self._layout_overlays()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().showEvent(event)
        self._layout_overlays()

    def _layout_overlays(self) -> None:
        """浮层定位：徽章左上、角标右上、占位提示铺满图区。"""
        self._badge.adjustSize()
        self._badge.move(6, 6)
        self._corner.adjustSize()
        self._corner.move(max(0, self._canvas.width()
                              - self._corner.width() - 6), 6)
        self._placeholder.setGeometry(self._canvas.rect())

    def apply_theme(self, dark: bool) -> None:
        """主题切换重刷令牌色（主窗 findChildren 鸭子类型遍历；幂等）。"""
        del dark    # 令牌经 design_tokens 按当前主题取值，与结果网格同款
        self._apply_qss()
        self._style_overlays()


class BScanGallery(QDialog):
    """总览墙：动态列数卡片网格；单击整卡 → sig_pick(key) 并关闭。"""

    sig_pick = pyqtSignal(str)      # 宿主接 → on_gallery_pick 语义不变

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('总览墙 — 打开的数据源')
        self.setModal(False)
        self._sources: list = []
        self._current_key = ''
        self._cards: list = []

        self._body = QWidget(self)
        self._grid = QGridLayout(self._body)
        self._grid.setContentsMargins(12, 12, 12, 12)
        self._grid.setSpacing(12)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)
        # 空状态：居中引导（源数为 0 时网格区不铺卡）
        self._empty = EmptyStateOverlay(
            self._scroll, icon=FIF.PHOTO, title='暂无打开的数据源',
            hint='选择测线或运行处理链后，此处展示原始数据与各步结果')
        self._empty.setVisible(True)
        self.setMinimumSize(max(420, _MIN_CARD_W + 48), 360)

    # ------------------------------------------------------------ 数据
    def set_sources(self, sources: list, current_key: str = '') -> None:
        """数据源清单 + 当前主区源 key → 重建卡片墙（含错峰淡入）。"""
        self._sources = [dict(s) for s in (sources or [])]
        self._current_key = str(current_key or '')
        self._rebuild()

    def _rebuild(self) -> None:
        for card in self._cards:
            # 先停错峰淡入动画再销毁：动画运行中目标被删是 Qt 侧
            # access violation 的经典来源（teardown 竞态）
            fade = getattr(card, '_fade_anim', None)
            if fade is not None:
                try:
                    fade.stop()
                except RuntimeError:
                    pass
            self._grid.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards = []
        # 列拉伸先全清：QGridLayout 的 setColumnStretch 会跨重排残留
        # （列数从 4 档回落时空列仍按旧 stretch 分走宽度——结果网格教训）
        for col in range(_MAX_COLUMNS):
            self._grid.setColumnStretch(col, 0)
        n = len(self._sources)
        self._empty.setVisible(n == 0)
        if n == 0:
            return
        cols = self._columns_for(n)
        for index, source in enumerate(self._sources):
            card = _GalleryCard(self._body)
            card.set_source(source, index)
            card.set_current(str(source.get('key', '')) == self._current_key)
            card.sig_clicked.connect(self._on_pick)
            row, col = divmod(index, cols)
            self._grid.addWidget(card, row, col)
            self._cards.append(card)
            self._start_fade(card, index)
        for col in range(cols):
            self._grid.setColumnStretch(col, 1)
        self.resize(max(640, cols * 260), 520)

    @staticmethod
    def _columns_for(n: int) -> int:
        """列数档位：n<=1→1；2→2；3→3；>=4→4（验收矩阵 1/3/5/8/12）。"""
        if n <= 1:
            return 1
        if n == 2:
            return 2
        if n == 3:
            return 3
        return 4

    def _start_fade(self, card: _GalleryCard, index: int) -> None:
        """错峰淡入：初始透明，index*60ms 后 240ms 淡入（结果网格规格）。"""
        effect = QGraphicsOpacityEffect(card)
        effect.setOpacity(0.0)
        card.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b'opacity', card)
        anim.setDuration(_FADE_MS)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        card._fade_anim = anim
        # singleShot 保留 anim 的引用：卡被提前销毁（关窗/重建）时
        # start() 落在已删对象上——守卫吞掉 RuntimeError
        QTimer.singleShot(
            index * _STAGGER_MS,
            lambda a=anim: self._safe_start_fade(a))

    @staticmethod
    def _safe_start_fade(anim: QPropertyAnimation) -> None:
        """延迟启动淡入；动画/目标已随卡销毁则静默跳过。"""
        try:
            anim.start()
        except RuntimeError:
            pass

    # ------------------------------------------------------------ 事件
    def _on_pick(self, key: str) -> None:
        """单击整卡：把该源送回主区（宿主接 sig_pick）并关闭。

        语义同旧版 on_gallery_pick 路径：设当前源 → 可见性规则换入主区。
        """
        self.sig_pick.emit(str(key))
        self.close()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------ 测试口
    def cards(self) -> list:
        """当前卡片清单（测试断言用）。"""
        return list(self._cards)

    def apply_theme(self, dark: bool) -> None:
        """主题切换：逐卡重刷令牌色（幂等，隐藏弹窗被遍历命中也安全）。"""
        for card in self._cards:
            card.apply_theme(dark)
