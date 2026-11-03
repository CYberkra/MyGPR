# -*- coding: utf-8 -*-
"""MethodPickerDialog — 非模态方法库选择弹窗（P0-2，2026-10-11）。

「+」号与「添加所选方法」的统一入口：顶部搜索 + 分类算法树（内嵌复用
MethodBrowser 的数据形态，``set_methods(list[dict])`` 同源灌入）。双击
条目 → ``sig_method_chosen(method_id)`` 且**保持打开**（连续添加，PRD
Q1 定案）；Esc / 点击外部关闭；再开状态复位（搜索词清空）。

窗口形态沿用 TVG 对话框先例（QDialog + setModal(False) +
Qt.WindowType.Tool，不进任务栏、随主窗），内联于处理页（parent = 页面）。

外部点击关闭判定：focusOutEvent 里查 ``QApplication.focusWidget()`` 是否
本窗后代——窗内控件互相切换（搜索框 → 树）焦点仍是本窗后代，不误关；
点到窗外则 200ms 复核一次再关（真机焦点移交有一拍延迟）。
"""
from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, PrimaryPushButton, ToolButton, isDarkTheme
from qfluentwidgets import FluentIcon as FIF

from ui import design_tokens, motion
from ui.widgets.method_browser import MethodBrowser

# 外部点击关闭的复核延迟（ms）：焦点移交到外部控件有一拍延迟，
# 立即判定会把「点开 colorbox 子菜单」这类窗内操作误判为失焦
_FOCUS_RECHECK_MS = 200


class MethodPickerDialog(QDialog):
    """方法库选择弹窗：双击条目发 sig_method_chosen，不关窗连续添加。"""

    sig_method_chosen = pyqtSignal(str)   # method_id（双击条目）

    def __init__(self, parent: QWidget | None = None, anchor: QWidget | None = None) -> None:
        super().__init__(parent)
        self._anchor = anchor
        self.setWindowTitle('添加处理步骤')
        self.setObjectName('methodPickerDialog')
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Tool)
        self.setMinimumSize(480, 420)
        self.resize(520, 520)
        self._methods_by_id = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        header = QHBoxLayout()
        header.setSpacing(8)
        self._count_label = CaptionLabel('', self)
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_label.setFixedHeight(24)
        self._close_btn = ToolButton(FIF.CLOSE, self)
        self._close_btn.setToolTip('关闭（Esc）')
        self._close_btn.setAccessibleName('关闭算法库')
        self._close_btn.clicked.connect(self.close)
        header.addStretch(1)
        header.addWidget(self._count_label)
        header.addWidget(self._close_btn)
        layout.addLayout(header)

        self._browser = MethodBrowser(self)
        self._browser.sig_add_requested.connect(self._on_add_requested)
        self._browser.sig_method_selected.connect(self._on_selection_changed)
        layout.addWidget(self._browser, 1)

        self._selected_name_label = BodyLabel('未选择', self)
        name_font = self._selected_name_label.font()
        name_font.setWeight(QFont.Weight.DemiBold)
        self._selected_name_label.setFont(name_font)
        self._description_label = CaptionLabel('', self)
        self._description_label.setWordWrap(True)
        self._description_label.setMaximumHeight(60)
        self._description_label.hide()

        self._details_btn = ToolButton(FIF.INFO, self)
        self._details_btn.setCheckable(True)
        self._details_btn.setToolTip('查看／隐藏算法用途')
        self._details_btn.setAccessibleName('查看算法用途')
        self._details_btn.toggled.connect(self._sync_details_visibility)
        self._add_selected_btn = PrimaryPushButton('添加', self, FIF.ADD)
        self._add_selected_btn.setFixedSize(96, 36)
        self._add_selected_btn.setEnabled(False)
        self._add_selected_btn.clicked.connect(self._add_current_method)
        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(self._selected_name_label, 1)
        footer.addWidget(self._details_btn)
        footer.addWidget(self._add_selected_btn)
        layout.addLayout(footer)
        layout.addWidget(self._description_label)

        self._focus_timer = QTimer(self)
        self._focus_timer.setSingleShot(True)
        self._focus_timer.timeout.connect(self._close_if_focus_left)
        self._geometry_animation = QPropertyAnimation(self, b'geometry', self)
        self._geometry_animation.setDuration(180)
        self._geometry_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._opacity_animation = QPropertyAnimation(self, b'windowOpacity', self)
        self._opacity_animation.setDuration(180)
        self._opacity_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.destroyed.connect(self._geometry_animation.stop)
        self.destroyed.connect(self._opacity_animation.stop)
        self.apply_theme(isDarkTheme())

    # ------------------------------------------------------------ 数据
    def set_methods(self, methods: list) -> None:
        """与方法库控件同源的方法清单（结构见 ProcessingPage.set_methods）。"""
        self._methods_by_id = {str(item.get('method_id') or ''): dict(item) for item in (methods or [])}
        self._browser.set_methods(methods or [])
        self._count_label.setText(f'{len(methods or [])} 个方法')
        self._on_selection_changed(self._browser.current_method_id())

    # ------------------------------------------------------------ 复位
    def reset_state(self) -> None:
        """清空搜索与选择，仅展开第一组并收起算法详情。"""
        self._browser._search.clear()
        self._browser._tree.setCurrentItem(None)
        for index in range(self._browser._tree.topLevelItemCount()):
            self._browser._tree.topLevelItem(index).setExpanded(index == 0)
        self._details_btn.setChecked(False)
        self._description_label.hide()
        self._on_selection_changed(None)

    # ------------------------------------------------------------ 信号
    def _on_add_requested(self, method_id: str) -> None:
        """双击条目：转发选中方法，**不关窗**（连续添加，PRD Q1 定案）。"""
        if method_id:
            self.sig_method_chosen.emit(str(method_id))

    def _on_selection_changed(self, method_id) -> None:
        method = self._methods_by_id.get(str(method_id or ''))
        if method is None:
            self._selected_name_label.setText('未选择')
            self._description_label.setText('')
            self._description_label.setToolTip('')
            self._add_selected_btn.setEnabled(False)
            return
        name = str(method.get('display_name') or method.get('name') or method_id)
        description = str(method.get('description') or '该算法没有补充说明。')
        self._selected_name_label.setText(name)
        self._description_label.setText(description)
        self._description_label.setToolTip(description)
        self._add_selected_btn.setEnabled(True)
        self._sync_details_visibility()

    def _sync_details_visibility(self) -> None:
        self._description_label.setVisible(
            self._details_btn.isChecked() and bool(self._description_label.text()))

    def _add_current_method(self) -> None:
        method_id = self._browser.current_method_id()
        if method_id:
            self._on_add_requested(method_id)

    def _position_near_anchor(self) -> None:
        if self._anchor is None or not self._anchor.isVisible():
            return
        available = self._anchor.screen().availableGeometry()
        below = self._anchor.mapToGlobal(QPoint(0, self._anchor.height() + 6)).y()
        y = below if below + self.height() <= available.bottom() else \
            self._anchor.mapToGlobal(QPoint(0, -self.height() - 6)).y()
        x = self._anchor.mapToGlobal(QPoint(self._anchor.width() - self.width(), 0)).x()
        x = max(available.left(), min(x, available.right() - self.width()))
        y = max(available.top(), min(y, available.bottom() - self.height()))
        self.move(x, y)

    # ------------------------------------------------------------ 事件
    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        """每次显示复位状态：关闭再开 = 全新浏览态（PRD 验收④）。"""
        self.reset_state()
        super().showEvent(event)
        self._position_near_anchor()
        self._browser._search.setFocus()
        self._geometry_animation.stop()
        self._opacity_animation.stop()
        if motion.animations_enabled():
            final = self.geometry()
            self._geometry_animation.setStartValue(final.translated(0, 8))
            self._geometry_animation.setEndValue(final)
            self.setWindowOpacity(0.0)
            self._opacity_animation.setStartValue(0.0)
            self._opacity_animation.setEndValue(1.0)
            self._geometry_animation.start()
            self._opacity_animation.start()
        else:
            self.setWindowOpacity(1.0)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key.Key_F and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._browser._search.setFocus()
            self._browser._search.selectAll()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._add_current_method()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        # 窗内控件切换时 focusWidget 仍是本窗后代（不误关）；
        # 点到窗外 → 复核一次再关（真机下焦点移交有一拍延迟）
        self._focus_timer.start(_FOCUS_RECHECK_MS)
        super().focusOutEvent(event)

    def _close_if_focus_left(self) -> None:
        """复核焦点是否已落到本窗之外；是则关闭。

        复核由 singleShot 延迟触发：窗可能已被宿主销毁（teardown 竞态），
        届时 sip 包装器失效——静默丢弃，不得把 RuntimeError 漏进测试/运行期。
        """
        try:
            widget = QApplication.focusWidget()
            while widget is not None:
                if widget is self:
                    return
                widget = widget.parentWidget()
            self.close()
        except RuntimeError:
            return

    # ------------------------------------------------------------ 主题
    def apply_theme(self, dark: bool) -> None:
        """主题切换钩子（主窗 findChildren 鸭子类型遍历自动调用）。

        必须幂等（纯令牌重刷）：遍历可能命中隐藏弹窗。
        """
        self.setStyleSheet(
            f'QDialog#methodPickerDialog {{ background: {design_tokens.color("bg_surface", dark)};'
            f' border: 1px solid {design_tokens.color("border_default", dark)};'
            f' border-radius: {design_tokens.radius("lg")}px; }}'
            f'SubtitleLabel {{ color: {design_tokens.color("text_primary", dark)}; }}'
            f'BodyLabel {{ color: {design_tokens.color("text_secondary", dark)}; }}'
            f'CaptionLabel {{ color: {design_tokens.color("text_muted", dark)}; }}'
        )
        self._count_label.setStyleSheet(
            f'color: {design_tokens.color("primary", dark)};'
            f' background: {design_tokens.color("primary_subtle", dark)};'
            ' border-radius: 4px; padding: 0 8px;'
        )
        self._selected_name_label.setStyleSheet(
            f'color: {design_tokens.color("text_primary", dark)};'
        )
        self._description_label.setStyleSheet(
            f'color: {design_tokens.color("text_secondary", dark)};'
        )
        self._browser.apply_theme(dark)
