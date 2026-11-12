# -*- coding: utf-8 -*-
"""Anchored editor for a single draft step’s parameter snapshot."""
from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
                             QScrollArea, QSizePolicy, QVBoxLayout, QWidget)
from qfluentwidgets import (BodyLabel, CaptionLabel, PrimaryPushButton, PushButton,
                            ToolButton, isDarkTheme)
from qfluentwidgets import FluentIcon as FIF

from ui import design_tokens, motion
from ui.widgets.param_form import ParamForm
from ui.widgets.separators import make_h_separator

_POPUP_WIDTH = 420
_PARAM_AREA_MAX_H = 320       # 参数区最大高，超出区内滚动（Q4 定案）
_FOCUS_RECHECK_MS = 200       # 外部点击关闭的复核延迟（焦点移交有一拍延迟）


class StepEditorPopup(QDialog):
    """Edit the supplied parameter snapshot and apply it explicitly."""

    gain_preview_requested = pyqtSignal(object)
    sig_applied = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('步骤编辑')
        self.setObjectName('stepEditorPopup')
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Tool)      # 不进任务栏，随主窗
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setFixedWidth(_POPUP_WIDTH)
        self._index = -1
        self._schema: list = []

        # 失焦复核走 owned 可停 QTimer：裸 singleShot 的回调会在窗销毁后
        # 仍被派发（悬空引用），炸下一个 processEvents 的宿主
        self._focus_timer = QTimer(self)
        self._focus_timer.setSingleShot(True)
        self._focus_timer.timeout.connect(self._close_if_focus_left)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        root.addLayout(self._build_header())
        root.addWidget(make_h_separator(alpha=55))

        self._param_title = BodyLabel('参数设置', self)
        title_font = self._param_title.font()
        title_font.setWeight(QFont.Weight.DemiBold)
        self._param_title.setFont(title_font)
        self._param_hint = CaptionLabel('该方法无可调参数', self)
        self._param_hint.setStyleSheet(
            f'color: {design_tokens.color("text_muted")};')
        self._form = ParamForm(self)
        self._form.gain_preview_requested.connect(self.gain_preview_requested)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setMaximumHeight(_PARAM_AREA_MAX_H)
        self._scroll.setWidget(self._form)
        root.addWidget(self._param_title)
        root.addWidget(self._param_hint)
        root.addWidget(self._scroll, 1)

        root.addWidget(make_h_separator(alpha=55))
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)
        self._reset_btn = PushButton('恢复默认', self)
        self._reset_btn.setToolTip('参数全部重置为该算法默认值（未应用）')
        self._reset_btn.clicked.connect(self._on_reset_defaults)
        bottom_row.addWidget(self._reset_btn)
        bottom_row.addStretch(1)
        self._apply_btn = PrimaryPushButton('应用', self)
        self._apply_btn.setToolTip('应用参数（关闭时丢弃未应用的改动）')
        self._apply_btn.clicked.connect(self._on_apply)
        bottom_row.addWidget(self._apply_btn)
        root.addLayout(bottom_row)
        self._reset_btn.setFixedHeight(36)
        self._apply_btn.setFixedSize(88, 36)
        self._opacity_animation = QPropertyAnimation(self, b'windowOpacity', self)
        self._opacity_animation.setDuration(180)
        self._opacity_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.destroyed.connect(self._opacity_animation.stop)
        self.apply_theme(isDarkTheme())

    # ------------------------------------------------------------ 构建
    def _build_header(self):
        row = QHBoxLayout()
        row.setSpacing(8)
        self._title_label = QLabel(self)
        self._title_label.setWordWrap(True)
        self._title_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Preferred)
        title_font = self._title_label.font()
        title_font.setPointSize(15)
        title_font.setWeight(QFont.Weight.DemiBold)
        self._title_label.setFont(title_font)
        self._title_label.setStyleSheet(f'color: {design_tokens.color("text_primary")};')
        row.addWidget(self._title_label, 1)
        self._close_btn = ToolButton(FIF.CLOSE, self)
        self._close_btn.setFixedSize(32, 32)
        self._close_btn.setToolTip('关闭，不应用更改（Esc）')
        self._close_btn.setAccessibleName('关闭步骤编辑')
        self._close_btn.clicked.connect(self.close)
        row.addWidget(self._close_btn, 0, Qt.AlignmentFlag.AlignTop)
        return row

    def load_snapshot(self, index: int, snapshot: dict) -> None:
        snapshot = dict(snapshot or {})
        self._index = int(index)
        self._schema = [dict(item) for item in (snapshot.get('schema') or [])]
        title = str(snapshot.get('title') or snapshot.get('method_id') or '')
        self._title_label.setText(title)
        self.setWindowTitle(title)
        self.setToolTip(title)
        if self._schema:
            self._form.set_schema(self._schema)
            self._form.set_values(dict(snapshot.get('params') or {}))
        else:
            self._form.clear()
        self._param_hint.setVisible(not self._schema)
        self._scroll.setVisible(bool(self._schema))
    def _on_apply(self) -> None:
        if self._index < 0:
            self.close()
            return
        self.sig_applied.emit({
            'index': self._index,
            'params': self._form.values(),
        })
        self.close()

    def _on_reset_defaults(self) -> None:
        """「恢复默认」：按 schema 各 default 灌回表单（仅弹窗内，未应用）。"""
        defaults = {item.get('name'): item.get('default')
                    for item in self._schema
                    if item.get('name') is not None}
        self._form.set_values(defaults)

    # ------------------------------------------------------------ 事件
    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            self.setFixedWidth(min(_POPUP_WIDTH, max(300, available.width() - 24)))
            self.setMaximumHeight(max(200, available.height() - 24))
        super().showEvent(event)
        self._opacity_animation.stop()
        if motion.animations_enabled():
            self.setWindowOpacity(0.0)
            self._opacity_animation.setStartValue(0.0)
            self._opacity_animation.setEndValue(1.0)
            self._opacity_animation.start()
        else:
            self.setWindowOpacity(1.0)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        # 窗内控件切换时 focusWidget 仍是本窗后代（不误关）；
        # 点到窗外 → 复核一次再关（真机下焦点移交有一拍延迟）
        self._focus_timer.start(_FOCUS_RECHECK_MS)
        super().focusOutEvent(event)

    def _close_if_focus_left(self) -> None:
        """复核焦点是否已落到本窗之外；是则关闭（未应用改动随窗丢弃）。

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
        """主题切换重刷令牌色（主窗鸭子类型派发；必须幂等——遍历可能
        命中隐藏弹窗）。qfw 控件随主题自动，这里只重刷自绘样式。"""
        self.setStyleSheet(
            f'QDialog#stepEditorPopup{{background:{design_tokens.color("bg_surface", dark)};'
            f'border:1px solid {design_tokens.color("border_default", dark)};'
            f'border-radius:{design_tokens.radius("lg")}px;}}'
            'QScrollArea{background:transparent;border:0;}'
            'QScrollArea > QWidget > QWidget{background:transparent;}')
        tool_style = (
            'ToolButton{background:transparent;border:0;border-radius:4px;}'
            f'ToolButton:hover{{background:{design_tokens.color("bg_muted", dark)};}}'
            f'ToolButton:pressed{{background:{design_tokens.rgba("primary", .12, dark)};}}')
        self._close_btn.setStyleSheet(tool_style)
        self._title_label.setStyleSheet(
            f'color: {design_tokens.color("text_primary", dark)};')
        self._param_title.setStyleSheet(
            f'color: {design_tokens.color("text_primary", dark)};')
        self._param_hint.setStyleSheet(
            f'color: {design_tokens.color("text_muted", dark)};')
