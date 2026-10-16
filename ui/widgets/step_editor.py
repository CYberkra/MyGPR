# -*- coding: utf-8 -*-
"""StepEditorPopup — chip 右键合一弹窗（P0-3，2026-10-11）。

chip 右键的**唯一**入口：顶部 pill 启停 + 图标动作栏（上移/下移/复制/
替换/删除），中部按 ``parameter_schema`` 就地渲染参数（复用 ParamForm，
多参数时区内滚动），底部「恢复默认 / 应用」。

纪律（与 ChainStrip「只做展示与手势」一致）：
- **不持有 steps 数据**：打开时由宿主页经 ``load_snapshot`` 灌一次性
  快照，所有结构动作发信号回页面处理；参数与启停改动只有点「应用」
  才经 ``sig_applied`` 写回（Esc / 点外部关闭 = 未应用改动随窗丢弃）。
- 结构动作（上移/下移/复制/替换/删除）按下即关窗并转发信号。

窗口形态沿用弹窗三件式：QDialog + setModal(False) +
Qt.WindowType.Tool + focusOutEvent 后代判断 + Esc；``apply_theme`` 供
主窗 findChildren 遍历（必须幂等）。
"""
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
                             QScrollArea, QSizePolicy, QVBoxLayout, QWidget)
from qfluentwidgets import (CaptionLabel, PrimaryPushButton, PushButton, SwitchButton,
                            ToolButton, isDarkTheme)
from qfluentwidgets import FluentIcon as FIF

from ui import design_tokens
from ui.widgets.param_form import ParamForm

_POPUP_WIDTH = 380            # 合一弹窗宽（Q4 定案，580px 高真机窗口走查）
_PARAM_AREA_MAX_H = 320       # 参数区最大高，超出区内滚动（Q4 定案）
_FOCUS_RECHECK_MS = 200       # 外部点击关闭的复核延迟（焦点移交有一拍延迟）


class StepEditorPopup(QDialog):
    """单步合一编辑弹窗：启停 pill + 图标动作栏 + 参数表单 + 应用/恢复默认。"""

    sig_applied = pyqtSignal(dict)             # {index, params, enabled}（仅「应用」）
    sig_move_requested = pyqtSignal(int, int)  # index, 插入位（下移 = index+2）
    sig_duplicate_requested = pyqtSignal(int)
    sig_replace_requested = pyqtSignal(int)
    sig_remove_requested = pyqtSignal(int)

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
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        # ---------------- 顶部：标题 + 启停 pill（常驻开关，非 hover 动作）
        top_row = self._build_header()
        self._enable_switch = SwitchButton('启用', self)
        # qfw SwitchButton toggle 时用 onText/offText 覆写主标签（默认冒出
        # 尾随 On/Off）——两者恒等于语义文案（minimal-text）
        self._enable_switch.setOnText('启用')
        self._enable_switch.setOffText('屏蔽')
        self._enable_switch.setToolTip('启用 / 屏蔽该步骤（屏蔽 = 运行跳过，不删配置）')
        root.addLayout(top_row)

        # ---------------- 图标动作栏：全部 tooltip；删除红色警示
        action_row = QHBoxLayout()
        action_row.setSpacing(2)
        self._up_btn = self._make_action('up', FIF.UP, '上移该步骤')
        self._down_btn = self._make_action('down', FIF.DOWN, '下移该步骤')
        self._dup_btn = self._make_action('duplicate', FIF.COPY,
                                          '复制步骤（同算法同参数）')
        self._replace_btn = self._make_action(
            'replace', FIF.SYNC, '替换算法（位置不变，参数按新算法取默认）')
        self._remove_btn = self._make_action(
            'remove', FIF.DELETE, '删除该步骤（不可恢复）', danger=True)
        for btn in (self._up_btn, self._down_btn, self._dup_btn,
                    self._replace_btn, self._remove_btn):
            action_row.addWidget(btn)
        action_row.addStretch(1)
        action_row.addWidget(self._enable_switch)
        root.addLayout(action_row)
        # 动作按钮表：测试直驱用（契约见架构共享知识 5.2）
        self._action_buttons = {
            'up': self._up_btn, 'down': self._down_btn,
            'duplicate': self._dup_btn, 'replace': self._replace_btn,
            'remove': self._remove_btn,
        }

        # ---------------- 参数区：ParamForm 套滚动，窗体高度不爆炸
        self._param_hint = CaptionLabel('该方法无可调参数', self)
        self._param_hint.setStyleSheet(
            f'color: {design_tokens.color("text_muted")};')
        self._form = ParamForm(self)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setMaximumHeight(_PARAM_AREA_MAX_H)
        self._scroll.setWidget(self._form)
        root.addWidget(self._param_hint)
        root.addWidget(self._scroll, 1)

        # ---------------- 底部：恢复默认 / 应用
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        self._reset_btn = PushButton('恢复默认', self)
        self._reset_btn.setToolTip('参数全部重置为该算法默认值（未应用）')
        self._reset_btn.clicked.connect(self._on_reset_defaults)
        bottom_row.addWidget(self._reset_btn)
        bottom_row.addStretch(1)
        self._apply_btn = PrimaryPushButton('应用', self)
        self._apply_btn.setToolTip('写回参数与启停状态（未应用的改动丢弃）')
        self._apply_btn.clicked.connect(self._on_apply)
        bottom_row.addWidget(self._apply_btn)
        root.addLayout(bottom_row)
        self._reset_btn.setFixedHeight(30)
        self._apply_btn.setFixedSize(76, 30)
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
        title_font.setWeight(QFont.Weight.DemiBold)
        self._title_label.setFont(title_font)
        self._title_label.setStyleSheet(f'color: {design_tokens.color("text_primary")};')
        row.addWidget(self._title_label, 1)
        self._close_btn = ToolButton(FIF.CLOSE, self)
        self._close_btn.setFixedSize(28, 28)
        self._close_btn.setToolTip('关闭，不应用更改（Esc）')
        self._close_btn.setAccessibleName('关闭步骤编辑')
        self._close_btn.clicked.connect(self.close)
        row.addWidget(self._close_btn, 0, Qt.AlignmentFlag.AlignTop)
        return row

    def _make_action(self, action: str, icon, tooltip: str, *,
                     danger: bool = False) -> ToolButton:
        """图标动作钮：icon-only 配 accessibleName（屏幕阅读器可念）。"""
        btn = ToolButton(icon, self)
        btn.setFixedSize(28, 28)
        btn.setToolTip(tooltip)
        btn.setAccessibleName(tooltip)
        if danger:
            self._style_remove_button(btn)
        btn.clicked.connect(
            lambda _checked=False, a=action: self._on_action(a))
        return btn

    @staticmethod
    def _style_remove_button(btn: ToolButton, dark: bool | None = None) -> None:
        """删除钮红色警示：色值走 error 令牌（不跳色铁律），随主题重刷。

        ``dark=None`` 跟随当前主题（构造期）；apply_theme 显式传入目标主题。
        """
        error = design_tokens.color('error', dark)
        btn.setIcon(FIF.DELETE.icon(color=QColor(error)))
        btn.setStyleSheet(
            f'ToolButton{{color:{error};background:transparent;'
            'border:none;border-radius:4px;}'
            f'ToolButton:hover{{background:{design_tokens.rgba("error", 0.10, dark)};}}'
            f'ToolButton:pressed{{background:{design_tokens.rgba("error", 0.18, dark)};}}'
            f'ToolButton:disabled{{color:{design_tokens.color("disabled", dark)};}}')

    # ------------------------------------------------------------ 快照
    def load_snapshot(self, index: int, snapshot: dict) -> None:
        """一次性灌入步骤快照：{title,method_id,params,enabled,is_first,
        is_last,schema}。弹窗只持快照，写回由宿主页经公开 API 完成。"""
        snapshot = dict(snapshot or {})
        self._index = int(index)
        self._schema = [dict(item) for item in (snapshot.get('schema') or [])]
        title = str(snapshot.get('title') or snapshot.get('method_id') or '')
        self._title_label.setText(title)
        self.setWindowTitle(title)
        self.setToolTip(title)
        # 启停 pill（弹窗内状态；点「应用」才写回链路）
        self._enable_switch.setChecked(bool(snapshot.get('enabled', True)))
        # 参数表单
        if self._schema:
            self._form.set_schema(self._schema)
            for label in self._form.findChildren(CaptionLabel):
                label.setWordWrap(True)
                label.setMinimumWidth(100)
                label.setMaximumWidth(140)
                label.setSizePolicy(QSizePolicy.Policy.Preferred,
                                    QSizePolicy.Policy.Preferred)
            self._form.set_values(dict(snapshot.get('params') or {}))
        else:
            self._form.clear()
        self._param_hint.setVisible(not self._schema)
        self._scroll.setVisible(bool(self._schema))
        # 首/末步禁用态：链序边界不可再移
        self._up_btn.setEnabled(not bool(snapshot.get('is_first', False)))
        self._down_btn.setEnabled(not bool(snapshot.get('is_last', False)))

    # ------------------------------------------------------------ 动作
    def _on_apply(self) -> None:
        """「应用」：参数 + 启停一次性写回（载荷 {index, params, enabled}）。"""
        if self._index < 0:
            self.close()
            return
        self.sig_applied.emit({
            'index': self._index,
            'params': self._form.values(),
            'enabled': self._enable_switch.isChecked(),
        })
        self.close()

    def _on_reset_defaults(self) -> None:
        """「恢复默认」：按 schema 各 default 灌回表单（仅弹窗内，未应用）。"""
        defaults = {item.get('name'): item.get('default')
                    for item in self._schema
                    if item.get('name') is not None}
        self._form.set_values(defaults)

    def _on_action(self, action: str) -> None:
        """结构动作：即发对应信号 + 关窗（由宿主页转发既有处理器）。"""
        index = self._index
        if index < 0:
            self.close()
            return
        if action == 'up':
            self.sig_move_requested.emit(index, index - 1)
        elif action == 'down':
            # 插入位语义，与旧右键菜单/拖拽一致（落点 = 目标后一位）
            self.sig_move_requested.emit(index, index + 2)
        elif action == 'duplicate':
            self.sig_duplicate_requested.emit(index)
        elif action == 'replace':
            self.sig_replace_requested.emit(index)
        elif action == 'remove':
            self.sig_remove_requested.emit(index)
        self.close()

    # ------------------------------------------------------------ 事件
    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            self.setFixedWidth(min(_POPUP_WIDTH, max(300, available.width() - 24)))
            self.setMaximumHeight(max(200, available.height() - 24))
        super().showEvent(event)

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
        self._style_remove_button(self._remove_btn, dark)
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
        for button in (self._close_btn, self._up_btn, self._down_btn,
                       self._dup_btn, self._replace_btn):
            button.setStyleSheet(tool_style)
        self._title_label.setStyleSheet(
            f'color: {design_tokens.color("text_primary", dark)};')
        self._param_hint.setStyleSheet(
            f'color: {design_tokens.color("text_muted", dark)};')
