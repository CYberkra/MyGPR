# -*- coding: utf-8 -*-
"""ChainStrip — 处理链极简 chip 条（处理页 v2 顶部）。

设计（2026-09-25 v2；2026-10-16 可交互待执行链）：
- 一行：输入、算法名与关键参数摘要；完整名称、参数与启停状态走 tooltip；
- **hover 才显**启用圆点与 ✕（常驻噪音更少），链尾虚线 ＋ 追加；
- **溢出可达**（2026-10-09 深查 P1）：chip 放不下时右缘浮「+N」胶囊，
  点击弹全部步骤菜单（每步子菜单 = 选中/编辑）——
  隐藏步骤的手势与右键菜单等价可达（wrapping=False + 滚动条常关的补偿）；
- **运行钮就近翻转**（2026-10-09 深查 P2）：运行中同位置变「取消」
  （红描边），结束恢复「运行」——取消与运行视线不跳；进行中反馈由
  进度行承担（原 spinner 文字动画退役）；
- 拖拽排序：启用 InternalMove 拿拖拽视觉，落点由 _ChipList.dropEvent
  委托宿主接管（不调 Qt 默认实现——挪 item 会丢 setItemWidget 行控件，
  与 PipelineList 同款坑）；
- **手势替代路径**（guidelines：拖拽需有点击/键盘替代）：右键 chip →
  合一编辑弹窗（P0-3：启停/排序/复制/替换/参数/删除一处完成）+
  Delete 键删当前 chip；Shift+F10/Menu 编辑当前 chip；
- 「输入」chip 固定在首位、不可拖不可删。

ChainStrip 只缓存展示字段与手势，**处理草稿仍由宿主页维护**。
"""
from PyQt6.QtCore import (QEvent, QEasingCurve, QPointF, QPropertyAnimation, Qt,
                          QTimer, pyqtSignal)
from PyQt6.QtGui import QDrag, QFont, QKeySequence, QMouseEvent, QShortcut
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QWidget)
from PyQt6.QtWidgets import QPushButton
from qfluentwidgets import (CaptionLabel, FluentIcon as FIF,
                            PrimaryPushButton, RoundMenu, ToolButton)

from ui import constants
from ui.widgets.context_menus import add_action, make_menu


def _muted_text() -> str:
    """次级文字色（设计令牌单源）。"""
    from ui.design_tokens import color
    return color('text_muted')


def _chip_qss(enabled: bool, selected: bool = False) -> str:
    """chip 胶囊 QSS（启用实线 / 禁用虚线 + hover 反馈），色值走令牌。

    状态矩阵：default / hover（描边转 primary 淡色，预示可点）/ selected
    （由滑动胶囊承担，不在此表）/ disabled（虚线 + 次级文字，真不可点由
    信号宿主保证）。
    """
    from ui.design_tokens import color, radius, rgba
    background = rgba('primary', .10) if selected else color('bg_subtle')
    border = rgba('primary', .65) if selected else rgba('border_default', .65)
    line = 'solid' if enabled else 'dashed'
    style = (f'#chip{{background:{background};border:1px {line} {border};'
             f'border-radius:{radius("lg")}px;}}')
    if enabled:
        style += (f'#chip:hover{{border-color:{rgba("primary", .55)};'
                  f'background:{rgba("primary", .07)};}}')
    return style


class _Chip(QWidget):
    """单个步骤 chip：胶囊造型，算法名（常显）+ 启用圆点 / ✕（hover 显）。

    名称精简：目录显示名常带英文后缀（如「零时校正 (Dewow)」），chip 上
    只保留中文名（tooltip 保留全名）——长名省略号截断，不顶出滚动条。
    """

    _MAX_NAME_PX = 132

    def __init__(self, index: int, label: str, enabled: bool, host,
                 parent=None, param_summary: str = ''):
        super().__init__(parent)
        self.index = index                  # -1 = 输入 chip
        self._host = host
        self._selected = False
        self.setObjectName('chip')
        self.setFixedHeight(28)
        # 普通 QWidget 子类必须开这个才会绘制样式表背景（否则 chip 无胶囊底）
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setToolTip(label)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 3, 6, 3)
        row.setSpacing(6)
        short = label.split('（')[0].split(' (')[0].strip() or label
        self.name = QLabel(self)
        name_font = self.name.font()
        name_font.setWeight(QFont.Weight.DemiBold)
        self.name.setFont(name_font)
        self.installEventFilter(host._list)
        self.name.installEventFilter(host._list)
        self.name.setMaximumWidth(self._MAX_NAME_PX)
        # 省略号截断（2026-10-09 深查 P2）：QLabel 默认硬裁无「…」，截断
        # 处看着像数据缺失；全名走 chip 级 tooltip（__init__ 已设）
        fm = self.name.fontMetrics()
        self.name.setText(fm.elidedText(short, Qt.TextElideMode.ElideRight,
                                        self._MAX_NAME_PX))
        row.addWidget(self.name)
        self.summary = QLabel(self)
        self.summary.setMaximumWidth(150)
        self.summary.installEventFilter(host._list)
        row.addWidget(self.summary)
        # 启用开关 = 圆点（绿=启用/灰=禁用）——不用 ✓ 形图标（用户定案：
        # 全应用不出现勾形元素，避免与「运行」产生歧义）
        self.dot_btn = QPushButton(self)
        self.dot_btn.setFixedSize(18, 18)
        self.dot_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dot_btn.setToolTip('启用 / 禁用该步骤')
        # icon-only 按钮无文字：屏幕阅读器念 accessibleName（2026-10-09 深查）
        self.dot_btn.setAccessibleName('启用或禁用该步骤')
        self.dot_btn.clicked.connect(
            lambda _c=False, i=index: host._on_dot_clicked(i))
        self._set_dot_visual(enabled)
        row.addWidget(self.dot_btn)
        self.del_btn = ToolButton(FIF.CLOSE, self)
        # 20px（原 18）：紧凑 chip 内尽量大的点击目标（深查 P3 复审）
        self.del_btn.setFixedSize(20, 20)
        self.del_btn.setToolTip('删除该步骤')
        self.del_btn.setAccessibleName('删除该步骤')
        self.del_btn.clicked.connect(
            lambda _c=False, i=index: host._on_chip_deleted(i))
        row.addWidget(self.del_btn)
        for button in (self.dot_btn, self.del_btn):
            policy = button.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            button.setSizePolicy(policy)
        self.dot_btn.setVisible(False)
        self.del_btn.setVisible(False)
        self.set_enabled_visual(enabled)
        self.set_content(label, param_summary, enabled)

    def set_content(self, label: str, param_summary: str, enabled: bool) -> None:
        short = label.split('（')[0].split(' (')[0].strip() or label
        self.name.setText(self.name.fontMetrics().elidedText(
            short, Qt.TextElideMode.ElideRight, self._MAX_NAME_PX))
        self.summary.setText(self.summary.fontMetrics().elidedText(
            param_summary, Qt.TextElideMode.ElideRight, 150))
        self.summary.setVisible(bool(param_summary))
        state = '已启用' if enabled else '已禁用（运行时跳过）'
        self.setToolTip('\n'.join(p for p in (label, param_summary, state) if p))
        self.setAccessibleName(self.toolTip().replace('\n', '，'))
        self.dot_btn.setToolTip('停用该步骤（运行时跳过）' if enabled else '启用该步骤')
        self.dot_btn.setAccessibleName(self.dot_btn.toolTip())
        self.set_enabled_visual(enabled)

    def _set_dot_visual(self, enabled: bool) -> None:
        """状态点走语义令牌（2026-10-09）：原硬编码 #7CC464/#5A5A56 不随
        主题，绿色也不在设计令牌里（跳色纪律）。

        hover 不再发明第二个绿：细描边环提示可点（18px 小目标的反馈由
        光标 + 邻近 chip 的 hover 描边共同承担）。
        """
        from ui.design_tokens import color
        bg = color('success') if enabled else color('disabled')
        self.dot_btn.setStyleSheet(
            f'QPushButton{{color:{bg};background:transparent;'
            'border:none;border-radius:4px;font-size:10px;}'
            f'QPushButton:hover{{border:1px solid {color("border_strong")};}}')
        self.dot_btn.setText('●')

    def set_enabled_visual(self, enabled: bool) -> None:
        from ui.design_tokens import color
        self._set_dot_visual(enabled)
        self.name.setStyleSheet(f'color:{color("text_primary") if enabled else _muted_text()}')
        self.summary.setStyleSheet(f'color:{_muted_text()}')
        self.setStyleSheet(_chip_qss(enabled, self._selected))

    def set_selected_visual(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self.set_enabled_visual(self._host._steps[self.index]['enabled'])

    def enterEvent(self, event) -> None:
        if self.index >= 0:                 # 输入 chip 无操作
            self.dot_btn.setVisible(True)
            self.del_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.dot_btn.setVisible(False)
        self.del_btn.setVisible(False)
        super().leaveEvent(event)


class _ChipList(QListWidget):
    """chip 容器：dropEvent 委托宿主（绕开 Qt InternalMove 丢 itemWidget）。"""

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self._host = host
        self._press_source = -1
        self._press_generation = -1
        self._drag_active = False
        self._drop_marker = QWidget(self.viewport())
        self._drop_marker.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._drop_marker.hide()

    def eventFilter(self, watched, event):
        # Item widgets otherwise absorb pointer/drag gestures before QListWidget.
        # Only the chip body and labels forward; action buttons keep their clicks.
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove,
                            QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick):
            pos = watched.mapTo(self.viewport(), event.position().toPoint())
            forwarded = QMouseEvent(event.type(), QPointF(pos), event.globalPosition(),
                                    event.button(), event.buttons(), event.modifiers())
            QApplication.sendEvent(self.viewport(), forwarded)
            return True
        if event.type() == QEvent.Type.ContextMenu:
            self._host._on_context_menu(watched.mapTo(self.viewport(), event.pos()))
            return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event):
        self._press_source = -1
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self._press_source = self.indexAt(event.position().toPoint()).row()
            self._press_generation = self._host._display_generation
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        source = self._press_source
        generation = self._press_generation
        super().mouseReleaseEvent(event)
        self._press_source = -1
        if (self.isEnabled() and event.button() == Qt.MouseButton.LeftButton
                and generation == self._host._display_generation and source >= 0
                and self.indexAt(event.position().toPoint()).row() == source):
            self._host.sig_step_clicked.emit(source)

    def startDrag(self, actions):
        if (self._press_source < 0 or not self.isEnabled()
                or self._press_generation != self._host._display_generation):
            return
        self.setCurrentRow(self._press_source)
        self._drag_active = True
        try:
            super().startDrag(actions)
        finally:
            self._drag_active = False
            self._press_source = -1
            self._drop_marker.hide()

    def insertion_point(self, pos):
        index = self.indexAt(pos).row()
        if index < 0:
            if not self.count():
                return 0
            first = self.visualItemRect(self.item(0))
            return 0 if pos.x() < first.left() else self.count()
        rect = self.visualItemRect(self.item(index))
        return index + int(pos.x() > rect.center().x())

    def dragMoveEvent(self, event):
        if (event.source() is not self or self._press_source < 0
                or self._press_generation != self._host._display_generation):
            event.ignore()
            self._drop_marker.hide()
            return
        target = self.insertion_point(event.position().toPoint())
        item = self.item(min(target, self.count() - 1))
        rect = self.visualItemRect(item)
        x = rect.left() - 4 if target < self.count() else rect.right() + 4
        from ui.design_tokens import color
        self._drop_marker.setStyleSheet(f'background:{color("primary")};border-radius:1px;')
        self._drop_marker.setGeometry(max(0, min(x, self.viewport().width() - 3)), 4, 3, 30)
        self._drop_marker.show()
        self._drop_marker.raise_()
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._drop_marker.hide()
        super().dragLeaveEvent(event)

    def keyPressEvent(self, event):
        if (event.key() == Qt.Key.Key_Menu or
                (event.key() == Qt.Key.Key_F10 and
                 event.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
            self._host._request_editor(self.currentRow())
            event.accept()
            return
        super().keyPressEvent(event)

    def dropEvent(self, event) -> None:
        self._host._list_drop_event(event, source=self._press_source,
                                    generation=self._press_generation)


class ChainStrip(QWidget):
    """处理链 chip 条 + 追加钮 + 运行钮（运行由宿主页接线）。"""

    sig_step_selected = pyqtSignal(int)     # 步骤索引（-1 = 输入/未选）
    sig_step_clicked = pyqtSignal(int)      # 每次实际单击，包括已选中的步骤
    sig_step_toggled = pyqtSignal(int, bool)
    sig_step_removed = pyqtSignal(int)
    sig_step_moved = pyqtSignal(int, int)   # source → 插入位
    sig_add_requested = pyqtSignal()
    # chip 右键唯一入口（P0-3 合一弹窗）：步骤索引 → 宿主页弹 StepEditorPopup
    sig_step_editor_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps = []
        self._display_generation = 0
        self._context = object()
        self._overflow_menus = []
        self._input_widget = None
        self._list = _ChipList(self, self)
        self._list.setFlow(QListWidget.Flow.LeftToRight)
        self._list.setWrapping(False)
        self._list.setSpacing(5)
        self._list.setFixedHeight(38)
        self._list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 手势替代路径：右键菜单 + Delete 键（guidelines 要求拖拽必须有
        # 点击/键盘替代——PipelineList 的等价能力随右栏隐藏后在此补齐）
        self._list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self._delete_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Delete), self._list,
            context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(
            lambda: self._on_chip_deleted(self._list.currentRow()))
        # 深浅主题下与卡片底色融合；选中高亮交给滑动胶囊（Qt 默认蓝块弃用）
        self._apply_list_qss()
        self._list.currentRowChanged.connect(self._on_current_row)
        self._list.horizontalScrollBar().valueChanged.connect(
            lambda _value: self._overlay_timer.start(0))

        self._add_btn = ToolButton(FIF.ADD, self)
        self._add_btn.setFixedSize(24, 24)
        self._add_btn.setToolTip('添加处理步骤（弹出方法选择器，双击加入链尾）')
        self._add_btn.setAccessibleName('添加处理步骤')
        self._add_btn.clicked.connect(self.sig_add_requested)
        self._dirty_label = CaptionLabel('已修改 · 点运行更新', self)
        from ui.design_tokens import color
        self._dirty_label.setStyleSheet(f'color:{color("warning")}')
        self._dirty_label.setVisible(False)
        self._run_btn = PrimaryPushButton('运行', self)
        self._run_btn.setFixedWidth(76)
        self._run_btn.setToolTip('按当前处理链运行（Ctrl+R）；改算法/参数后需运行才更新结果')
        # flash 复位用可停的 QTimer——旧裸 singleShot 引用无法取消：1.2s
        # 内再次点运行会把「取消」态错误复位成「运行」（深查 P3 竞态）
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(
            lambda: self._run_btn.setText('运行'))
        # 浮层重算同理走可停 QTimer：裸 singleShot(0, ...) 在 strip 被
        # deleteLater 后回调仍会派发（悬空引用），炸在下一个 processEvents
        # 的测试/窗口上（CI gui-linux/windows 全量套件确定性段错误根因）
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.timeout.connect(self._update_overlays)
        # 空链引导（2026-10-09 深查 P2）：空链时列表区一片空白，新用户
        # 不知道从哪开始——浮层文字提示（不挡鼠标，＋ 钮照常可点）
        self._empty_hint = CaptionLabel('从左侧方法库添加算法',
                                        self._list.viewport())
        self._empty_hint.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._empty_hint.setStyleSheet(f'color:{color("text_muted")}')
        self._empty_hint.setVisible(False)
        # 溢出胶囊（2026-10-09 深查 P1）：浮在列表右缘，放不下的 chip 折叠
        # 成 +N，点击弹全部步骤菜单。做成视口浮层而非行内控件——不挤占
        # 本就紧张的 chip 空间（行内加控件会反过来缩小视口，测量振荡）
        self._overflow_btn = QPushButton(self._list.viewport())
        self._overflow_btn.setObjectName('overflow_chip')
        self._overflow_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._overflow_btn.setVisible(False)
        self._overflow_btn.clicked.connect(self._on_overflow_menu)
        self._style_overflow_btn()

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(constants.CARD_SPACING)
        row.addWidget(self._list, 1)
        row.addWidget(self._dirty_label)
        row.addWidget(self._add_btn)
        row.addWidget(self._run_btn)
        self._init_pill()

    def run_button(self):
        """运行钮：由宿主页接线（Ctrl+R 与此处同一入口）。"""
        return self._run_btn

    def set_run_button_visible(self, visible: bool) -> None:
        """Hosts with a shared header action can reclaim the strip's button space."""
        self._run_btn.setVisible(visible)

    def step_anchor(self, index: int):
        """Return the current rendered chip for the host's parameter popup."""
        item = self._list.item(index) if 0 <= index < self._list.count() else None
        return self._list.itemWidget(item) if item is not None else None

    # ------------------------------------------------------------ 主题
    def _apply_list_qss(self) -> None:
        """chip 容器底色清零 + 键盘焦点环（令牌随当前主题取值）。"""
        from ui.design_tokens import color
        self._list.setStyleSheet(
            'QListWidget{background:transparent;border:none;}'
            'QListWidget::item{background:transparent;border:none;}'
            'QListWidget::item:selected{background:transparent;'
            'border:none;color:palette(window-text);}'
            'QListWidget::item:hover{background:transparent;}'
            # 键盘焦点可见（WCAG 2.4.7）：当前行由滑动胶囊指示，列表自身
            # 再给一圈 focus 环，Tab 进来时知道焦点在哪
            f'QListWidget:focus{{border:1px solid {color("border_focus")};'
            f'border-radius:4px;}}')

    def apply_theme(self, dark: bool) -> None:
        """主题切换重刷令牌色（主窗鸭子类型派发，见 main_window）。

        chip 的 QSS/状态点在 ``set_enabled_visual`` 时按**当时**主题取
        令牌——换主题后必须逐 chip 重跑；胶囊、焦点环、脏标记同理。
        """
        from ui.design_tokens import color, radius, rgba
        for i in range(self._list.count()):
            chip = self._list.itemWidget(self._list.item(i))
            if chip is None:
                continue
            idx = chip.index
            chip.set_enabled_visual(
                self._steps[idx]['enabled']
                if 0 <= idx < len(self._steps) else True)
        self._dirty_label.setStyleSheet(f'color:{color("warning")}')
        self._empty_hint.setStyleSheet(f'color:{color("text_muted")}')
        self._style_overflow_btn()
        self._pill.setStyleSheet(
            f'background:{rgba("primary", 0.08)};'
            f'border-radius:{radius("lg")}px')
        self._apply_list_qss()

    # ------------------------------------------- 选中指示条（滑动胶囊）
    def _init_pill(self) -> None:
        """选中 chip 的滑动指示胶囊（动效移植②，来自 BeUI「Tabs」思路）。

        半透明覆盖层 + geometry 动画（180ms OutCubic）；不吃鼠标事件，
        位于 chip 之下（lower）以免盖住文字。
        """
        self._pill = QWidget(self._list.viewport())
        self._pill.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        from ui.design_tokens import radius, rgba
        # 原为硬编码的旧 Windows 蓝（与 primary 令牌漂移）→ 统一到
        # primary，改强调色只需改令牌一处
        self._pill.setStyleSheet(
            f'background:{rgba("primary", 0.08)};'
            f'border-radius:{radius("lg")}px')
        self._pill.hide()
        self._pill_anim = QPropertyAnimation(self._pill, b'geometry')
        self._pill_anim.setDuration(180)
        self._pill_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _move_pill(self, row: int) -> None:
        for index in range(self._list.count()):
            chip = self._list.itemWidget(self._list.item(index))
            if chip is not None:
                chip.set_selected_visual(index == row)
        item = self._list.item(row)
        rect = self._list.visualItemRect(item) if item is not None else None
        if rect is None or not rect.isValid():
            self._pill.hide()
            return
        if self._pill.isHidden():
            self._pill.setGeometry(rect)
            self._pill.show()
            self._pill.lower()
            return
        self._pill_anim.stop()
        self._pill_anim.setStartValue(self._pill.geometry())
        self._pill_anim.setEndValue(rect)
        self._pill_anim.start()

    # ------------------------------------------------ 运行钮：取消就近翻转
    def _style_overflow_btn(self) -> None:
        """+N 胶囊样式：左缘渐隐过渡——底下被遮的 chip 文字渐隐入胶囊，
        不再硬切（半遮字像故障）；色值走令牌，apply_theme 重刷。"""
        from ui.design_tokens import color
        bg = color('bg_subtle')
        self._overflow_btn.setStyleSheet(
            'QPushButton#overflow_chip{'
            'background:qlineargradient(x1:0,y1:0,x2:1,y2:0,'
            f'stop:0 rgba(0,0,0,0), stop:0.35 {bg}, stop:1 {bg});'
            f'color:{color("text_secondary")};'
            f'border:1px solid {color("border_default")};'
            'border-radius:13px;padding:3px 10px;}'
            'QPushButton#overflow_chip:hover{'
            f'border:1px solid {color("border_focus")};}}')

    def set_running(self, running: bool) -> None:
        """运行态：运行钮**就近翻转**为「取消」（红描边，同一位置）。

        spinner 文字动画退役（2026-10-09 深查）：进行中反馈由进度行承担，
        按钮专职取消入口——原取消钮在可能收起的左栏执行卡里，与顶部
        运行钮空间分离（深查实锤的工作流断点）。点击仍走 run_button 的
        clicked，由宿主页按运行态分流转发为取消。
        """
        self._flash_timer.stop()
        if running:
            from ui.design_tokens import color, rgba
            self._run_btn.setEnabled(True)
            self._run_btn.setText('取消')
            self._run_btn.setToolTip('取消正在运行的处理任务')
            self._run_btn.setStyleSheet(
                'PrimaryPushButton{background:transparent;'
                f'color:{color("error")};'
                f'border:1px solid {color("error")};border-radius:5px;}}'
                'PrimaryPushButton:hover{'
                f'background:{rgba("error", 0.08)};}}'
                'PrimaryPushButton:pressed{'
                f'background:{rgba("error", 0.16)};}}')
        else:
            self._run_btn.setEnabled(True)
            self._run_btn.setText('运行')
            self._run_btn.setToolTip('按当前处理链运行（Ctrl+R）；'
                                     '改算法/参数后需运行才更新结果')
            self._run_btn.setStyleSheet('')

    def set_dirty(self, dirty: bool) -> None:
        """链/参数已改但结果未重算：琥珀色提示（运行后清除）。"""
        self._dirty_label.setVisible(bool(dirty))

    def flash_success(self) -> None:
        """运行成功：按钮短暂变「完成」（不带勾），再回到「运行」。"""
        self._run_btn.setText('完成')
        self._flash_timer.start(1200)

    # ---------------------------------------------------------------- 数据
    def set_input_widget(self, widget) -> None:
        """条首承载「输入」选择（测线 / 成果下拉）——不单独占一行纵向。"""
        if widget is self._input_widget:
            return
        if self._input_widget is not None:
            self.layout().removeWidget(self._input_widget)
            self._input_widget.hide()
        self._input_widget = widget
        if widget is None:
            return
        self.layout().insertWidget(0, widget)

    def set_context(self, context) -> None:
        """Bind gestures to an opaque host draft identity without rebuilding chips."""
        if context == self._context:
            return
        self._context = context
        self._invalidate_interactions()

    def _invalidate_interactions(self) -> None:
        self._display_generation += 1
        if self._list._drag_active:
            QDrag.cancel()
        self._list._press_source = -1
        self._list._drop_marker.hide()
        for menu in self._overflow_menus:
            for sub in menu._step_submenus:
                sub.close()
            menu.close()

    def set_steps(self, steps) -> None:
        """Render {label, enabled, param_summary}; the host owns draft state."""
        display = [{'label': str(s.get('label') or s.get('method_id', '')),
                    'enabled': bool(s.get('enabled', True)),
                    'param_summary': str(s.get('param_summary') or '')}
                   for s in (steps or [])]
        if display == self._steps:
            self._overlay_timer.start(0)
            return
        self._invalidate_interactions()
        self._steps = display
        # Parameter edits keep item widgets and keyboard focus stable.
        if self._list.count() == len(display):
            for i, step in enumerate(display):
                item = self._list.item(i)
                chip = self._list.itemWidget(item)
                chip.set_content(f'{i + 1} {step["label"]}',
                                 step['param_summary'], step['enabled'])
                item.setSizeHint(chip.sizeHint())
            self._overlay_timer.start(0)
            return
        self._rebuild()

    def _rebuild(self) -> None:
        prev = self._list.currentRow()
        self._list.blockSignals(True)
        self._list.clear()
        for i, step in enumerate(self._steps):
            self._add_chip(i, f'{i + 1} {step["label"]}', step['enabled'],
                           step['param_summary'])
        self._list.blockSignals(False)
        if 0 <= prev < len(self._steps):
            self._list.setCurrentRow(prev)
        if self._list.currentRow() >= 0:
            self._move_pill(self._list.currentRow())
        # 布局一拍后重算浮层（此时 visualItemRect 才有效）
        self._overlay_timer.start(0)

    def _add_chip(self, index: int, label: str, enabled: bool,
                  param_summary: str = '') -> None:
        item = QListWidgetItem()
        chip = _Chip(index, label, enabled, self, self._list, param_summary)
        item.setSizeHint(chip.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, chip)

    # ------------------------------------------------------------ 手势回调
    def _on_current_row(self, row: int) -> None:
        self._move_pill(row)
        self.sig_step_selected.emit(row)         # row 即步骤索引

    def _on_dot_clicked(self, index: int) -> None:
        if self.isEnabled() and 0 <= index < len(self._steps):
            self.sig_step_toggled.emit(index, not self._steps[index]['enabled'])

    def _on_chip_deleted(self, index: int) -> None:
        if self.isEnabled() and 0 <= index < len(self._steps):
            self.sig_step_removed.emit(index)

    # -------------------------------------- 右键（替代路径 → 合一弹窗）
    def _on_context_menu(self, pos) -> None:
        """右键 chip → 置当前行 + 发 sig_step_editor_requested（P0-3 合一
        弹窗唯一入口；旧四动作菜单删除，动作全部收敛进步骤编辑弹窗）。"""
        # 原 rowAt 系 QListView 不存在的方法（QTableView 才有）——右键即
        # AttributeError 的潜伏 bug，随本改造一并修正为 itemAt
        item = self._list.itemAt(pos)
        row = self._list.row(item) if item is not None else -1
        index = row                          # row 即步骤索引
        self._request_editor(index)

    def _request_editor(self, index: int) -> None:
        if not self.isEnabled() or index < 0 or index >= len(self._steps):
            return
        self._list.setCurrentRow(index)
        self._move_pill(index)
        self.sig_step_editor_requested.emit(index)

    def _list_drop_event(self, event, *, source=None, generation=None) -> None:
        """Emit the pressed source and insertion boundary; never move Qt items."""
        # Legacy direct callers supply a synthetic event with the selected row.
        # Real Qt drops always pass the press-captured source from _ChipList.
        if source is None:
            source = self._list.currentRow()
            generation = self._display_generation
        self._list._drop_marker.hide()
        if (not self.isEnabled() or source < 0 or source >= len(self._steps)
                or generation != self._display_generation):
            if hasattr(event, 'ignore'):
                event.ignore()
            return
        target = self._list.insertion_point(event.position().toPoint())
        self._list._press_source = -1
        if target in (source, source + 1):
            if hasattr(event, 'ignore'):
                event.ignore()
            return
        event.setDropAction(Qt.DropAction.IgnoreAction)
        event.accept()
        self.sig_step_moved.emit(source, target)

    def select_step(self, index: int) -> None:
        """外部选中某步骤（参数区跟随，效果同用户点 chip）；-1 = 无选中。"""
        row = index if 0 <= index < len(self._steps) else -1
        self._list.blockSignals(True)
        self._list.setCurrentRow(row)
        self._list.blockSignals(False)
        if row >= 0:
            self._list.scrollToItem(self._list.item(row))
        self._move_pill(row)

    # --------------------------------------- 浮层：空态提示 + 溢出胶囊
    def _update_overlays(self) -> None:
        """重算空态提示与 +N 溢出胶囊（_rebuild / 尺寸变化后调用）。

        溢出判定 = 右缘被视口裁掉的 chip 数（wrapping=False + 滚动条常关，
        被裁即不可达）——半可见的 chip 也计入，其 ✕/圆点可能已在视口外。
        """
        lst = self._list
        self._move_pill(lst.currentRow())
        vp = lst.viewport()
        n = lst.count()
        self._empty_hint.setVisible(n == 0)
        hidden = 0
        if n and vp.width() > 0:
            for i in range(n):
                rect = lst.visualItemRect(lst.item(i))
                if rect.right() > vp.width() - 1 or rect.left() < 0:
                    hidden += 1
        self._overflow_btn.setVisible(hidden > 0)
        if hidden:
            self._overflow_btn.setText(f'+{hidden}')
            self._overflow_btn.setToolTip(
                f'共 {n} 个步骤，还有 {hidden} 个未显示')
            self._overflow_btn.adjustSize()
            self._overflow_btn.move(
                max(0, vp.width() - self._overflow_btn.width() - 2),
                max(0, (vp.height() - self._overflow_btn.height()) // 2))
            self._overflow_btn.raise_()
        self._empty_hint.adjustSize()
        self._empty_hint.move(
            12, max(0, (vp.height() - self._empty_hint.height()) // 2))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._overlay_timer.start(0)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._overlay_timer.start(0)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange and not self.isEnabled():
            self._invalidate_interactions()

    def _on_overflow_menu(self) -> None:
        if not self.isEnabled():
            return
        menu = self._build_overflow_menu()
        # Fluent exec() returns immediately. make_menu owns disposal after the
        # close signal and synchronous action dispatch have both completed.
        menu.exec(self._overflow_btn.mapToGlobal(
            self._overflow_btn.rect().bottomLeft()))

    def _build_overflow_menu(self) -> RoundMenu:
        """+N 菜单：每步骤一个子菜单 = 「选中 / 编辑」两项（P0-3 定案 Q3：
        结构动作全进合一弹窗，不维护两套入口——被裁掉的步骤与可见 chip
        手势等价可达，guidelines：内容不得因布局不可达）。
        构建与弹出拆开，便于测试菜单结构。
        """
        menu = make_menu(self)
        self._overflow_menus.append(menu)
        # Drop tracking only after native destruction, never at exec() return.
        # Capture the list and identity, without retaining self or menu again.
        def forget_menu(_obj=None, key=id(menu), menus=self._overflow_menus):
            menus[:] = [existing for existing in menus if id(existing) != key]
        menu.destroyed.connect(forget_menu)
        generation = self._display_generation
        subs = []
        for i, step in enumerate(self._steps):
            label = f'{i + 1} {step["label"]}'
            if not step['enabled']:
                label += '（已禁用）'
            sub = RoundMenu(label, menu)
            # triggered 会带 checked 实参：lambda 必须显式吞掉，
            # 否则 ix 被 False 覆盖（原「选中」对第 0 步以外的步骤
            # 实际选中第 0 行的潜伏 bug，本次一并修正）
            add_action(sub, FIF.VIEW, '选中',
                       lambda _checked=False, ix=i, gen=generation:
                       self._overflow_action(ix, gen, edit=False))
            add_action(sub, FIF.EDIT, '编辑',
                       lambda _checked=False, ix=i, gen=generation:
                       self._overflow_action(ix, gen, edit=True))
            menu.addMenu(sub)
            subs.append(sub)
        # qfw 未暴露子菜单清单——挂引用供测试访问（menu._step_submenus）
        menu._step_submenus = subs
        return menu

    def _overflow_action(self, index: int, generation: int, *, edit: bool) -> None:
        if (not self.isEnabled() or generation != self._display_generation
                or not 0 <= index < len(self._steps)):
            return
        if edit:
            # The host owns selection while opening the editor. Keep this a
            # single edit request, as in the existing overflow API contract.
            self.sig_step_editor_requested.emit(index)
        else:
            self._list.setCurrentRow(index)
            self._list.scrollToItem(self._list.item(index))
