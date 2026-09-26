# -*- coding: utf-8 -*-
"""ChainStrip — 处理链极简 chip 条（处理页 v2 顶部）。

设计（2026-09-25 v2）：
- 一行：``输入 → 算法1 → 算法2 …``，chip 只写**算法名**（序号与位置自明，
  不写「第 N 步/处理链」等冗余字样）；
- **hover 才显**启用圆点与 ✕（常驻噪音更少），链尾虚线 ＋ 追加；
- 拖拽排序：启用 InternalMove 拿拖拽视觉，落点由 _ChipList.dropEvent
  委托宿主接管（不调 Qt 默认实现——挪 item 会丢 setItemWidget 行控件，
  与 PipelineList 同款坑）；
- **手势替代路径**（guidelines：拖拽需有点击/键盘替代）：右键菜单
  （上移/下移/启用-禁用/删除）+ Delete 键删当前 chip；
- 「输入」chip 固定在首位、不可拖不可删。

ChainStrip 只做展示与手势，**步骤数据仍由宿主页维护**（这里不存 steps）。
"""
from PyQt6.QtCore import (QEasingCurve, QPropertyAnimation, Qt,
                          QTimer, pyqtSignal)
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QWidget)
from qfluentwidgets import (CaptionLabel, FluentIcon as FIF,
                            PrimaryPushButton, ToolButton)

from ui import constants
from ui.widgets.context_menus import add_action, make_menu


class _Chip(QWidget):
    """单个步骤 chip：胶囊造型，算法名（常显）+ 启用圆点 / ✕（hover 显）。

    名称精简：目录显示名常带英文后缀（如「零时校正 (Dewow)」），chip 上
    只保留中文名（tooltip 保留全名）——长名省略号截断，不顶出滚动条。
    """

    _MAX_NAME_PX = 132

    def __init__(self, index: int, label: str, enabled: bool, host,
                 parent=None):
        super().__init__(parent)
        self.index = index                  # -1 = 输入 chip
        self._host = host
        self.setObjectName('chip')
        # 普通 QWidget 子类必须开这个才会绘制样式表背景（否则 chip 无胶囊底）
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setToolTip(label)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 4, 8, 4)
        row.setSpacing(4)
        short = label.split('（')[0].split(' (')[0].strip() or label
        self.name = QLabel(short, self)
        self.name.setMaximumWidth(self._MAX_NAME_PX)
        row.addWidget(self.name)
        self.dot_btn = ToolButton(FIF.ACCEPT, self)
        self.dot_btn.setFixedSize(14, 14)
        self.dot_btn.setCheckable(True)
        self.dot_btn.setChecked(enabled)
        self.dot_btn.setToolTip('启用 / 禁用该步骤')
        self.dot_btn.clicked.connect(
            lambda _c=False, i=index: host._on_dot_clicked(i))
        row.addWidget(self.dot_btn)
        self.del_btn = ToolButton(FIF.CLOSE, self)
        self.del_btn.setFixedSize(14, 14)
        self.del_btn.setToolTip('删除该步骤')
        self.del_btn.clicked.connect(
            lambda _c=False, i=index: host._on_chip_deleted(i))
        row.addWidget(self.del_btn)
        self.dot_btn.setVisible(False)
        self.del_btn.setVisible(False)
        self.set_enabled_visual(enabled)

    def set_enabled_visual(self, enabled: bool) -> None:
        self.name.setStyleSheet('color:#8A8A85' if not enabled else '')
        self.setStyleSheet(
            '#chip{background:rgba(128,128,128,0.16);'
            'border:1px solid rgba(128,128,128,0.22);border-radius:14px}'
            if enabled else
            '#chip{background:rgba(128,128,128,0.08);'
            'border:1px dashed rgba(128,128,128,0.30);border-radius:14px}')

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

    def dropEvent(self, event) -> None:
        self._host._list_drop_event(event)


class ChainStrip(QWidget):
    """处理链 chip 条 + 追加钮 + 运行钮（运行由宿主页接线）。"""

    sig_step_selected = pyqtSignal(int)     # 步骤索引（-1 = 输入/未选）
    sig_step_toggled = pyqtSignal(int, bool)
    sig_step_removed = pyqtSignal(int)
    sig_step_moved = pyqtSignal(int, int)   # source → 插入位
    sig_add_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps = []
        self._list = _ChipList(self, self)
        self._list.setFlow(QListWidget.Flow.LeftToRight)
        self._list.setWrapping(False)
        self._list.setSpacing(8)
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
            lambda: self._on_chip_deleted(self._list.currentRow() - 1))
        # 深浅主题下与卡片底色融合；选中高亮交给滑动胶囊（Qt 默认蓝块弃用）
        self._list.setStyleSheet(
            'QListWidget{background:transparent;border:none;}'
            'QListWidget::item{background:transparent;border:none;}'
            'QListWidget::item:selected{background:transparent;'
            'border:none;color:palette(window-text);}'
            'QListWidget::item:hover{background:transparent;}')
        self._list.currentRowChanged.connect(self._on_current_row)

        self._add_btn = ToolButton(FIF.ADD, self)
        self._add_btn.setFixedSize(22, 22)
        self._add_btn.setToolTip('添加所选算法（＋）')
        self._add_btn.clicked.connect(self.sig_add_requested)
        self._dirty_label = CaptionLabel('已修改 · 点运行更新', self)
        self._dirty_label.setStyleSheet('color:#E0A83A')
        self._dirty_label.setVisible(False)
        self._run_btn = PrimaryPushButton('运行', self)
        self._run_btn.setFixedWidth(76)
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(320)
        self._spin_timer.timeout.connect(self._tick_spin)
        self._run_btn.setToolTip('按当前处理链运行（Ctrl+R）；改算法/参数后需运行才更新结果')

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

    # ------------------------------------------- 选中指示条（滑动胶囊）
    def _init_pill(self) -> None:
        """选中 chip 的滑动指示胶囊（动效移植②，来自 BeUI「Tabs」思路）。

        半透明覆盖层 + geometry 动画（180ms OutCubic）；不吃鼠标事件，
        位于 chip 之下（lower）以免盖住文字。
        """
        self._pill = QWidget(self._list.viewport())
        self._pill.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._pill.setStyleSheet('background:rgba(0,120,212,0.20);'
                                 'border-radius:11px')
        self._pill.hide()
        self._pill_anim = QPropertyAnimation(self._pill, b'geometry')
        self._pill_anim.setDuration(180)
        self._pill_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _move_pill(self, row: int) -> None:
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

    # -------------------------------------------------- 运行钮：spinner → ✓
    def set_running(self, running: bool) -> None:
        """运行态：按钮转 spinner（点动画），结束回到「运行」。

        动效移植自 Transitions.dev 的「Spinner to check morph」思路——
        Qt 侧用轻量点动画代替旋转指示（避免引入新控件）。
        """
        if running:
            self._run_btn.setEnabled(False)
            self._run_btn.setText('运行中·')
            self._spin_phase = 0
            self._spin_timer.start(320)
        else:
            self._spin_timer.stop()
            self._run_btn.setEnabled(True)
            self._run_btn.setText('运行')

    def set_dirty(self, dirty: bool) -> None:
        """链/参数已改但结果未重算：琥珀色提示（运行后清除）。"""
        self._dirty_label.setVisible(bool(dirty))

    def flash_success(self) -> None:
        """运行成功：按钮短暂变 ✓ 完成，再回到「运行」（明确的结果反馈）。"""
        self._run_btn.setText('✓ 完成')
        QTimer.singleShot(1200, lambda: self._run_btn.setText('运行'))

    def _tick_spin(self) -> None:
        self._spin_phase = (getattr(self, '_spin_phase', 0) + 1) % 3
        self._run_btn.setText('运行中' + '·' * (self._spin_phase + 1))

    # ---------------------------------------------------------------- 数据
    def set_input_widget(self, widget) -> None:
        """条首承载「输入」选择（测线 / 成果下拉）——不单独占一行纵向。"""
        self.layout().insertWidget(0, widget)

    def set_steps(self, steps) -> None:
        """steps: [{label, enabled}]（宿主页为唯一数据源；这里只渲染）。"""
        self._steps = [{'label': s.get('label') or s.get('method_id', ''),
                        'enabled': bool(s.get('enabled', True))}
                       for s in (steps or [])]
        self._rebuild()

    def _rebuild(self) -> None:
        prev = self._list.currentRow() - 1
        self._list.blockSignals(True)
        self._list.clear()
        self._add_chip(-1, '输入', True)
        for i, step in enumerate(self._steps):
            self._add_chip(i, f'{i + 1} {step["label"]}', step['enabled'])
        self._list.blockSignals(False)
        if 0 <= prev < len(self._steps):
            self._list.setCurrentRow(prev + 1)
        if self._list.currentRow() >= 0:
            self._move_pill(self._list.currentRow())

    def _add_chip(self, index: int, label: str, enabled: bool) -> None:
        item = QListWidgetItem()
        chip = _Chip(index, label, enabled, self, self._list)
        item.setSizeHint(chip.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, chip)

    # ------------------------------------------------------------ 手势回调
    def _on_current_row(self, row: int) -> None:
        self._move_pill(row)
        self.sig_step_selected.emit(row - 1)     # 0 号是输入 chip

    def _on_dot_clicked(self, index: int) -> None:
        if 0 <= index < len(self._steps):
            self.sig_step_toggled.emit(index, not self._steps[index]['enabled'])

    def _on_chip_deleted(self, index: int) -> None:
        if 0 <= index < len(self._steps):
            self.sig_step_removed.emit(index)

    # -------------------------------------------------- 右键菜单（替代路径）
    def _on_context_menu(self, pos) -> None:
        row = self._list.rowAt(pos.y())
        if row < 1:                          # 0 号是输入 chip，无操作
            return
        index = row - 1
        if index >= len(self._steps):
            return
        self._list.setCurrentRow(row)
        self._move_pill(row)
        menu = make_menu(self)
        self._fill_step_menu(menu, index)
        menu.exec(self._list.viewport().mapToGlobal(pos))

    def _fill_step_menu(self, menu, index: int) -> None:
        """单步右键菜单动作（与 PipelineList 等价；供测试直接驱动）。"""
        enabled = bool(self._steps[index]['enabled'])
        add_action(menu, FIF.UP, '上移',
                   lambda: self.sig_step_moved.emit(index, index - 1),
                   enabled=index > 0)
        add_action(menu, FIF.DOWN, '下移',
                   lambda: self.sig_step_moved.emit(index, index + 2),
                   enabled=index < len(self._steps) - 1)
        menu.addSeparator()
        add_action(menu, FIF.ACCEPT if enabled else FIF.CANCEL,
                   '禁用' if enabled else '启用',
                   lambda: self.sig_step_toggled.emit(index, not enabled))
        menu.addSeparator()
        add_action(menu, FIF.DELETE, '删除',
                   lambda: self.sig_step_removed.emit(index))

    def _list_drop_event(self, event) -> None:
        """落点重排：源=当前选中 chip，目标按落点左右半决定插入位。"""
        source = self._list.currentRow() - 1        # 换算成步骤索引
        if source < 0 or source >= len(self._steps):
            return
        target = self._list.indexAt(event.position().toPoint()).row() - 1
        if target < 0:
            target = len(self._steps)
        item = self._list.item(max(target + 1, 0))
        if item is not None:
            rect = self._list.visualItemRect(item)
            if event.position().toPoint().x() > rect.center().x():
                target += 1
        if target in (source, source + 1):
            return
        event.setDropAction(Qt.DropAction.IgnoreAction)
        event.accept()
        self.sig_step_moved.emit(source, target)

    def select_step(self, index: int) -> None:
        """外部选中某步骤（参数区跟随，效果同用户点 chip）。"""
        row = index + 1 if 0 <= index < len(self._steps) else 0
        self._list.blockSignals(True)
        self._list.setCurrentRow(row)
        self._list.blockSignals(False)
        self._move_pill(row)
