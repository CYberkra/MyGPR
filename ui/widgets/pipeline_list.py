"""PipelineList — 处理链步骤列表控件（SPEC §5.5）。

每行：序号 + 方法名 + 启用 CheckBox + ↑ ↓ 删除 小按钮；
选中行高亮并发 sig_step_selected（-1 无）；任何行操作后发 sig_changed。

排序三条等价路径：行内小按钮 ↑↓ / 左键拖拽重排（QListWidget InternalMove，
落点带插入指示线）/ Ctrl+↑↓；方法库拖到列表任意位置插入（METHOD_MIME），
发 sig_method_drop_requested 由页面层补默认参数后 insert_step。

右键菜单（RoundMenu）：上移 / 下移 / 启用-禁用切换 / 删除，
与行内小按钮等价（小按钮难发现的补偿路径）；Delete 键删除当前行。

删除防误：三种删除路径统一走 _remove_step，删除前把步骤快照压入
撤销栈（Ctrl+Z 撤销删除，右键菜单同款入口）；set_steps 整体替换时
清空撤销栈（外部重设链语义 = 新编辑会话）。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QSizePolicy,
                             QVBoxLayout, QWidget)
from qfluentwidgets import CheckBox, TransparentToolButton
from qfluentwidgets import FluentIcon as FIF

from ui.widgets.context_menus import add_action, make_menu

# 方法库 → 处理链 拖拽的自定义 MIME（载荷 = method_id 的 utf-8 字节）
METHOD_MIME = 'application/x-mygpr-method-id'


class _ElidedLabel(QLabel):
    """宽度不足时右侧省略号截断，完整文本放 tooltip。

    步骤行空间被序号/启用框/按钮挤占后，长方法名（如"周期条带伪影
    抑制（实验）"）不再顶出横向滚动条；悬浮可看到全名。
    """

    def __init__(self, text='', parent=None):
        super().__init__(text, parent)
        self._full_text = text
        self.setToolTip(text)
        self.setMinimumWidth(0)
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.setSizePolicy(policy)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideRight,
            max(self.width() - 4, 10))
        super().setText(elided)


class _StepRow(QWidget):
    """单个步骤行。"""

    def __init__(self, index, step, parent=None):
        super().__init__(parent)
        self.index_label = QLabel(str(index + 1), self)
        self.index_label.setMinimumWidth(20)
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label = _ElidedLabel(step.get('label') or step.get('method_id', ''),
                                       self)
        self.enabled_box = CheckBox('启用', self)
        self.enabled_box.setChecked(bool(step.get('enabled', True)))
        self.up_btn = TransparentToolButton(FIF.UP, self)
        self.down_btn = TransparentToolButton(FIF.DOWN, self)
        self.del_btn = TransparentToolButton(FIF.DELETE, self)
        for btn, tip in ((self.up_btn, '上移'), (self.down_btn, '下移'),
                         (self.del_btn, '删除（Ctrl+Z 可撤销）')):
            btn.setFixedSize(28, 28)
            btn.setToolTip(tip)

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(4)
        row.addWidget(self.index_label)
        row.addWidget(self.name_label, 1)
        row.addWidget(self.enabled_box)
        row.addWidget(self.up_btn)
        row.addWidget(self.down_btn)
        row.addWidget(self.del_btn)


class _PipelineListWidget(QListWidget):
    """内部列表：行内拖拽重排 + 接收方法库拖入。

    InternalMove 内部移动后 Qt 会丢弃行控件（setItemWidget 不随行迁移），
    因此移动完成统一交回 PipelineList 按 item 携带的原始序号重排数据并
    整体重建——行控件总是新鲜的，不存在半迁移状态。
    """

    def __init__(self, owner: 'PipelineList'):
        super().__init__(owner)
        self._owner = owner
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(METHOD_MIME):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(METHOD_MIME):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasFormat(METHOD_MIME):
            method_id = bytes(event.mimeData().data(METHOD_MIME)).decode('utf-8')
            self._owner._on_method_dropped(
                method_id, self._row_at(event.position().toPoint()))
            event.acceptProposedAction()
            return
        dragged = self.currentItem()
        dragged_key = (dragged.data(Qt.ItemDataRole.UserRole)
                       if dragged is not None else None)
        super().dropEvent(event)
        self._owner._sync_order_after_drag(dragged_key)

    def _row_at(self, pos) -> int:
        """落点 → 插入行：空白处 = 末尾；行上半 = 该行前，下半 = 该行后。"""
        item = self.itemAt(pos)
        if item is None:
            return self.count()
        rect = self.visualItemRect(item)
        return self.row(item) + (1 if pos.y() > rect.center().y() else 0)


class PipelineList(QWidget):
    """处理链编辑列表。"""

    sig_changed = pyqtSignal()
    sig_step_selected = pyqtSignal(int)   # 当前编辑步索引，-1 无
    # 方法库拖入（method_id, 插入行）；页面层补默认参数后调 insert_step
    sig_method_drop_requested = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps = []
        self._undo_stack = []   # 删除撤销栈：[(原索引, 步骤快照 dict)]，栈顶 = 最近删除
        self._list = _PipelineListWidget(self)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self._delete_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Delete), self._list,
            context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(
            lambda: self._remove_step(self._list.currentRow()))
        # Ctrl+Z 撤销删除（与 Delete 同一 WidgetWithChildrenShortcut 作用域）
        self._undo_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Undo), self._list,
            context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._undo_shortcut.activated.connect(self.undo_delete)
        for key, delta in (('Ctrl+Up', -1), ('Ctrl+Down', 1)):
            shortcut = QShortcut(QKeySequence(key), self._list,
                                 context=Qt.ShortcutContext
                                 .WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda d=delta: self._move_step(self._list.currentRow(), d))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)

    # ------------------------------------------------------------- 数据
    def set_steps(self, steps) -> None:
        """steps: [{method_id,label,params(dict),enabled}]。"""
        self._steps = []
        for s in (steps or []):
            self._steps.append({
                'method_id': s.get('method_id', ''),
                'label': s.get('label', ''),
                'params': dict(s.get('params') or {}),
                'enabled': bool(s.get('enabled', True)),
            })
        self._undo_stack.clear()   # 整体替换 = 新编辑会话，旧撤销快照作废
        self._rebuild()

    def steps(self) -> list:
        return [{'method_id': s['method_id'], 'label': s['label'],
                 'params': dict(s['params']), 'enabled': s['enabled']}
                for s in self._steps]

    def count(self) -> int:
        """处理链步骤数。"""
        return len(self._steps)

    def select_step(self, index: int) -> None:
        """选中指定步骤行（效果同用户点选，发 sig_step_selected 刷新参数表单）。"""
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)

    def add_step(self, method_id: str, label: str, params: dict) -> None:
        self.insert_step(len(self._steps), method_id, label, params)

    def insert_step(self, index: int, method_id: str, label: str,
                    params: dict) -> None:
        """在 index 处插入步骤（越界自动夹取到两端），插入后选中该步。"""
        index = max(0, min(int(index), len(self._steps)))
        self._steps.insert(index, {'method_id': method_id, 'label': label,
                                   'params': dict(params or {}),
                                   'enabled': True})
        self._rebuild(select=index)
        self.sig_changed.emit()

    def update_step_params(self, index: int, params: dict) -> bool:
        """直接更新第 index 步的 params 并重绘该行，保持选中状态不变。"""
        if not (0 <= index < len(self._steps)):
            return False
        self._steps[index]['params'] = dict(params or {})
        self._rebuild(select=index)
        self.sig_changed.emit()
        return True

    # ------------------------------------------------------------- 内部
    def _rebuild(self, select=None):
        self._list.blockSignals(True)
        self._list.clear()
        for i, step in enumerate(self._steps):
            item = QListWidgetItem()
            # 原始序号随行：InternalMove 后按 item 序号回收新顺序
            # （dict 经 QVariant 往返会被深拷贝，引用不可靠，序号不会）
            item.setData(Qt.ItemDataRole.UserRole, i)
            row = _StepRow(i, step, self._list)
            row.enabled_box.setChecked(step['enabled'])
            row.enabled_box.stateChanged.connect(
                lambda _state, idx=i: self._on_enabled_toggled(idx))
            row.up_btn.clicked.connect(
                lambda _checked=False, idx=i: self._move_step(idx, -1))
            row.down_btn.clicked.connect(
                lambda _checked=False, idx=i: self._move_step(idx, 1))
            row.del_btn.clicked.connect(
                lambda _checked=False, idx=i: self._remove_step(idx))
            item.setSizeHint(row.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
        self._list.blockSignals(False)
        if select is not None and 0 <= select < self._list.count():
            self._list.setCurrentRow(select)
        elif self._list.count() == 0:
            self.sig_step_selected.emit(-1)

    def _on_row_changed(self, row):
        self.sig_step_selected.emit(row if row >= 0 else -1)

    # ------------------------------------------------------------- 拖拽
    def _on_method_dropped(self, method_id: str, row: int) -> None:
        """方法库拖入 → 转发页面层（补默认参数后 insert_step）。"""
        self.sig_method_drop_requested.emit(method_id, row)

    def _sync_order_after_drag(self, dragged_key) -> None:
        """InternalMove 落点后：按 item 携带的原始序号重排数据并整体重建。

        行控件不随行迁移，必须重建；被拖步骤保持选中。序号集合对不上
        （拖拽被打断等异常）时按原数据重建，不采纳列表序。
        """
        order = [self._list.item(i).data(Qt.ItemDataRole.UserRole)
                 for i in range(self._list.count())]
        if (len(order) != len(self._steps)
                or any(not isinstance(k, int) for k in order)
                or sorted(order) != list(range(len(self._steps)))):
            self._rebuild(select=self._list.currentRow())
            return
        self._steps = [self._steps[k] for k in order]
        select = order.index(dragged_key) if dragged_key in order else None
        self._rebuild(select=select)
        self.sig_changed.emit()

    def _on_context_menu(self, pos) -> None:
        row = self._list.rowAt(pos.y())
        if row < 0 or row >= len(self._steps):
            # 空白区右键：仍提供撤销删除入口（行删除后手滑点空白也能撤销）
            menu = make_menu(self)
            add_action(menu, FIF.RETURN, '撤销删除（Ctrl+Z）',
                       self.undo_delete, enabled=bool(self._undo_stack))
            menu.exec(self._list.viewport().mapToGlobal(pos))
            return
        self._list.setCurrentRow(row)
        enabled = bool(self._steps[row]['enabled'])
        menu = make_menu(self)
        add_action(menu, FIF.UP, '上移',
                   lambda: self._move_step(row, -1), enabled=row > 0)
        add_action(menu, FIF.DOWN, '下移',
                   lambda: self._move_step(row, 1),
                   enabled=row < len(self._steps) - 1)
        menu.addSeparator()
        add_action(menu, FIF.ACCEPT if enabled else FIF.CANCEL,
                   '禁用' if enabled else '启用',
                   lambda: self._toggle_enabled(row))
        menu.addSeparator()
        add_action(menu, FIF.DELETE, '删除',
                   lambda: self._remove_step(row))
        if self._undo_stack:
            add_action(menu, FIF.RETURN, '撤销删除（Ctrl+Z）',
                       self.undo_delete)
        menu.exec(self._list.viewport().mapToGlobal(pos))

    def _toggle_enabled(self, idx) -> None:
        if 0 <= idx < len(self._steps):
            self._steps[idx]['enabled'] = not self._steps[idx]['enabled']
            self._rebuild(select=idx)
            self.sig_changed.emit()

    def _on_enabled_toggled(self, idx):
        if 0 <= idx < len(self._steps):
            item = self._list.item(idx)
            row = self._list.itemWidget(item)
            self._steps[idx]['enabled'] = row.enabled_box.isChecked()
            self.sig_changed.emit()

    def _move_step(self, idx, delta):
        target = idx + delta
        if not (0 <= idx < len(self._steps)):
            return
        if not (0 <= target < len(self._steps)):
            return
        self._steps[idx], self._steps[target] = \
            self._steps[target], self._steps[idx]
        self._rebuild(select=target)
        self.sig_changed.emit()

    def _remove_step(self, idx):
        if not (0 <= idx < len(self._steps)):
            return
        # 撤销快照：深拷贝 params，防后续编辑污染已删步骤的恢复数据
        snapshot = dict(self._steps[idx])
        snapshot['params'] = dict(self._steps[idx].get('params') or {})
        self._undo_stack.append((idx, snapshot))
        if len(self._undo_stack) > 50:
            del self._undo_stack[0]
        del self._steps[idx]
        self._rebuild()
        self.sig_changed.emit()

    def can_undo(self) -> bool:
        """是否有可撤销的删除（右键菜单启用态/测试用）。"""
        return bool(self._undo_stack)

    def undo_delete(self) -> bool:
        """撤销最近一次删除：步骤回插到原索引（越界夹取），保持原启用态。

        返回 True = 已恢复；栈空返回 False。
        """
        if not self._undo_stack:
            return False
        idx, step = self._undo_stack.pop()
        idx = max(0, min(int(idx), len(self._steps)))
        self._steps.insert(idx, dict(step))
        self._rebuild(select=idx)
        self.sig_changed.emit()
        return True
