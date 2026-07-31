"""PipelineList — 处理链步骤列表控件（SPEC §5.5）。

每行：序号 + 方法名 + 启用 CheckBox + ↑ ↓ 删除 小按钮；
选中行高亮并发 sig_step_selected（-1 无）；任何行操作后发 sig_changed。

右键菜单（RoundMenu）：上移 / 下移 / 启用-禁用切换 / 删除，
与行内小按钮等价（小按钮难发现的补偿路径）；Delete 键删除当前行。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QVBoxLayout, QWidget)
from qfluentwidgets import CheckBox, TransparentToolButton
from qfluentwidgets import FluentIcon as FIF

from ui.widgets.context_menus import add_action, make_menu


class _StepRow(QWidget):
    """单个步骤行。"""

    def __init__(self, index, step, parent=None):
        super().__init__(parent)
        self.index_label = QLabel(str(index + 1), self)
        self.index_label.setMinimumWidth(20)
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label = QLabel(step.get('label') or step.get('method_id', ''),
                                 self)
        self.enabled_box = CheckBox('启用', self)
        self.enabled_box.setChecked(bool(step.get('enabled', True)))
        self.up_btn = TransparentToolButton(FIF.UP, self)
        self.down_btn = TransparentToolButton(FIF.DOWN, self)
        self.del_btn = TransparentToolButton(FIF.DELETE, self)
        for btn, tip in ((self.up_btn, '上移'), (self.down_btn, '下移'),
                         (self.del_btn, '删除')):
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


class PipelineList(QWidget):
    """处理链编辑列表。"""

    sig_changed = pyqtSignal()
    sig_step_selected = pyqtSignal(int)   # 当前编辑步索引，-1 无

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps = []
        self._list = QListWidget(self)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self._delete_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Delete), self._list,
            context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(
            lambda: self._remove_step(self._list.currentRow()))

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
        self._rebuild()

    def steps(self) -> list:
        return [{'method_id': s['method_id'], 'label': s['label'],
                 'params': dict(s['params']), 'enabled': s['enabled']}
                for s in self._steps]

    def add_step(self, method_id: str, label: str, params: dict) -> None:
        self._steps.append({'method_id': method_id, 'label': label,
                            'params': dict(params or {}), 'enabled': True})
        self._rebuild(select=len(self._steps) - 1)
        self.sig_changed.emit()

    # ------------------------------------------------------------- 内部
    def _rebuild(self, select=None):
        self._list.blockSignals(True)
        self._list.clear()
        for i, step in enumerate(self._steps):
            item = QListWidgetItem()
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

    def _on_context_menu(self, pos) -> None:
        row = self._list.rowAt(pos.y())
        if row < 0 or row >= len(self._steps):
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
        del self._steps[idx]
        self._rebuild()
        self.sig_changed.emit()
