"""Batch application of the current processing draft to persisted raw line inputs."""

from copy import deepcopy

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QListWidget, QListWidgetItem, QVBoxLayout
from qfluentwidgets import CheckBox, LineEdit, PrimaryPushButton, PushButton

from ui.page_scaffold import card_title, make_hint


class BatchProcessingDialog(QDialog):
    def __init__(self, lines, steps, parent=None):
        super().__init__(parent)
        self._steps = deepcopy(list(steps or []))
        self.setWindowTitle('批量应用处理链')
        self.setMinimumSize(520, 480)
        layout = QVBoxLayout(self)
        title = card_title('选择目标测线')
        title.setParent(self)
        layout.addWidget(title)
        hint = make_hint(
            f'将当前 {len(self._steps)} 步处理链应用到各测线的原始输入；每条测线独立运行，失败不阻断其他测线。',
            parent=self,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._lines_list = QListWidget(self)
        for line in lines or []:
            line_id = str(getattr(line, 'line_id', '') or '')
            if not line_id:
                continue
            name = str(getattr(line, 'name', '') or line_id)
            item = QListWidgetItem(f'{line_id} · {name}' if name != line_id else line_id)
            item.setData(Qt.ItemDataRole.UserRole, line_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._lines_list.addItem(item)
        layout.addWidget(self._lines_list, 1)
        self._result_name_edit = LineEdit(self)
        self._result_name_edit.setPlaceholderText('可选成果前缀；实际名称会自动追加测线号')
        layout.addWidget(self._result_name_edit)
        self._save_intermediates_check = CheckBox('保存每步中间成果（增加内存与磁盘占用）', self)
        self._save_intermediates_check.setChecked(True)
        layout.addWidget(self._save_intermediates_check)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._cancel_btn = PushButton('取消', self)
        self._run_btn = PrimaryPushButton('开始批量处理', self)
        self._cancel_btn.clicked.connect(self.reject)
        self._run_btn.clicked.connect(self._accept_if_valid)
        buttons.addWidget(self._cancel_btn)
        buttons.addWidget(self._run_btn)
        layout.addLayout(buttons)

    def selected_line_ids(self):
        return [
            str(self._lines_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self._lines_list.count())
            if self._lines_list.item(row).checkState() == Qt.CheckState.Checked
        ]

    def payload(self):
        return {
            'line_ids': self.selected_line_ids(),
            'steps': deepcopy(self._steps),
            'result_name': self._result_name_edit.text().strip(),
            'input_artifact_ids': {},
            'save_intermediates': self._save_intermediates_check.isChecked(),
        }

    def _accept_if_valid(self):
        if self.selected_line_ids():
            self.accept()
        else:
            self._run_btn.setToolTip('请至少选择一条目标测线')
