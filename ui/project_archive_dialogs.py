"""Project backup and restore dialogs."""
from uuid import uuid4

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CheckBox, LineEdit, PrimaryPushButton, PushButton, SpinBox

from ui import file_dialogs
from ui.widgets.themed_dialog import ThemedDialog


class _ArchiveDialog(ThemedDialog):
    def __init__(self, title, button_text, directory, parent):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(510)
        self._directory = directory
        layout = QVBoxLayout(self)
        self._form = QFormLayout()
        layout.addLayout(self._form)
        self._extra = QVBoxLayout()
        layout.addLayout(self._extra)
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = PushButton('取消', self)
        cancel.clicked.connect(self.reject)
        self._submit = PrimaryPushButton(button_text, self)
        self._submit.setEnabled(False)
        self._submit.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(self._submit)
        layout.addLayout(buttons)

    def _path_field(self, label, browse):
        field = LineEdit(self)
        field.setReadOnly(True)
        button = PushButton('浏览…', self)
        button.clicked.connect(browse)
        row = QHBoxLayout()
        row.addWidget(field, 1)
        row.addWidget(button)
        self._form.addRow(label, row)
        return field

    def _browse_directory(self, field, title):
        path = file_dialogs.getExistingDirectory(self, title, field.text() or self._directory)
        if path:
            field.setText(path)


class BackupDialog(_ArchiveDialog):
    def __init__(self, directory, parent=None):
        super().__init__('备份当前项目', '开始备份', directory, parent)
        self._destination = self._path_field('备份目录', self._select_destination)
        self._destination.textChanged.connect(lambda text: self._submit.setEnabled(bool(text)))

        more = PushButton('更多选项', self)
        more.setCheckable(True)
        self._extra.addWidget(more, alignment=Qt.AlignmentFlag.AlignLeft)
        options = QWidget(self)
        form = QFormLayout(options)
        self._full = CheckBox('创建全量备份', options)
        form.addRow(self._full)
        self._prune = CheckBox('自动清理旧恢复点', options)
        self._keep = SpinBox(options)
        self._keep.setRange(1, 100)
        self._keep.setValue(10)
        self._keep.setEnabled(False)
        self._keep.setAccessibleName('保留最近的恢复点数')
        self._prune.toggled.connect(self._keep.setEnabled)
        self._prune.setToolTip('新备份通过验证后，保留指定数量的恢复点及所需基准包。')
        form.addRow(self._prune, self._keep)
        options.hide()
        more.toggled.connect(options.setVisible)
        self._extra.addWidget(options)

    def _select_destination(self):
        self._browse_directory(self._destination, '选择备份目录')

    def options(self):
        return {
            'destination_dir': self._destination.text(),
            'incremental': not self._full.isChecked(),
            'retention_keep': self._keep.value() if self._prune.isChecked() else None,
        }


class RestoreDialog(_ArchiveDialog):
    def __init__(self, directory, controller, parent=None):
        super().__init__('从备份恢复', '恢复为新项目', directory, parent)
        self._controller = controller
        self._token = ''
        self._chain_ok = False
        self._archive = self._path_field('备份包', self._select_archive)
        self._destination = self._path_field('恢复位置', self._select_destination)
        self._destination.setText(directory)
        self._destination.textChanged.connect(self._update_submit)
        self._name = LineEdit(self)
        self._name.setPlaceholderText('自动命名')
        self._form.addRow('新项目目录名', self._name)
        self._status = BodyLabel(self)
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        self._status.setWordWrap(True)
        self._status.hide()
        self._extra.addWidget(self._status)
        controller.archive_checked.connect(self._on_archive_checked)

    def _select_archive(self):
        path, _ = file_dialogs.getOpenFileName(
            self, '选择备份包', self._archive.text() or self._directory, 'MyGPR 备份 (*.zip);;所有文件 (*)')
        if path:
            self._archive.setText(path)
            self._token = uuid4().hex
            self._chain_ok = False
            self._update_submit()
            self._status.setText('正在检查备份链…')
            self._status.show()
            self._controller.check_backup_archive(self._token, path)

    def _on_archive_checked(self, token, error):
        if token == self._token:
            self._chain_ok = not error
            self._status.setText(error or '备份链完整')
            self._update_submit()

    def _select_destination(self):
        self._browse_directory(self._destination, '选择恢复位置')

    def _update_submit(self):
        self._submit.setEnabled(self._chain_ok and bool(self._destination.text()))

    def options(self):
        return {
            'archive_path': self._archive.text(),
            'destination_root': self._destination.text(),
            'project_dir_name': self._name.text().strip(),
        }
