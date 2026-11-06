"""Compact version and update preferences, using the existing themed controls."""
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import CheckBox, ComboBox, CaptionLabel, PushButton


class UpdatePanel(QWidget):
    check_requested = pyqtSignal()
    releases_requested = pyqtSignal()
    preferences_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.check_button = PushButton('检查更新', self)
        self.check_button.clicked.connect(self.check_requested)
        self.releases_button = PushButton('发布说明 / 离线下载', self)
        self.releases_button.clicked.connect(self.releases_requested)
        row.addWidget(self.check_button)
        row.addWidget(self.releases_button)
        row.addStretch(1)
        layout.addLayout(row)
        options = QHBoxLayout()
        self.automatic = CheckBox('自动检查更新', self)
        self.automatic.toggled.connect(self.preferences_changed)
        self.channel = ComboBox(self)
        self.channel.addItem('稳定版', userData='stable')
        self.channel.addItem('预览版', userData='preview')
        self.channel.currentIndexChanged.connect(self.preferences_changed)
        options.addWidget(self.automatic)
        options.addWidget(self.channel)
        options.addStretch(1)
        layout.addLayout(options)
        self.status = CaptionLabel('', self)
        self.status.setWordWrap(True)
        self.status.hide()
        layout.addWidget(self.status)

    def set_status(self, text):
        self.status.setText(text)
        self.status.setVisible(bool(text))

    def set_busy(self, busy):
        self.check_button.setEnabled(not busy)
        self.automatic.setEnabled(not busy)
        self.channel.setEnabled(not busy)

    def load(self, automatic, channel):
        for widget in (self.automatic, self.channel):
            widget.blockSignals(True)
        self.automatic.setChecked(bool(automatic))
        self.channel.setCurrentIndex(max(0, self.channel.findData(channel)))
        for widget in (self.automatic, self.channel):
            widget.blockSignals(False)
