#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据加载进度条对话框"""

from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
)
from qfluentwidgets import PushButton


class DataLoaderThread(QThread):
    """数据加载线程"""

    progress_updated = pyqtSignal(int, str)  # (progress, message)
    loading_finished = pyqtSignal(object)  # data
    loading_failed = pyqtSignal(str)  # error message

    def __init__(self, loader_func, *args, **kwargs):
        super().__init__()
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs
        self._is_cancelled = False

    def run(self):
        """执行加载"""
        try:
            # 调用加载函数，传入进度回调
            def progress_callback(percent, message):
                if not self._is_cancelled:
                    self.progress_updated.emit(percent, message)

            result = self.loader_func(
                *self.args, progress_callback=progress_callback, **self.kwargs
            )

            if not self._is_cancelled:
                self.loading_finished.emit(result)

        except Exception as e:
            if not self._is_cancelled:
                self.loading_failed.emit(str(e))

    def cancel(self):
        """取消加载"""
        self._is_cancelled = True


class LoadingProgressDialog(QDialog):
    """数据加载进度条对话框"""

    def __init__(self, parent=None, title="加载数据"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )

        self.loader_thread = None
        self.setup_ui()

    def setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 状态标签
        self.status_label = QLabel("正在加载...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 详细信息标签
        self.detail_label = QLabel("")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setProperty("class", "textSecondary")
        layout.addWidget(self.detail_label)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = PushButton("取消")
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

    def start_loading(self, loader_func, *args, **kwargs):
        """开始加载"""
        self.loader_thread = DataLoaderThread(loader_func, *args, **kwargs)
        self.loader_thread.progress_updated.connect(self._on_progress_updated)
        self.loader_thread.loading_finished.connect(self._on_loading_finished)
        self.loader_thread.loading_failed.connect(self._on_loading_failed)
        self.loader_thread.start()

    def _on_progress_updated(self, percent, message):
        """更新进度"""
        self.progress_bar.setValue(percent)
        self.detail_label.setText(message)

    def _on_loading_finished(self, data):
        """加载完成"""
        self.progress_bar.setValue(100)
        self.status_label.setText("加载完成！")
        self.btn_cancel.setText("关闭")
        self.accept()

        # 通知父窗口
        if hasattr(self.parent(), "_on_data_loaded"):
            self.parent()._on_data_loaded(data)

    def _on_loading_failed(self, error_msg):
        """加载失败"""
        self.status_label.setText("加载失败")
        self.detail_label.setText(error_msg)
        self.btn_cancel.setText("关闭")

    def _on_cancel(self):
        """取消/关闭"""
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.cancel()
            self.loader_thread.wait(1000)
        self.reject()
