"""Wire the compact settings entry to the optional Windows updater."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

from PyQt6.QtCore import QObject, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication

from ui.controllers.update_controller import UpdateController
from ui.desktop_backend_facade import windows_update_client, windows_update_configuration
from ui.pages.settings_page import _read_version

RELEASES = 'https://github.com/CYberkra/MyGPR/releases'


class UpdateBinding(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.panel = window._page('settingsInterface').update_panel
        self._frozen_windows = []
        self._active_channel = None
        self._channel_pending = False
        self._unavailable_reason = ''
        self.controller = UpdateController(prepare_close=self.prepare_close,
            freeze=self.freeze, close_window=window.close, parent=self)
        self.controller.status_changed.connect(self.set_status)
        self.controller.busy_changed.connect(self.set_busy)
        self.panel.releases_requested.connect(lambda: QDesktopServices.openUrl(QUrl(RELEASES)))
        self.panel.check_requested.connect(self.check)
        self.panel.preferences_changed.connect(self.preferences_changed)
        self.panel.load(window.settings.get('update_automatic', True),
                        window.settings.get('update_channel', 'stable'))
        self._configuration = None
        self._root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.automatic_check)
        QApplication.instance().aboutToQuit.connect(self.controller.stop)
        window.destroyed.connect(self.controller.stop)
        self.configure()
        self._timer.start(3000)

    def configure(self):
        if self._active_channel is not None:
            raise RuntimeError('原生更新渠道需要重启后切换')
        self._unavailable_reason = ''
        self.controller.stop()
        if sys.platform != 'win32' or not getattr(sys, 'frozen', False):
            self._unavailable_reason = '开发环境：请从发布页获取安装包'
            self.set_status('')
            return
        try:
            self._configuration = windows_update_configuration(self._root / 'config/windows_updates.json')
            channel = self.window.settings.get('update_channel', 'stable')
            self.panel.channel.setItemEnabled(1, bool(self._configuration.preview_feed))
            if channel == 'preview' and not self._configuration.preview_feed:
                channel = 'stable'
                self.window.settings.set('update_channel', channel)
                self.panel.load(self.window.settings.get('update_automatic', True), channel)
                self.window.settings.save()
            self._configuration.feed(channel)
            client = windows_update_client(self._root / 'WinSparkle.dll')
            self.controller.start(client, self._configuration, _read_version(), channel)
            self._active_channel = channel
            self.set_busy(False)
        except (OSError, ValueError, RuntimeError, AttributeError) as exc:
            self._unavailable_reason = str(exc) or '更新器初始化失败，请从发布页下载安装包'
            self.controller.stop()
            self.set_status('')

    def preferences_changed(self):
        if self._active_channel is None:
            self.set_status('')
            return
        settings = self.window.settings
        settings.set('update_automatic', self.panel.automatic.isChecked())
        settings.set('update_channel', self.panel.channel.currentData())
        self._channel_pending = self._active_channel is not None and settings.get('update_channel') != self._active_channel
        self.set_busy(self.controller.busy)
        if not settings.save():
            self.panel.set_status('更新偏好未能保存')
            return
        # WinSparkle 0.9.4 UI cleanup leaves checkers alive. Never replace its
        # global feed/callback configuration while this process is running.
        self.panel.set_status('重启 MyGPR 后切换更新渠道' if self._channel_pending else '')

    def set_status(self, message):
        self.panel.set_status(self._unavailable_reason or
                             ('重启 MyGPR 后切换更新渠道' if self._channel_pending else message))

    def set_busy(self, busy):
        self.panel.set_busy(busy)
        if self._active_channel is None:
            self.panel.automatic.setEnabled(False)
            self.panel.channel.setEnabled(False)
        if self._channel_pending or self._active_channel is None:
            self.panel.check_button.setEnabled(False)

    def check(self):
        if not self._channel_pending and self.controller.check():
            self._record_check()

    def _record_check(self):
        self.window.settings.set('update_last_check', datetime.now(timezone.utc).isoformat(timespec='milliseconds'))
        self.window.settings.save()

    def automatic_check(self):
        if self._channel_pending or not self.window.settings.get('update_automatic', True):
            return
        last = self.window.settings.get('update_last_check', '')
        try:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
            if 0 <= elapsed < 86400:
                return
        except (ValueError, TypeError):
            pass
        if self.controller.check(silent=True):
            self._record_check()

    def prepare_close(self):
        window = self.window
        project = window.page_coordinator.project
        coordinator = window.page_coordinator
        if (project._busy_sources or project._active_job_ids
                or coordinator.processing.processing_job_id
                or coordinator.jobs.import_job_ids or coordinator.jobs.spatial_job_ids
                or not window._backend_ready or not window._pages_warmed):
            return False, '请等待当前任务结束后再安装更新'
        interpretation = window.interpretation_controller
        if interpretation is not None and interpretation.has_unsaved_changes():
            return False, '请先保存或明确放弃当前标注，再安装更新'
        if not window.settings.save():
            return False, '设置未能保存，已取消安装'
        return True, ''

    def freeze(self, frozen):
        if frozen:
            self._frozen_windows = [widget for widget in QApplication.topLevelWidgets()
                                   if (widget is self.window or self.window.isAncestorOf(widget))
                                   and widget.isEnabled()]
            for widget in self._frozen_windows:
                widget.setEnabled(False)
        else:
            for widget in self._frozen_windows:
                try:
                    widget.setEnabled(True)
                except RuntimeError:  # A floating window may have been destroyed during shutdown.
                    continue
            self._frozen_windows = []
