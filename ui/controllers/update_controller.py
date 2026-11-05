"""Qt-thread update lifecycle and bounded native-to-GUI shutdown handshake."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


@dataclass(eq=False)
class _ShutdownRequest:
    generation: int = 0
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    approved: bool = False
    cancelled: bool = False


class UpdateController(QObject):
    status_changed = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)
    _native_event = pyqtSignal(int, str)
    _probe = pyqtSignal(object)
    _close = pyqtSignal(int)

    def __init__(self, *, prepare_close, freeze, close_window, parent=None):
        super().__init__(parent)
        self._prepare_close, self._freeze = prepare_close, freeze
        self._close_window = close_window
        self._client = None
        self._generation = 0
        self._closed = threading.Event()
        self._request_lock = threading.Lock()
        self._requests: set[_ShutdownRequest] = set()
        self._busy = False
        self._checking = False
        self._frozen = False
        self._native_event.connect(self._on_event)
        self._probe.connect(self._on_probe)
        self._close.connect(self._on_close)

    @property
    def busy(self):
        return self._busy

    def start(self, client, configuration, version, channel):
        if self._busy:
            raise RuntimeError('请先关闭更新窗口，再切换渠道')
        self.stop()
        self._closed.clear()
        self._generation += 1
        generation = self._generation
        self._client = client

        def event(value):
            if not self._closed.is_set():
                self._native_event.emit(generation, value)

        client.start(version=version, public_key=configuration.public_key,
                     feed=configuration.feed(channel), event=event,
                     can_shutdown=lambda: self.request_shutdown(generation=generation),
                     shutdown=lambda: self._close.emit(generation) if not self._closed.is_set() else None)
        self.status_changed.emit('')

    def check(self, *, silent=False):
        if self._client is None or self._busy or self._closed.is_set():
            return False
        self._set_busy(True)
        self._checking = True
        self.status_changed.emit('正在检查更新…')
        try:
            self._client.check(silent=silent)
        except (OSError, RuntimeError, ValueError):
            self._on_event(self._generation, 'error')
        return True

    def request_shutdown(self, timeout=5.0, *, generation=None):
        """Called only by a WinSparkle worker. Never inspect Qt objects here."""
        generation = self._generation if generation is None else generation
        request = _ShutdownRequest(generation=generation)
        with self._request_lock:
            if self._closed.is_set() or generation != self._generation:
                return 0
            self._requests.add(request)
        self._probe.emit(request)
        if not request.done.wait(timeout):
            with request.lock:
                request.cancelled = True
            if not self._closed.is_set():
                self._native_event.emit(generation, 'timeout')
        with self._request_lock:
            self._requests.discard(request)
        return int(request.approved and not request.cancelled and not self._closed.is_set())

    @pyqtSlot(object)
    def _on_probe(self, request):
        with request.lock:
            if request.cancelled or self._closed.is_set() or request.generation != self._generation:
                request.done.set()
                return
        # Saving may block on disk. The worker must still be able to cancel its
        # request while this work runs; a late GUI response must never freeze it.
        try:
            allowed, reason = self._prepare_close()
        except (OSError, RuntimeError, ValueError) as exc:
            allowed, reason = False, f'暂时无法安全退出：{exc}'
        with request.lock:
            if request.cancelled or self._closed.is_set() or request.generation != self._generation:
                request.done.set()
                return
            if allowed:
                self._freeze(True)
                self._frozen = True
                request.approved = True
            else:
                self.status_changed.emit(reason)
            request.done.set()

    @pyqtSlot(int)
    def _on_close(self, generation):
        if generation != self._generation or self._closed.is_set():
            return
        if not self._close_window():
            self._thaw()
            self.status_changed.emit('关闭已取消；请在安装器中取消安装')

    def _set_busy(self, value):
        self._busy = bool(value)
        self.busy_changed.emit(self._busy)

    def _thaw(self):
        if self._frozen:
            self._freeze(False)
            self._frozen = False

    @pyqtSlot(int, str)
    def _on_event(self, generation, event):
        if generation != self._generation or self._closed.is_set():
            return
        # Approval follows a completed download. A checker error is not a
        # launch-failure notification in 0.9.4; it must not unlock that window.
        if self._frozen and event == 'error':
            return
        if event in ('available', 'current', 'error'):
            self._checking = False
        elif event in ('cancelled', 'dismissed') and self._checking:
            self.status_changed.emit('正在结束更新检查…')
            return
        messages = {'error': '更新检查或下载失败，请重试或打开发布页',
                    'available': '发现新版本', 'current': '已是当前渠道的最新版本',
                    'timeout': '暂时无法退出，请稍后重试'}
        if event != 'available':
            self._set_busy(False)
        if event in ('error', 'cancelled', 'timeout'):
            self._thaw()
        if event in messages:
            self.status_changed.emit(messages[event])

    def stop(self):
        # Wake callbacks BEFORE native UI cleanup. 0.9.4 does not join checkers.
        self._closed.set()
        self._checking = False
        with self._request_lock:
            requests = list(self._requests)
        for request in requests:
            with request.lock:
                request.cancelled = True
                request.done.set()
        if self._client is not None:
            self._client.cleanup()
            self._client = None
        self._set_busy(False)
        self._thaw()
