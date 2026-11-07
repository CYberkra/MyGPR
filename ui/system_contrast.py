"""Keep native Qt control colors aligned with Windows contrast themes."""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from PyQt6.QtCore import QAbstractNativeEventFilter, QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


class _HighContrast(ctypes.Structure):
    _fields_ = [('cbSize', wintypes.UINT), ('dwFlags', wintypes.DWORD),
                ('lpszDefaultScheme', ctypes.c_void_p)]


def _system_colors():
    if sys.platform != 'win32':
        return None
    info = _HighContrast(ctypes.sizeof(_HighContrast), 0, None)
    user32 = ctypes.windll.user32
    if not user32.SystemParametersInfoW(0x0042, info.cbSize, ctypes.byref(info), 0):
        return None
    if not info.dwFlags & 1:
        return None
    get_color = user32.GetSysColor
    get_color.argtypes = [ctypes.c_int]
    get_color.restype = wintypes.DWORD
    return tuple(int(get_color(index)) for index in (5, 8, 15, 18, 13, 14, 17, 26))


def _qcolor(colorref: int) -> QColor:
    return QColor(colorref & 255, (colorref >> 8) & 255, (colorref >> 16) & 255)


class ContrastObserver(QAbstractNativeEventFilter):
    """Observe system color changes without consuming native messages."""

    def __init__(self, window):
        super().__init__()
        self._window = window
        self._app = QApplication.instance()
        self._saved_palette = None
        self._last_colors = None
        self._timer = QTimer(window)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.refresh)
        self._app.installNativeEventFilter(self)
        self.refresh()

    def nativeEventFilter(self, event_type, message):  # noqa: N802 - Qt override
        if bytes(event_type) in (b'windows_generic_MSG', b'windows_dispatcher_MSG'):
            native = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
            if native.message in (0x0015, 0x001A, 0x031A):  # SYSCOLORCHANGE, SETTINGCHANGE, THEMECHANGED
                self._timer.start(50)
        return False, 0

    def refresh(self):
        colors = _system_colors()
        if colors == self._last_colors:
            return
        if colors is None:
            if self._saved_palette is not None:
                self._set_palette(self._saved_palette)
                self._saved_palette = None
        else:
            if self._saved_palette is None:
                self._saved_palette = QPalette(self._app.palette())
            palette = QPalette(self._saved_palette)
            window, text, button, button_text, highlight, highlighted_text, disabled, link = map(_qcolor, colors)
            for role, color in ((QPalette.ColorRole.Window, window), (QPalette.ColorRole.Base, window),
                                (QPalette.ColorRole.AlternateBase, window), (QPalette.ColorRole.WindowText, text),
                                (QPalette.ColorRole.Text, text), (QPalette.ColorRole.Button, button),
                                (QPalette.ColorRole.ButtonText, button_text), (QPalette.ColorRole.Highlight, highlight),
                                (QPalette.ColorRole.HighlightedText, highlighted_text), (QPalette.ColorRole.Link, link)):
                palette.setColor(role, color)
            palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled)
            palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled)
            self._set_palette(palette)
        self._last_colors = colors

    def _set_palette(self, palette):
        from ui.theme_helpers import _refresh_native_view_palettes
        self._app.setPalette(palette)
        _refresh_native_view_palettes(self._app, palette)

    def theme_changed(self):
        if self._last_colors is not None:
            self._saved_palette = QPalette(self._app.palette())
            self._last_colors = None
            self.refresh()

    def shutdown(self):
        self._timer.stop()
        self._app.removeNativeEventFilter(self)


def install_contrast_observer(window):
    if sys.platform == 'win32' and QApplication.platformName() == 'windows':
        return ContrastObserver(window)
    return None
