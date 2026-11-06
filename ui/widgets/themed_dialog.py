"""Native dialog caption colors for the application's independent theme."""
import ctypes
import sys

from PyQt6.QtWidgets import QDialog
from qfluentwidgets import isDarkTheme, qconfig


def _apply_caption_theme(dialog):
    if sys.platform != 'win32':
        return
    try:
        from ctypes import wintypes

        class HighContrast(ctypes.Structure):
            _fields_ = [('cbSize', wintypes.UINT), ('dwFlags', wintypes.DWORD),
                        ('lpszDefaultScheme', wintypes.LPWSTR)]

        contrast = HighContrast()
        contrast.cbSize = ctypes.sizeof(contrast)
        user32 = ctypes.windll.user32
        # Do not replace the colors chosen by Windows accessibility themes.
        if not user32.SystemParametersInfoW(0x0042, contrast.cbSize, ctypes.byref(contrast), 0):
            return
        dark = isDarkTheme() and not (contrast.dwFlags & 1)
        caption = wintypes.DWORD(0x00211e1b if dark else 0xFFFFFFFF)
        text = wintypes.DWORD(0x00F5F5F5 if dark else 0xFFFFFFFF)
        dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute
        dwm.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        dwm.restype = ctypes.c_long
        hwnd = int(dialog.winId())
        for attribute, value in ((35, caption), (36, text)):
            dwm(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value))
    except (AttributeError, OSError, TypeError, ValueError):
        # Older Windows versions and other Qt platforms keep their native frame.
        return


class ThemedDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        qconfig.themeChangedFinished.connect(self._update_caption)

    def _update_caption(self):
        if self.isVisible():
            _apply_caption_theme(self)

    def showEvent(self, event):
        super().showEvent(event)
        _apply_caption_theme(self)
