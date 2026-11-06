"""Capture an actual Windows contrast theme on a disposable CI runner only."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys

import pytest


@pytest.mark.skipif(
    sys.platform != 'win32' or os.environ.get('GITHUB_ACTIONS') != 'true'
    or os.environ.get('MYGPR_HIGH_CONTRAST_CI') != '1',
    reason='Requires an explicitly designated disposable Windows CI runner',
)
def test_native_contrast_theme_captures_workbench_and_backup_dialog(tmp_path):
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from ui import constants
    from ui.main_window import MyGPRMainWindow
    from ui.project_archive_dialogs import BackupDialog
    from ui.settings_manager import SettingsManager
    from ui.theme_helpers import apply_theme

    class HighContrast(ctypes.Structure):
        _fields_ = [('cbSize', wintypes.UINT), ('dwFlags', wintypes.DWORD),
                    ('lpszDefaultScheme', ctypes.c_void_p)]

    user32 = ctypes.windll.user32
    user32.SystemParametersInfoW.argtypes = [wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT]
    user32.SystemParametersInfoW.restype = wintypes.BOOL
    free = ctypes.windll.kernel32.LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p

    def read_contrast():
        data = HighContrast(ctypes.sizeof(HighContrast), 0, None)
        if not user32.SystemParametersInfoW(0x0042, data.cbSize, ctypes.byref(data), 0):
            raise ctypes.WinError()
        try:
            scheme = ctypes.wstring_at(data.lpszDefaultScheme) if data.lpszDefaultScheme else None
            return int(data.dwFlags), scheme
        finally:
            if data.lpszDefaultScheme:
                free(data.lpszDefaultScheme)

    def write_contrast(flags, scheme):
        name = ctypes.create_unicode_buffer(scheme) if scheme else None
        pointer = ctypes.cast(name, ctypes.c_void_p) if name else None
        data = HighContrast(ctypes.sizeof(HighContrast), flags & ~0x1000, pointer)
        if not user32.SystemParametersInfoW(0x0043, data.cbSize, ctypes.byref(data), 0x0002):
            raise ctypes.WinError()

    original_flags, original_scheme = read_contrast()
    app = QApplication.instance() or QApplication([])
    apply_theme(constants.THEME_DARK)
    window = MyGPRMainWindow(settings=SettingsManager(str(tmp_path / 'settings.json')))
    dialog = None
    output = Path('output/ci-regressions')
    output.mkdir(parents=True, exist_ok=True)
    try:
        window.show()
        QTest.qWait(1000)
        window.splashScreen.close()
        dialog = BackupDialog(str(tmp_path), window)
        dialog.show()
        QTest.qWait(500)
        write_contrast(original_flags | 1, original_scheme)
        QTest.qWait(3000)
        assert read_contrast()[0] & 1, 'Windows did not activate its contrast theme'
        assert window.isVisible() and dialog.isVisible()
        screen = app.primaryScreen()
        assert screen is not None
        screenshot = screen.grabWindow(0)
        assert not screenshot.isNull()
        assert screenshot.save(str(output / 'windows-high-contrast-workbench.png'))
    finally:
        if dialog is not None:
            dialog.close()
        window.close()
        write_contrast(original_flags, original_scheme)
