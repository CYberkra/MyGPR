"""Keep the Inno application mutex alive until the frozen process exits."""
import ctypes
import sys

_HANDLE = None
MUTEX_NAME = r'Local\MyGPR.Application'


def hold_installer_mutex():
    global _HANDLE
    if sys.platform != 'win32' or not getattr(sys, 'frozen', False) or _HANDLE:
        return
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateMutexW
    create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    create.restype = ctypes.c_void_p
    _HANDLE = create(None, False, MUTEX_NAME)
    if not _HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
