"""Pinned WinSparkle C ABI; native workers never receive Qt widgets."""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Callable
from mygpr.infrastructure.updates.configuration import windows_build_version


class WinSparkle:
    def __init__(self, dll_path: Path, *, library=None):
        # cdecl is required, including on 32-bit test hosts. Never search PATH.
        self._dll = library if library is not None else ctypes.CDLL(str(dll_path.resolve()))
        self._callbacks: list = []
        self._initialized = False

    def _call(self, name, argument_types=(), result_type=None):
        function = getattr(self._dll, 'win_sparkle_' + name)
        function.argtypes = list(argument_types)
        function.restype = result_type
        return function

    def _callback(self, name, callback, result_type=None):
        wrapped = ctypes.CFUNCTYPE(result_type)(callback)
        self._callbacks.append(wrapped)  # C holds function pointers, not Python references.
        self._call('set_' + name + '_callback', [type(wrapped)])(wrapped)

    def start(self, *, version: str, public_key: str, feed: str,
              event: Callable[[str], None], can_shutdown: Callable[[], int],
              shutdown: Callable[[], None]):
        self._call('set_app_details', [ctypes.c_wchar_p] * 3)('MyGPR Project', 'MyGPR', version)
        self._call('set_app_build_version', [ctypes.c_wchar_p])(windows_build_version(version))
        self._call('set_lang', [ctypes.c_char_p])(b'zh_CN')
        self._call('set_appcast_url', [ctypes.c_char_p])(feed.encode('utf-8'))
        accepted = self._call('set_eddsa_public_key', [ctypes.c_char_p], ctypes.c_int)(public_key.encode('ascii'))
        if accepted != 1:
            raise ValueError('更新签名公钥未被验证器接受')
        # Scheduling/consent belongs to MyGPR settings; avoid a second registry preference.
        self._call('set_automatic_check_for_updates', [ctypes.c_int])(0)
        for name, status in (('error', 'error'), ('did_find_update', 'available'),
                             ('did_not_find_update', 'current'), ('update_cancelled', 'cancelled'),
                             ('update_dismissed', 'dismissed')):
            self._callback(name, lambda status=status: event(status))
        self._callback('can_shutdown', can_shutdown, ctypes.c_int)
        self._callback('shutdown_request', shutdown)
        self._call('init')()
        self._initialized = True

    def check(self, *, silent: bool = False):
        if not self._initialized:
            raise RuntimeError('更新器尚未就绪')
        name = 'check_update_without_ui' if silent else 'check_update_with_ui'
        self._call(name)()

    def cleanup(self):
        if self._initialized:
            self._call('cleanup')()
            self._initialized = False
        # Keep callback references through native cleanup and until object destruction.
