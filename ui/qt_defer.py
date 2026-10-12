# -*- coding: utf-8 -*-
"""owned 单发 QTimer 工厂——消灭 deleteLater 后的悬空回调。

裸 ``QTimer.singleShot(fn, delay)`` 的回调在宿主被 ``deleteLater`` 后仍会
被后续 ``processEvents`` 派发（Python 包装器已失效，C++ 侧悬空）——CI 全量
套件确定性段错误的根因之一（2026-10-12，chain_strip / main_window 踩坑）。
本工厂把定时器挂到宿主身上：宿主销毁，定时器随之销毁，回调自然失效。
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer


def defer_call(host: QObject, fn, delay_ms: int = 0) -> QTimer:
    """在 ``host`` 下建单发 QTimer，``delay_ms`` 后执行 ``fn``。

    返回定时器本身；宿主销毁时 Qt 父子机制保证回调不再派发。
    """
    timer = QTimer(host)
    timer.setSingleShot(True)
    timer.timeout.connect(fn)
    timer.start(delay_ms)
    return timer
