# -*- coding: utf-8 -*-
"""owned 单发 QTimer 工厂——消灭 deleteLater 后的悬空回调。

裸 ``QTimer.singleShot(fn, delay)`` 的回调在宿主被 ``deleteLater`` 后仍会
被后续 ``processEvents`` 派发（Python 包装器已失效，C++ 侧悬空）——CI 全量
套件确定性段错误的根因之一（2026-10-12，chain_strip / main_window 踩坑）。
本工厂把定时器挂到宿主身上：宿主销毁，定时器随之销毁，回调自然失效。
"""
from __future__ import annotations

from functools import partial
from itertools import count
import weakref

from PyQt6 import sip
from PyQt6.QtCore import QObject, QTimer


_TOKENS = count()
_PENDING_CALLS: dict[int, dict] = {}


def _release_call(token: int, _object=None) -> None:
    _PENDING_CALLS.pop(token, None)


def _start_call(timer_ref, *args, **kwargs) -> None:
    timer = timer_ref()
    if timer is None:
        raise RuntimeError('Deferred timer is no longer available')
    state = timer._defer_state
    _PENDING_CALLS[state['token']] = state
    try:
        QTimer.start(timer, *args, **kwargs)
    except (TypeError, OverflowError, RuntimeError):
        _release_call(state['token'])
        raise


def _stop_call(timer_ref) -> None:
    timer = timer_ref()
    if timer is not None:
        QTimer.stop(timer)
        _release_call(timer._defer_state['token'])


def _invoke_call(token: int) -> None:
    state = _PENDING_CALLS.get(token)
    if state is None:
        return
    timer = state['timer']
    try:
        state['callback']()
    finally:
        # A callback may stop/restart its timer or destroy its host itself.
        if sip.isdeleted(timer) or not timer.isActive():
            _release_call(token)

def defer_call(host: QObject, fn, delay_ms: int = 0) -> QTimer:
    """在 ``host`` 下建单发 QTimer，``delay_ms`` 后执行 ``fn``。

    返回定时器本身；stop() 取消保活，start() 可重新排队。除了 Qt 父子
    生命周期，活动期间也保留 Python 回调，避免 GC 在 deleteLater 派发前
    清空闭包默认参数。宿主销毁后不再派发。
    """
    timer = QTimer(host)
    token = next(_TOKENS)
    timer.setSingleShot(True)
    timer._defer_state = {'timer': timer, 'callback': fn, 'token': token}
    # Plain QTimer avoids a Python QObject-subclass destruction cycle. These
    # wrappers retain only a weak timer reference, including during cancellation.
    timer.start = partial(_start_call, weakref.ref(timer))
    timer.stop = partial(_stop_call, weakref.ref(timer))
    timer.timeout.connect(partial(_invoke_call, token))
    timer.destroyed.connect(partial(_release_call, token))
    timer.start(delay_ms)
    return timer
