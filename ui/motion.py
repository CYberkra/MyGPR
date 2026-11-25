# -*- coding: utf-8 -*-
"""Shared motion preference and interruptible progress animation."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

from PyQt6.QtCore import QAbstractAnimation, QEasingCurve, QVariantAnimation
from PyQt6.QtWidgets import QApplication, QProgressBar, QWidget

from ui.design_tokens import duration

SPI_GETCLIENTAREAANIMATION = 0x1042
WM_SETTINGCHANGE = 0x001A
_REDUCED_MOTION_PROPERTY = 'mygprReducedMotion'


def animations_enabled() -> bool:
    """Use the app override first, then the Windows accessibility preference."""
    app = QApplication.instance()
    if app is not None and app.property(_REDUCED_MOTION_PROPERTY):
        return False
    try:
        value = wintypes.BOOL(True)
        ok = ctypes.windll.user32.SystemParametersInfoW(
            SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(value), 0)
        return bool(value.value) if ok else True
    except (AttributeError, OSError):
        return True


def set_reduced_motion(reduced: bool) -> None:
    app = QApplication.instance()
    if app is not None:
        app.setProperty(_REDUCED_MOTION_PROPERTY, reduced)


def is_system_setting_change(event_type, message) -> bool:
    return (event_type == b'windows_generic_MSG' and bool(message)
            and wintypes.MSG.from_address(int(message)).message == WM_SETTINGCHANGE)


def finish_animation(animation) -> None:
    """Synchronously apply the endpoint and deliver Qt's completion signal."""
    if animation is not None and animation.state() == QAbstractAnimation.State.Running:
        animation.setCurrentTime(animation.duration())


def settle_window_motion(window: QWidget) -> None:
    """Complete in-flight effects at their latest requested targets."""
    for widget in [window, *window.findChildren(QWidget)]:
        settle = getattr(widget, 'settle_motion', None)
        if callable(settle):
            settle()
        finish_animation(getattr(widget, 'backgroundColorAni', None))
        if isinstance(widget, QProgressBar):
            animation = getattr(widget, '_motion_anim', None)
            if animation is not None and animation.state() == QAbstractAnimation.State.Running:
                stop_progress(widget, animation.endValue())


def _progress_value(bar: QProgressBar, value: int) -> int:
    return max(bar.minimum(), min(bar.maximum(), int(value)))


def _set_progress_value(bar: QProgressBar, value: int) -> None:
    bar.setValue(value)
    set_rendered = getattr(bar, 'setVal', None)
    if callable(set_rendered):
        set_rendered(float(value))


def stop_progress(bar: QProgressBar, target: int | None = None) -> None:
    """Stop the current tween and snap to target, or freeze the painted value."""
    current = _interrupt_progress(bar)
    _set_progress_value(bar, current if target is None else _progress_value(bar, target))


def _interrupt_progress(bar: QProgressBar) -> int:
    animation = getattr(bar, '_motion_anim', None)
    if animation is not None:
        animation.stop()
    # Fluent ProgressBar otherwise starts its own tween on every setValue call.
    if hasattr(bar, 'setUseAni'):
        current = bar.getVal()
        bar.ani.stop()
        bar.setUseAni(False)
    else:
        current = bar.value()
    return _progress_value(bar, current)


def animate_progress(bar: QProgressBar, target: int) -> None:
    """Retarget the single progress tween from its currently painted value."""
    current = _interrupt_progress(bar)
    target = _progress_value(bar, target)
    if not animations_enabled() or target == current:
        _set_progress_value(bar, target)
        return
    anim = getattr(bar, '_motion_anim', None)
    if anim is None:
        anim = QVariantAnimation(bar)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        # Integer endpoints emit int values; a QObject-bound receiver gives Qt
        # a destruction context instead of a closure retaining its parent bar.
        anim.valueChanged.connect(bar.setValue)
        bar._motion_anim = anim
    anim.setDuration(duration('progress'))
    anim.setStartValue(int(current))
    anim.setEndValue(int(target))
    anim.start()
