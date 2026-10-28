# -*- coding: utf-8 -*-
"""ui.motion — 轻量动效工具（GSAP 纪律的 Qt 移植）。

三条纪律：
1. **动画闸**：尊重 Windows"在 Windows 中显示动画"辅助功能开关
   （SPI_GETCLIENTAREAANIMATION）。关闭时本模块管理的动效立即落位。
2. **只动值，不动布局**：进度条仅由一套 QVariantAnimation 驱动绘制值，
   不叠加控件内置补间；徽章颜色过渡只更新自身样式。
3. **短促**：默认 200ms OutCubic，任何动效不超过 400ms——工程工具，
   不要烟花感。

徽章颜色过渡说明：QSS 的 background-color 无法被 QVariantAnimation 直接
驱动，这里对 RGB 三元组插值后整体重写 setStyleSheet，与静态徽章同一
QSS 模板，不引入新样式源。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

from PyQt6.QtCore import QEasingCurve, QVariantAnimation
from PyQt6.QtWidgets import QProgressBar, QWidget

SPI_GETCLIENTAREAANIMATION = 0x1042
DEFAULT_DURATION_MS = 200
MAX_DURATION_MS = 400


def animations_enabled() -> bool:
    """读取 Windows 辅助功能动画开关；非 Windows 或读取失败时视为开启。"""
    try:
        value = wintypes.BOOL(True)
        ok = ctypes.windll.user32.SystemParametersInfoW(
            SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(value), 0)
        return bool(value.value) if ok else True
    except (AttributeError, OSError):
        return True


def animate_progress(bar: QProgressBar, target: int, *,
                     parent: QWidget | None = None) -> None:
    """进度条值插值：200ms OutCubic 流向 target（GSAP overwrite 语义）。

    复用 bar 上的常驻动画（属性名 ``_motion_anim``），新值到来时 stop 旧的
    再 start，避免同属性动画叠加抖动；动画闸关闭或与当前值相同时直接
    setValue 落位。动画只驱动 value，不改变 range。
    """
    target = int(target)
    lo, hi = bar.minimum(), bar.maximum()
    target = max(lo, min(hi, target))
    current = bar.value()
    anim = getattr(bar, '_motion_anim', None)
    if anim is not None:
        anim.stop()
    # Fluent ProgressBar otherwise starts its own 150ms animation on every
    # setValue call from our tween. Keep one owner and resume the painted value.
    set_use_ani = getattr(bar, 'setUseAni', None)
    get_rendered = getattr(bar, 'getVal', None)
    if callable(set_use_ani) and callable(get_rendered):
        current = max(lo, min(hi, int(get_rendered())))
        inner = getattr(bar, 'ani', None)
        if inner is not None:
            inner.stop()
        set_use_ani(False)
    if (not animations_enabled() or target == current
            or hi <= lo):
        bar.setValue(target)
        set_rendered = getattr(bar, 'setVal', None)
        if callable(set_rendered):
            set_rendered(float(target))
        return
    if anim is None:
        anim = QVariantAnimation(bar)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        # Integer endpoints emit int values; a QObject-bound receiver gives Qt
        # a destruction context instead of a closure retaining its parent bar.
        anim.valueChanged.connect(bar.setValue)
        bar._motion_anim = anim
    anim.setDuration(DEFAULT_DURATION_MS)
    anim.setStartValue(int(current))
    anim.setEndValue(int(target))
    anim.start()


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """'#3b82f6' → (59, 130, 246)；非法输入回退中性灰。"""
    text = color.lstrip('#')
    if len(text) == 6:
        try:
            return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
        except ValueError:
            pass
    return 0x9c, 0xa3, 0xaf  # #9ca3af


def _mix(start: tuple[int, int, int], end: tuple[int, int, int],
         t: float) -> str:
    r = round(start[0] + (end[0] - start[0]) * t)
    g = round(start[1] + (end[1] - start[1]) * t)
    b = round(start[2] + (end[2] - start[2]) * t)
    return '#%02x%02x%02x' % (max(0, min(255, r)),
                              max(0, min(255, g)),
                              max(0, min(255, b)))


class _BadgeColorAnimation(QVariantAnimation):
    """Qt owns the callback context; no closure retains the parent badge."""

    def __init__(self, badge) -> None:
        super().__init__(badge)
        self.transition = None

    def updateCurrentValue(self, value) -> None:
        if self.transition is not None and isinstance(value, (int, float)):
            template, start, end = self.transition
            self.parent().setStyleSheet(template % _mix(start, end, float(value)))


def animate_badge_color(badge, qss_template: str, start_hex: str,
                        end_hex: str) -> None:
    """状态徽章背景色 200ms 渐变（同一 QSS 模板重写，无新样式源）。

    动画闸关闭或起止同色时立即落位；徽章销毁由父子关系自动接管
    （QVariantAnimation 以 badge 为 parent），无泄漏。
    """
    start, end = hex_to_rgb(start_hex), hex_to_rgb(end_hex)
    anim = getattr(badge, '_motion_color_anim', None)
    previous_color = None
    if anim is not None:
        if anim.state() == QVariantAnimation.State.Running:
            _qss, old_start, old_end = anim.transition
            previous_color = _mix(old_start, old_end, float(anim.currentValue()))
        anim.stop()
    if not animations_enabled() or start == end:
        badge.setStyleSheet(qss_template % end_hex)
        return
    if previous_color is not None:
        start = hex_to_rgb(previous_color)
    if anim is None:
        anim = _BadgeColorAnimation(badge)
        badge._motion_color_anim = anim
    anim.transition = (qss_template, start, end)
    anim.setDuration(DEFAULT_DURATION_MS)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.start()
