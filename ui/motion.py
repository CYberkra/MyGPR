# -*- coding: utf-8 -*-
"""ui.motion — 轻量动效工具（GSAP 纪律的 Qt 移植）。

三条纪律：
1. **动画闸**：尊重 Windows"在 Windows 中显示动画"辅助功能开关
   （SPI_GETCLIENTAREAANIMATION）。关闭时所有动效 duration=0、立即落位。
2. **只动值，不动布局**：进度条用 QVariantAnimation 插值 setValue（合成器
   缓存，不触发布局重排）；徽章颜色过渡只重写 QSS 字符串（低频操作）。
3. **短促**：默认 200ms OutCubic，任何动效不超过 400ms——工程工具，
   不要烟花感。

徽章颜色过渡说明：QSS 的 background-color 无法被 QVariantAnimation 直接
驱动，这里对 RGB 三元组插值后整体重写 setStyleSheet，与静态徽章同一
QSS 模板，不引入新样式源。
"""
from __future__ import annotations

import ctypes

from PyQt6.QtCore import QEasingCurve, QVariantAnimation
from PyQt6.QtWidgets import QProgressBar, QWidget

SPI_GETCLIENTAREAANIMATION = 0x1042
DEFAULT_DURATION_MS = 200
MAX_DURATION_MS = 400


def animations_enabled() -> bool:
    """读取 Windows 辅助功能动画开关；非 Windows 或读取失败时视为开启。"""
    try:
        value = ctypes.c_bool(True)
        ok = ctypes.windll.user32.SystemParametersInfoW(
            SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(value), 0)
        return value.value if ok else True
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
    if (not animations_enabled() or target == current
            or hi <= lo):
        bar.setValue(target)
        return
    anim = getattr(bar, '_motion_anim', None)
    if anim is not None and anim.state() == QVariantAnimation.State.Running:
        # overwrite="auto"：从当前渲染值续跑，而不是从旧起点跳变
        current = anim.currentValue()
        if isinstance(current, (int, float)):
            current = int(current)
        anim.stop()
    if anim is None:
        anim = QVariantAnimation(bar)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        # QVariantAnimation 对 float 起止值插值出 float，PyQt6 的
        # setValue(int) 收到 float 会抛 TypeError（实机 abort）→ 显式转 int。
        anim.valueChanged.connect(lambda value: bar.setValue(int(value)))
        bar._motion_anim = anim
    anim.setDuration(DEFAULT_DURATION_MS)
    anim.setStartValue(float(current))
    anim.setEndValue(float(target))
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


def animate_badge_color(badge, qss_template: str, start_hex: str,
                        end_hex: str) -> None:
    """状态徽章背景色 200ms 渐变（同一 QSS 模板重写，无新样式源）。

    动画闸关闭或起止同色时立即落位；徽章销毁由父子关系自动接管
    （QVariantAnimation 以 badge 为 parent），无泄漏。
    """
    start, end = hex_to_rgb(start_hex), hex_to_rgb(end_hex)
    if not animations_enabled() or start == end:
        badge.setStyleSheet(qss_template % end_hex)
        return
    old = getattr(badge, '_motion_color_anim', None)
    if old is not None:
        old.stop()
    anim = QVariantAnimation(badge)
    anim.setDuration(DEFAULT_DURATION_MS)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.valueChanged.connect(
        lambda t, b=badge, s=start, e=end, q=qss_template:
        b.setStyleSheet(q % _mix(s, e, float(t))))
    badge._motion_color_anim = anim
    anim.finished.connect(
        lambda b=badge, e=end_hex, q=qss_template:
        b.setStyleSheet(q % e))
    anim.start()
