# -*- coding: utf-8 -*-
"""离屏验证主题切换重绘（阶段 3 任务 3 专项）。

定量口径：整窗截图的平均亮度。浅→深后平均亮度必须显著下降
（原"两轮全窗遍历"的回归风险是深底上残留大面积浅色块）；
深→浅后必须回升。两次切换均不得抛异常。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)

from ui.main_window import MyGPRMainWindow  # noqa: E402


def _mean_brightness(window) -> float:
    image = window.grab().toImage()
    width, height = image.width(), image.height()
    if width <= 0 or height <= 0:
        return -1.0
    total = 0
    for y in range(0, height, 7):       # 抽样即可，避免逐像素太慢
        for x in range(0, width, 7):
            c = image.pixelColor(x, y)
            total += (c.red() + c.green() + c.blue()) / 3.0
    samples = len(range(0, height, 7)) * len(range(0, width, 7))
    return total / samples


def _settle(window, wait_s: float = 1.0) -> None:
    """主题切换后让事件循环真实空转：update() 排队的重绘派发完毕后截图。"""
    import time
    deadline = time.time() + wait_s
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def _switch_and_measure(window, theme: str) -> float:
    window._on_theme_changed(theme)
    _settle(window)
    return _mean_brightness(window)


window = MyGPRMainWindow()
window.show()
app.processEvents()

light = _switch_and_measure(window, '浅色主题')
dark = _switch_and_measure(window, '深色主题')
light_again = _switch_and_measure(window, '浅色主题')

print(f'brightness: light={light:.1f} dark={dark:.1f} light_again={light_again:.1f}')
assert dark < light * 0.75, f'浅→深后亮度未显著下降（残留浅色?）: {light} -> {dark}'
assert abs(light_again - light) < light * 0.25, \
    f'深→浅后亮度未回升（残留深色?）: {dark} -> {light_again}'

window.close()
print('theme switch verification PASSED')
