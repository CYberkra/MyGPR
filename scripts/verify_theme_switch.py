# -*- coding: utf-8 -*-
"""离屏验证主题切换重绘（阶段 3 任务 3 专项）。

定量口径：整窗截图的平均亮度。浅→深后平均亮度必须显著下降
（原"两轮全窗遍历"的回归风险是深底上残留大面积浅色块）；
深→浅后必须回升。两次切换均不得抛异常。

共用底座见 ``scripts/_qtprobe.py``（settle / 截图 / 亮度口径）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _qtprobe import create_app, mean_brightness, settle  # noqa: E402

app = create_app()

from ui.main_window import MyGPRMainWindow  # noqa: E402


def _switch_and_measure(window, theme: str) -> float:
    window._on_theme_changed(theme)
    settle(app, wait_s=1.0)
    return mean_brightness(window)


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
