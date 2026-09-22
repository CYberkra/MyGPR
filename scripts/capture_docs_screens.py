#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户文档配图截图工具（任务 F 候选 4 / P2）。

离屏驱动主窗口，浅色/深色双主题各截 8 页导航 + 整体视图，
输出到 ``docs/user/images/``。仅文档构建用，不进常规测试。

用法：``QT_QPA_PLATFORM=offscreen python scripts/capture_docs_screens.py``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / 'docs' / 'user' / 'images'
WINDOW_SIZE = (1450, 850)   # 与 constants.WINDOW_WIDTH/HEIGHT 一致


def _wait_backend(window, app, timeout_ms: int = 8000) -> bool:
    """轮询等待后端就绪（复用 app_qt 的 smoke 就绪逻辑）。"""
    import time
    waited = 0
    while not window._backend_ready and waited < timeout_ms:
        app.processEvents()
        time.sleep(0.1)
        waited += 100
    return window._backend_ready


def _assert_theme_applied(window, theme_name: str) -> None:
    """截图前断言主题真的生效（深色截图曾被静默拍成浅色）。

    采样窗口客户区左上角区域（页面边距，必为窗口背景色）：浅色主题下
    R/G/B 三通道均 > 200，深色主题下均 < 120。这条断言把「界面上看不出
    主题没生效」的静默失效变成显式失败。
    """
    image = window.grab().toImage()
    width, height = image.width(), image.height()
    # 取页面中栏上方空白处（避开左侧文件树与顶部页签条）
    color = image.pixelColor(int(width * 0.45), int(height * 0.30))
    channels = (color.red(), color.green(), color.blue())
    if theme_name == 'dark':
        ok = all(c < 120 for c in channels)
        expect = '深色（三通道 < 120）'
    else:
        ok = all(c > 200 for c in channels)
        expect = '浅色（三通道 > 200）'
    assert ok, (f'主题未生效：{theme_name} 期望{expect}，'
                f'实测 RGB{channels}。请检查 apply_theme() 是否被调用，'
                f'以及 settings.json 的 theme 是否被回放覆盖。')


def main() -> int:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from ui import constants
    from ui.main_window import MyGPRMainWindow

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    saved: list[Path] = []

    for theme_name, theme in (('light', constants.THEME_LIGHT),
                              ('dark', constants.THEME_DARK)):
        window = MyGPRMainWindow()
        # offscreen 屏幕尺寸受限（800×800），显式放大到标准窗口尺寸
        window.resize(*WINDOW_SIZE)
        # 走完整主题链路（_on_theme_changed → apply_theme）。**不能**只调
        # qfluentwidgets.setTheme()：那不改 palette / pyqtgraph 背景 / 原生
        # 控件 QSS，且随后 _init_state 会按 settings.json 回放主题把这里
        # 覆盖掉——深色截图曾被静默拍成浅色（2026-09-22 审阅修复）。
        window._on_theme_changed(theme)
        window.show()
        theme_dir = OUT_DIR / theme_name
        theme_dir.mkdir(parents=True, exist_ok=True)

        def capture() -> None:
            assert _wait_backend(window, app), 'backend not ready'
            # 后端就绪后重申主题：_finish_warmup 会补建延后页并注入设置，
            # 期间可能触碰主题状态。
            window._on_theme_changed(theme)
            app.processEvents()
            _assert_theme_applied(window, theme_name)
            pages = list(window.pages.items())
            for object_name, page in pages:
                window.switchTo(page)
                app.processEvents()
                path = theme_dir / f'{theme_name}_{object_name}.png'
                window.grab().save(str(path))
                saved.append(path)
            window.switchTo(pages[0][1])
            app.processEvents()
            overall = theme_dir / f'{theme_name}_overall.png'
            window.grab().save(str(overall))
            saved.append(overall)
            window.close()

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(capture)
        timer.start(3000)
        app.exec()

    for path in saved:
        print(f'[docs-shots] {path.name}')
    print(f'[docs-shots] OK, {len(saved)} screenshots -> {OUT_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
