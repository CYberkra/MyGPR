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


def main() -> int:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qfluentwidgets import Theme, setTheme

    from ui.main_window import MyGPRMainWindow

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    saved: list[Path] = []

    for theme_name, theme in (('light', Theme.LIGHT), ('dark', Theme.DARK)):
        setTheme(theme)
        window = MyGPRMainWindow()
        # offscreen 屏幕尺寸受限（800×800），显式放大到标准窗口尺寸
        window.resize(*WINDOW_SIZE)
        window.show()
        theme_dir = OUT_DIR / theme_name
        theme_dir.mkdir(parents=True, exist_ok=True)

        def capture() -> None:
            assert _wait_backend(window, app), 'backend not ready'
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
