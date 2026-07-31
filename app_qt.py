# -*- coding: utf-8 -*-
"""MyGPR Qt 前端入口（SPEC §0/§8）。

启动流程：QApplication → setTheme(Theme.LIGHT) → SplashScreen(600ms) → MyGPRMainWindow

``--smoke``：offscreen 验收模式——3 秒后将主窗口及各导航页截图保存到
``/tmp/mygpr_shots/``，随后以退出码 0 退出。
"""
import argparse
import os
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import Theme, setTheme

# 禁用 Qt6 的 Windows 系统深色模式自动接管：否则在系统深色模式下 Qt 会
# 反复用系统深色 palette 覆盖应用 palette，浅色主题下原生控件（表格/列表/
# 下拉框）仍是深色。禁用后由 theme_helpers.apply_theme 显式 setPalette，
# 深浅主题完全由应用内设置决定，跨机器表现一致。
os.environ.setdefault('QT_QPA_PLATFORM', 'windows:darkmode=0')

from ui import constants
from ui.main_window import MyGPRMainWindow

SMOKE_SHOTS_DIR = '/tmp/mygpr_shots'
SMOKE_DELAY_MS = 3000


def _run_smoke(window: MyGPRMainWindow) -> None:
    """截图全部导航页 + 整体窗口到 /tmp/mygpr_shots/，然后退出码 0。"""
    os.makedirs(SMOKE_SHOTS_DIR, exist_ok=True)
    app = QApplication.instance()
    saved = []

    pages = list(window.pages.items())
    for object_name, page in pages:
        window.switchTo(page)
        app.processEvents()
        shot_path = os.path.join(SMOKE_SHOTS_DIR, f'page_{object_name}.png')
        window.grab().save(shot_path)
        saved.append(shot_path)

    # 整体窗口（回到主页）
    if pages:
        window.switchTo(pages[0][1])
        app.processEvents()
    overall_path = os.path.join(SMOKE_SHOTS_DIR, 'window_overall.png')
    window.grab().save(overall_path)
    saved.append(overall_path)

    for path in saved:
        print(f'[smoke] saved: {path}')
    print(f'[smoke] OK, {len(saved)} screenshots -> {SMOKE_SHOTS_DIR}')
    app.exit(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=constants.APP_NAME)
    parser.add_argument('--smoke', action='store_true',
                        help='offscreen 验收：3s 后截图各页面到 /tmp/mygpr_shots/ 并退出')
    args, _ = parser.parse_known_args()

    # 高 DPI 自适应：125%/150% 等系统缩放下按真实比例换算逻辑像素，
    # 避免默认取整策略（150%→200%）把可用逻辑空间减半、按钮被挤出屏幕。
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName('MyGPR')
    # 强制 Fusion 样式：windowsvista 样式在 Win11 系统深色模式下会无视
    # QApplication.setPalette 自行绘制深色原生控件（表格/列表/下拉框），
    # 导致浅色应用主题里控件仍然深色。Fusion 完全遵循 palette，
    # 配合 theme_helpers.apply_theme 的显式 palette，深浅主题在所有
    # Windows 机器上表现一致。
    app.setStyle('fusion')
    app.setFont(QFont(constants.FONT_FAMILY, 10))
    setTheme(Theme.LIGHT)   # 默认浅色（跟随师兄），窗口内按设置回放

    window = MyGPRMainWindow()
    window.show()

    if args.smoke:
        QTimer.singleShot(SMOKE_DELAY_MS, lambda: _run_smoke(window))

    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
