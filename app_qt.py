# -*- coding: utf-8 -*-
"""MyGPR Qt 前端入口（SPEC §0/§8）。

启动流程：QApplication → setTheme(Theme.LIGHT) → SplashScreen(600ms) → MyGPRMainWindow

``--smoke``：offscreen 验收模式——3 秒后将主窗口及各导航页截图保存到
系统临时目录下的 ``mygpr_shots/``（可由 ``MYGPR_SMOKE_SHOTS_DIR`` 覆盖），
随后以退出码 0 退出。
"""
import argparse
import logging
import os
import platform
import sys
import tempfile
import time

# Python <3.11 没有 enum.StrEnum；给全局 enum 模块补一个最低限度兼容实现，
# 让 core / domain 各模块的 `from enum import StrEnum` 能直接工作。
try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import enum

    class StrEnum(str, enum.Enum):
        def __str__(self) -> str:
            return self.value

    enum.StrEnum = StrEnum

# 禁用 Qt6 的 Windows 系统深色模式自动接管：否则在系统深色模式下 Qt 会
# 反复用系统深色 palette 覆盖应用 palette，浅色主题下原生控件（表格/列表/
# 下拉框）仍是深色。禁用后由 theme_helpers.apply_theme 显式 setPalette，
# 深浅主题完全由应用内设置决定，跨机器表现一致。
if sys.platform == "win32":
    os.environ.setdefault('QT_QPA_PLATFORM', 'windows:darkmode=0')

from PyQt6.QtCore import QtMsgType, QTimer, qInstallMessageHandler
from PyQt6.QtWidgets import QApplication
from core.observability import (configure_structured_logging,
                                install_global_exception_hooks)
from ui import constants
from ui.main_window import MyGPRMainWindow, PlaceholderPage

# Keep smoke artifacts outside the installation directory and avoid the
# POSIX-only /tmp path when validating Windows packages.  CI can override it.
SMOKE_SHOTS_DIR = os.environ.get(
    'MYGPR_SMOKE_SHOTS_DIR', os.path.join(tempfile.gettempdir(), 'mygpr_shots'))
SMOKE_DELAY_MS = 3000
SMOKE_BACKEND_TIMEOUT_MS = 5000


def _setup_diagnostics() -> None:
    """接入 core.observability：崩溃捕获 + 结构化事件日志 + Qt 消息转发。

    产物（均写入 ``core.app_paths`` 决定的 MyGPR 日志目录）：
    - ``crash-*.json``      未捕获 Python 异常报告（含 traceback）；
    - ``native-crash.log``  faulthandler 捕获的原生崩溃堆栈（段错误等）；
    - ``mygpr-events.jsonl`` 结构化事件（启动/退出/Qt 警告/各模块日志）。

    日志里出现 ``app start`` 而无 ``app exit`` 即上次运行是崩溃退出。
    """
    install_global_exception_hooks(constants.LOG_DIR)
    configure_structured_logging(constants.LOG_DIR)

    # 各模块 logger（ui.widgets.* 等）默认 propagate 到 root——把 mygpr
    # 的文件 handler 挂到 root，让所有模块日志都落进同一个 jsonl；
    # mygpr 自身关掉 propagate 防双写。
    mygpr_logger = logging.getLogger('mygpr')
    root_logger = logging.getLogger()
    for handler in mygpr_logger.handlers:
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)
    mygpr_logger.propagate = False
    if root_logger.level in (logging.NOTSET,) or root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    def _qt_message_handler(mode, context, message) -> None:
        qt_logger = logging.getLogger('mygpr.qt')
        location = f'{context.file}:{context.line}' if context.file else ''
        if mode in (QtMsgType.QtDebugMsg, QtMsgType.QtInfoMsg):
            qt_logger.debug('qt %s %s', message, location)
        elif mode == QtMsgType.QtWarningMsg:
            qt_logger.warning('qt %s %s', message, location)
        else:
            qt_logger.error('qt %s %s', message, location)

    qInstallMessageHandler(_qt_message_handler)

    logging.getLogger('mygpr.app').info(
        'app start python=%s platform=%s pid=%s',
        sys.version.split()[0], platform.platform(), os.getpid())


def _run_smoke(window: MyGPRMainWindow) -> None:
    """截图全部导航页 + 整体窗口到 ``SMOKE_SHOTS_DIR``，并做结构性断言。

    断言内容：
    - 页面数量与命名符合预期；
    - 没有页面降级为 PlaceholderPage；
    - ProjectController / ProcessingController 已成功加载；
    - 后端在超时前进入 ready 状态。

    任一断言失败均以非 0 退出码终止，避免“页面建设中”也被判定为成功。
    """
    os.makedirs(SMOKE_SHOTS_DIR, exist_ok=True)
    app = QApplication.instance()
    saved = []
    errors = []

    expected_names = [
        'homeInterface', 'projectInterface', 'processingInterface',
        'interpretationInterface', 'spatialInterface', 'deliveryInterface',
        'jobsInterface', 'settingsInterface',
    ]

    # 1) 页面数量与命名
    actual_names = list(window.pages.keys())
    if actual_names != expected_names:
        errors.append(
            f'页面列表不匹配: expected={expected_names}, actual={actual_names}')

    # 2) 没有占位页
    placeholder_names = [
        name for name, page in window.pages.items()
        if isinstance(page, PlaceholderPage)
    ]
    if placeholder_names:
        errors.append(f'以下页面降级为占位页: {placeholder_names}')

    # 3) 核心 Controller 已加载
    if window.project_controller is None:
        errors.append('ProjectController 未加载')
    if window.processing_controller is None:
        errors.append('ProcessingController 未加载')

    # 4) 后端就绪（等待最多 SMOKE_BACKEND_TIMEOUT_MS）
    waited = 0
    step_ms = 100
    while not window._backend_ready and waited < SMOKE_BACKEND_TIMEOUT_MS:
        app.processEvents()
        # QThread.msleep 会阻塞事件循环，改用轮询 processEvents
        time.sleep(step_ms / 1000.0)
        waited += step_ms
    if not window._backend_ready:
        errors.append(
            f'后端未在 {SMOKE_BACKEND_TIMEOUT_MS}ms 内就绪')

    if errors:
        for err in errors:
            print(f'[smoke] FAIL: {err}', file=sys.stderr)
        window.close()
        app.exit(1)
        return

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
    window.close()
    app.exit(0)
    # Qt 6.11 offscreen 的进程拆除阶段存在偶发段错误（所有断言与截图
    # 完成之后才发生；本地 Windows 与 CI Linux 均可复现，非回归）。
    # 验收信号已全部产出，验收模式下直接硬退出绕开 teardown。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> int:
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont, QGuiApplication
        from PyQt6.QtWidgets import QApplication
        from qfluentwidgets import setTheme, Theme
    except ImportError as exc:
        print(
            "ERROR: MyGPR GUI dependencies are not installed.\n"
            "Install them with: pip install -e '.[gui]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    _setup_diagnostics()

    parser = argparse.ArgumentParser(description=constants.APP_NAME)
    parser.add_argument('--smoke', action='store_true',
                        help='offscreen 验收：3s 后截图各页面到临时目录（可由 MYGPR_SMOKE_SHOTS_DIR 覆盖）并退出')
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

    exit_code = app.exec()
    # 与 'app start' 配对：日志里有 start 无 exit 即上次运行崩溃退出
    logging.getLogger('mygpr.app').info('app exit code=%s', exit_code)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
