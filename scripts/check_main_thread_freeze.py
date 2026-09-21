# -*- coding: utf-8 -*-
"""主线程冻结门禁：open_project 期间"最长连续不返回事件循环"的时长。

这才是问题 1 的**判据**。视觉闪动 = Windows 合成器拿不到新帧 =
主线程被独占超过一个合成周期。所以真正要压的是 **max freeze**，
而不是 open 全链墙钟（墙钟里大部分是 worker 线程 I/O，主线程是空闲的）。

门禁阈值：最长冻结不得超过 MAX_FREEZE_MS（默认 400 ms）。
用 5ms 周期 QTimer 测主线程的连续占用窗口；退出码非 0 即视为回归。

做法：起一个 QTimer 以 5 ms 周期打点，记录相邻两次回调之间的实际间隔。
间隔 >> 5 ms 的地方就是主线程被独占的窗口。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

_GAPS: list[tuple[float, float]] = []      # (t_start, gap_ms)
_T0 = time.perf_counter()
_LAST: list[float] = []
PERIOD_MS = 5


def now_ms() -> float:
    return (time.perf_counter() - _T0) * 1000


def drive(app, ms=0, pulses=1):
    for _ in range(max(1, pulses)):
        app.processEvents()
        if ms:
            app.thread().msleep(ms)
        app.processEvents()


def install_watchdog(app):
    timer = QTimer()
    timer.setTimerType(__import__('PyQt6.QtCore', fromlist=['Qt']).Qt.TimerType.PreciseTimer)
    timer.setInterval(PERIOD_MS)

    def tick():
        t = now_ms()
        if _LAST:
            gap = t - _LAST[0]
            _GAPS.append((_LAST[0], gap))
        _LAST[:] = [t]

    timer.timeout.connect(tick)
    timer.start()
    return timer


def report():
    """打印冻结分布，返回 (max_freeze_ms, over_threshold_ms, over_count)。"""
    if not _GAPS:
        print('无采样')
        return 0.0, 0.0, 0
    gaps = sorted(_GAPS, key=lambda g: -g[1])
    print()
    print('=' * 68)
    print(f'采样 {len(_GAPS)} 次（周期 {PERIOD_MS} ms）')
    print(f'{"最长冻结":<28}{gaps[0][1]:>9.1f} ms   起点 {gaps[0][0]:.0f} ms')
    for i, (t, g) in enumerate(gaps[1:6], start=2):
        print(f'{f"第 {i} 长":<28}{g:>9.1f} ms   起点 {t:.0f} ms')
    over = [g for _t, g in _GAPS if g > PERIOD_MS * 4]
    print('-' * 68)
    print(f'冻结 > {PERIOD_MS * 4} ms 的次数        : {len(over)}')
    print(f'冻结 > {PERIOD_MS * 4} ms 的总时长      : {sum(over):>9.1f} ms')
    print('=' * 68)
    return gaps[0][1], sum(over), len(over)


def main() -> int:
    app = QApplication(sys.argv)

    from ui.main_window import MyGPRMainWindow
    from ui.settings_manager import SettingsManager

    settings = SettingsManager()
    root = sys.argv[1] if len(sys.argv) > 1 else str(
        settings.get('project_root', '') or '')

    # 门禁阈值：修复后实测 85–104 ms；修复前 1665–2883 ms。
    # 400 ms 留足 4× 余量，避免 CI 抖动误报，又能抓住量级回归。
    max_freeze_ms = float(os.environ.get('MYGPR_MAX_FREEZE_MS', '400'))

    win = MyGPRMainWindow(settings=settings)
    win.show()
    drive(app, 30, 25)
    try:
        win.splashScreen.close()
    except Exception:
        pass
    drive(app, 30, 15)
    for _ in range(300):
        drive(app, 20, 1)
        if win._backend_ready:
            break

    if not root:
        print('未指定项目目录（且设置里无 project_root），门禁跳过')
        win.close()
        return 0

    print(f'{now_ms():.0f}ms 后端就绪，开始 open_project + 主线程冻结计时')
    _GAPS.clear()
    _LAST.clear()
    timer = install_watchdog(app)

    t0 = time.perf_counter()
    win.project_controller.open_project(root)
    idle = 0
    for _ in range(1500):
        app.processEvents()
        app.thread().msleep(PERIOD_MS)
        app.processEvents()
        idle = idle + 1 if len(_GAPS) and _GAPS[-1][1] <= PERIOD_MS * 4 else 0
        if idle > 150:
            break
    wall = (time.perf_counter() - t0) * 1000
    timer.stop()

    print(f'{now_ms():.0f}ms open 全链结束，墙钟 {wall:.0f} ms')
    worst, _over_total, _over_n = report()
    win.close()
    drive(app, 30, 8)

    print()
    if worst > max_freeze_ms:
        print(f'FAIL 最长主线程冻结 {worst:.1f} ms > 阈值 {max_freeze_ms:.0f} ms')
        print('     → 打开项目时主线程被长时间独占，Windows 合成器会拿不到')
        print('       新帧，表现为整窗"消失再出现"。检查是否有新的同步全量')
        print('       重绘 / 大点集渲染被挂进了数据扇出链。')
        return 1
    print(f'PASS 最长主线程冻结 {worst:.1f} ms ≤ 阈值 {max_freeze_ms:.0f} ms')
    return 0


if __name__ == '__main__':
    sys.exit(main())
