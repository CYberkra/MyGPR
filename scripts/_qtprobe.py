# -*- coding: utf-8 -*-
"""离屏 Qt 探针的共用底座：QApplication 生命周期 / settle / 截图 / 后端就绪等待。

抽出动机（来自 2026-10-05 代码审查 §5.3）：`verify_ux_polish.py` 与
`verify_theme_switch.py` 各自实现了一份 `_settle`（逐字相同），而
`capture_docs_screens.py` 与 `capture_docs_states.py` 各自实现了一份
`_wait_backend`，且**两份已经漂移**：

- timeout 默认值8000 vs 10000，无任何理由；
- ``capture_docs_states.py`` 仍用裸 ``setTheme(Theme.LIGHT)``，而
  ``capture_docs_screens.py`` 已按2026-09-22 的教训改走
  ``window._on_theme_changed()`` 全链路并加了「主题没生效就显式失败」
  的像素断言——**正是审查报告担心的「各自演进导致行为不一致」**。

「离屏截图深色被静默拍成浅色」这个坑已经踩过一次；把它留在两份会漂移
的副本里等于给同一个坑第二次机会。

本模块只放**确定可复用**的部分：QApplication 持有、事件循环空转、窗口
截图、后端就绪轮询、主题生效断言。业务步骤留在各自脚本里。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> None:
    """把仓库根加入 sys.path（脚本以``python scripts/xxx.py`` 直接跑时需要）。"""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def create_app() -> object:
    """建/取 QApplication 单例并强制离屏。

    必须先设``QT_QPA_PLATFORM`` 再建QApplication：环境变量在QApplication
    构造时读取，之后设置无效。返回 QApplication 实例（已有则复用）。

    字体（2026-10-09）：offscreen 插件用 FreeType 精简字体库，默认从
    ``QT_QPA_FONTDIR``（未设置时≈exe 目录）扫描字体——venv 里没有字体
    文件，中文全部渲染成豆腐块（□）。指向系统字体目录恢复中英文渲染
    （A/B 实证见 ``output/probes_archive/_font_test_*.png``）；目录不存在
    （非 Windows）时保持 Qt 默认行为。
    """
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    if not os.environ.get('QT_QPA_FONTDIR'):
        win_fonts = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'Fonts'
        if win_fonts.is_dir():
            os.environ['QT_QPA_FONTDIR'] = str(win_fonts)
    ensure_repo_on_path()
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def settle(app, wait_s: float = 0.3) -> None:
    """让事件循环真实空转（布局 resize、节流定时器派发完毕）。

    截图前必须调用：``update()`` 只是把重绘排队，``processEvents`` 才
    真正派发。三个verify/capture 脚本曾各自实现一份完全相同的副本。
    """
    deadline = time.time() + wait_s
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def wait_backend(window, app, timeout_ms: int = 10000) -> bool:
    """轮询等待后端就绪（复用 app_qt 的 smoke 就绪逻辑）。

    默认 10000ms：营山测线导入类页面首帧要建索引，离屏机器上8000ms
    曾偶发不够（``capture_docs_screens.py`` 保留 8000，
    ``capture_docs_states.py`` 用 10000 —— 无理由漂移，此处取宽的）。
    """
    waited = 0
    while not window._backend_ready and waited < timeout_ms:
        app.processEvents()
        time.sleep(0.1)
        waited += 100
    return window._backend_ready


def shot(widget, path: Path, app=None) -> Path:
    """截图到 ``path``；给了 app 就先settle 一帧再拍。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if app is not None:
        settle(app)
    widget.grab().save(str(path), 'PNG')
    return path


def assert_theme_applied(window, theme_name: str) -> None:
    """截图前断言主题真的生效（深色截图曾被静默拍成浅色）。

    采样窗口客户区中栏空白处：浅色主题下 R/G/B 三通道均 > 200，深色
    主题下均< 120。这条断言把「界面上看不出主题没生效」的静默失效
    变成显式失败。
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


def mean_brightness(window, step: int = 7) -> float:
    """整窗截图的平均亮度（抽样，主题切换验证的定量口径）。"""
    image = window.grab().toImage()
    width, height = image.width(), image.height()
    if width <= 0 or height <= 0:
        return -1.0
    total = 0
    for y in range(0, height, step):
        for x in range(0, width, step):
            c = image.pixelColor(x, y)
            total += (c.red() + c.green() + c.blue()) / 3.0
    samples = len(range(0, height, step)) * len(range(0, width, step))
    return total / samples