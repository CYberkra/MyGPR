#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""带数据状态截图：建演示项目 + 导入营山测线 + 处理成果预览，再拍主页/项目页。

用途：文档配图需要"用起来的样子"而非空壳页（任务 F 候选 4 taste pass）。
用法：QT_QPA_PLATFORM=offscreen MYGPR_YINGSHAN_DATA=<数据目录> python scripts/capture_docs_states.py

离屏底座见 ``scripts/_qtprobe.py``。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _qtprobe import ROOT, create_app, wait_backend  # noqa: E402

OUT_DIR = ROOT / 'docs' / 'user' / 'images'
WINDOW_SIZE = (1450, 850)


def main() -> int:
    from ui import constants
    from ui.main_window import MyGPRMainWindow

    from PyQt6.QtCore import QTimer

    data_dir = os.environ.get('MYGPR_YINGSHAN_DATA')
    if not data_dir or not Path(data_dir).exists():
        print('[docs-states] MYGPR_YINGSHAN_DATA 未设置或不存在——保留空态截图')
        return 0
    source = sorted(Path(data_dir).glob('Line*origin(36).csv'))[0]

    app = create_app()
    window = MyGPRMainWindow()
    window.resize(*WINDOW_SIZE)
    # 与 capture_docs_screens.py 同走完整主题链路（_on_theme_changed →
    # apply_theme），不调裸 setTheme()：后者不改 palette / pyqtgraph 背景，
    # 且随后 _init_state 会按 settings.json 回放主题覆盖掉——这正是
    # 2026-09-22「深色截图被静默拍成浅色」的成因。
    window._on_theme_changed(constants.THEME_LIGHT)
    window.show()
    theme_dir = OUT_DIR / 'light'
    theme_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    state = {'phase': 0, 'project_root': str(Path(os.environ.get('TEMP', '/tmp')) / 'mygpr_docs_demo')}

    def _snapshot(name: str) -> None:
        app.processEvents()
        path = theme_dir / name
        window.grab().save(str(path))
        saved.append(path)

    def _step() -> None:
        phase = state['phase']
        state['phase'] += 1
        if phase == 0:
            assert wait_backend(window, app), 'backend not ready'
            # 建演示项目（走真实 controller，触发全部 UI 状态刷新）
            summary = window.project_controller._backend().projects.create_project(
                state['project_root'], name='yingshan-demo',
                coordinate_system='CGCS2000 / 3-degree GK zone 36')
            window.project_controller._current = summary
            window.project_controller.project_opened.emit(summary)
            window.project_controller.refresh_lines()
            QTimer.singleShot(1500, _step)
        elif phase == 1:
            # 导入营山测线（同步调用，导入后等 job 完成）
            backend = window.project_controller._backend()
            project_id = window.project_controller._project_id_or_warn()
            job_id = backend.submit_line_import(project_id, str(source), line_id='L01', name='Line3')
            backend.jobs.wait(job_id, timeout=120)
            # 触发测线列表刷新
            window.project_controller.refresh_lines()
            QTimer.singleShot(1200, _step)
        elif phase == 2:
            # 选中测线 → 主页预览 bundle
            _snapshot('light_projectInterface_data.png')
            window.switchTo(window.pages['homeInterface'])
            _snapshot('light_homeInterface_data.png')
            # 预览 bundle：主页 B-scan 展示
            backend = window.project_controller._backend()
            project_id = window.project_controller._project_id_or_warn()
            from ui.desktop_backend_facade import build_preview_bundle
            line_data = backend.projects.read_dataset(project_id, 'L01')
            bundle = build_preview_bundle('L01', line_data.data, title='L01 原始数据')
            window._page('homeInterface').set_preview_bundle(bundle)
            app.processEvents()
            _snapshot('light_homeInterface_preview.png')
            window.switchTo(window.pages['processingInterface'])
            _snapshot('light_processingInterface_data.png')
            print(f'[docs-states] OK, {len(saved)} 状态截图')
            window.close()
        QTimer.singleShot(50, _noop)

    def _noop() -> None:
        pass

    QTimer.singleShot(3000, _step)
    app.exec()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
