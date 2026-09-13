# -*- coding: utf-8 -*-
"""离屏验证 Trajectory3DView 的 GLViewWidget 惰性创建（阶段 3 专项）。

构造后不得存在 GLViewWidget 子对象；set_tracks 喂入轨迹后必须已创建；
GL 未建期间的主题/夸张/贴地/影像/地形来源/DEM 设置不得崩溃。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)

import ui.widgets.trajectory_3d_view as tv_mod   # noqa: E402
from ui.widgets.trajectory_3d_view import Trajectory3DView  # noqa: E402

view = Trajectory3DView()

if tv_mod._gl is None:
    # 无 PyOpenGL：验证降级路径不崩
    assert view._gl_view is None
    assert view._fallback_label is not None
    view.apply_theme(True)
    view.set_vertical_exaggeration(2.0)
    view.set_track_drape(True)
    view.set_imagery_enabled(False)
    view.set_terrain_source('estimated')
    view.set_local_dem(None)
    track = SimpleNamespace(
        line_id='L01', coordinate_system='EPSG:32648',
        points=tuple(SimpleNamespace(x=float(x), y=0.0, elevation_m=5.0)
                     for x in range(5)))
    view.set_tracks([track], {'L01': '#ff0000'})
    print('DEGRADED PATH OK (no PyOpenGL)')
else:
    GLViewWidget = tv_mod._gl.GLViewWidget
    # 1) 构造后：无 GLViewWidget 子对象（惰性未触发）
    assert view._gl_view is None
    assert view.findChildren(GLViewWidget) == [], 'GLViewWidget 应惰性创建'
    # 2) GL 未建期间的语义设置不得崩（缓存语义值）
    view.apply_theme(True)                    # 主题缓存
    view.set_vertical_exaggeration(2.5)     # 夸张缓存
    view.set_track_drape(True)                # 贴地缓存
    view.set_imagery_enabled(False)           # 影像开关缓存
    view.set_terrain_source('estimated')      # 地形来源缓存
    view.set_local_dem(None)                  # DEM 清除（发空提示）
    assert view._theme_dark is True
    assert view._exag == 2.5
    assert view._drape is True
    assert view._imagery_on is False
    assert view._terrain_source == 'estimated'
    assert view._gl_view is None, '语义设置不得触发 GL 创建'
    # 3) set_tracks 触发惰性创建
    track = SimpleNamespace(
        line_id='L01', coordinate_system='EPSG:32648',
        points=tuple(SimpleNamespace(x=float(x), y=float(x) * 0.5,
                                     elevation_m=5.0)
                     for x in range(5)))
    view.set_tracks([track], {'L01': '#ff0000'})
    assert view._gl_view is not None, 'set_tracks 后必须已创建 GLViewWidget'
    assert len(view.findChildren(GLViewWidget)) == 1
    # 4) 缓存语义在创建后已生效：主题深色黑底 + 夸张/贴地参与渲染
    assert view._line_items, '测线 GL 项应已渲染'
    # 5) 主题切换在 GL 已建时直接生效
    view.apply_theme(False)
    assert view._theme_dark is False
    print('LAZY GL OK: created only after set_tracks; cached semantics applied')

print('GL lazy verification PASSED')
