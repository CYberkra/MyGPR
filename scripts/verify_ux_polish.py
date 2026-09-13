# -*- coding: utf-8 -*-
"""阶段 4 UX 打磨离屏自证（参照 scripts/verify_theme_switch.py 写法）。

定量断言（非"看一眼"）：
1. 空态引导：spatial/delivery 测线列表为空时引导文案占位、有数据时列表
   复现；project 成果表空表给一行不可选占位文案；
2. 窄窗自动折叠：processing/spatial 页中栏低于阈值时两侧栏自动折叠，
   恢复宽度后仅自动展开"自动折叠过"的栏；用户手动折叠的栏保持不动；
3. 解释页点表：拾取为末行增量插入（O(1)），单点删除局部删行 + 行号重排；
4. B-scan：显示模式切换才重置视野，新数据保持用户缩放；十字读数文本
   30ms 节流（首次即时，连续移动合并刷新）；
5. 地图：同指纹轨迹转换走缓存（切底图往返不重复做 GCJ-02 逐点转换）；
6. LogPanel：日志块数上限 5000，超出丢弃最旧；
7. 深度切片：ColorBarItem 色标存在且 levels 与值域同步，主题切换不炸。

截图证据输出到 output/ux_polish_verify/（该目录已被 .gitignore 忽略）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)

SHOT_DIR = Path(__file__).resolve().parents[1] / 'output' / 'ux_polish_verify'
SHOT_DIR.mkdir(parents=True, exist_ok=True)


def _settle(wait_s: float = 0.3) -> None:
    """让事件循环真实空转（resize 布局、节流定时器派发）。"""
    deadline = time.time() + wait_s
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def _shot(widget, name: str) -> None:
    widget.grab().save(str(SHOT_DIR / f'{name}.png'), 'PNG')


def _track(line_id: str, n: int = 5):
    from types import SimpleNamespace
    return SimpleNamespace(
        line_id=line_id, name=line_id, source='verify',
        coordinate_system='EPSG:4326',
        points=tuple(SimpleNamespace(x=114.0 + i * 0.001, y=22.0 + i * 0.001,
                                     elevation_m=10.0 + i)
                     for i in range(n)),
    )


# ---------------------------------------------------------------- 1. 空态引导
def verify_empty_states() -> None:
    from ui.pages.delivery_page import DeliveryPage
    from ui.pages.project_page import ProjectPage
    from ui.pages.spatial_page import SpatialPage

    spatial = SpatialPage()
    spatial.set_auto_prefetch_enabled(False)   # 防后台瓦片下载干扰离屏验证
    spatial.resize(1100, 700)
    spatial.show()
    _settle()
    assert not spatial._line_list.isVisibleTo(spatial), '空态下测线列表应隐藏'
    assert spatial._lines_empty_hint.isVisibleTo(spatial), '空态下引导文案应显示'
    assert '导入' in spatial._lines_empty_hint.text()
    _shot(spatial, 'spatial_empty_lines')

    spatial.set_tracks([_track('L01'), _track('L02')])
    _settle()
    assert spatial._line_list.isVisibleTo(spatial), '有测线后列表应显示'
    assert not spatial._lines_empty_hint.isVisibleTo(spatial), '有测线后引导应隐藏'
    assert spatial._line_list.count() == 2
    spatial.close()

    delivery = DeliveryPage()
    delivery.resize(900, 700)
    delivery.show()
    _settle()
    assert not delivery._lines_list.isVisibleTo(delivery), '空态下测线列表应隐藏'
    assert delivery._lines_empty_hint.isVisibleTo(delivery), '空态下引导文案应显示'
    delivery.set_lines([_track('L01'), _track('L02'), _track('L03')])
    _settle()
    assert delivery._lines_list.isVisibleTo(delivery), '有测线后列表应显示'
    assert not delivery._lines_empty_hint.isVisibleTo(delivery)
    assert delivery._lines_list.count() == 3
    delivery.close()

    project = ProjectPage()
    project.resize(1200, 800)
    project.show()
    _settle()
    project.set_artifacts([])
    _settle()
    table = project._artifacts_table
    assert table.rowCount() == 1, '空成果表应给一行占位'
    placeholder = table.item(0, 0)
    assert '暂无成果' in placeholder.text()
    assert not (placeholder.flags() & Qt.ItemFlag.ItemIsEnabled), '占位行不可交互'
    assert table.columnSpan(0, 0) == 6, '占位行应横跨 6 列'
    _shot(project, 'project_empty_artifacts')

    from types import SimpleNamespace
    project.set_artifacts([
        SimpleNamespace(artifact_id='a1', name='成果A', method_name='dewow',
                        shape=(10, 20), created_at='2026-09-14',
                        sha256='abcdef0123', line_id='L01'),
    ])
    _settle()
    assert table.rowCount() == 1 and table.item(0, 0).text() == '成果A'
    project.set_artifacts([])
    _settle()
    assert table.rowCount() == 1 and '暂无成果' in table.item(0, 0).text(), \
        '清空后占位行应复现'
    project.close()
    print('empty states PASSED')


# ---------------------------------------------------------------- 2. 窄窗折叠
def verify_narrow_collapse() -> None:
    """自动折叠策略验证。

    独立窗口有内容最小宽度约束（ProcessingPage 最小 ~1166px，贴近真实
    阈值 360 的触发线 1090），无法靠 resize 在真实阈值下触发；这里临时把
    page_scaffold._MIDDLE_MIN_PX 提到 500 走完整真实链路
    （resizeEvent → _auto_collapse_side_panels → CollapsiblePanel），
    另用真实阈值断言默认宽度下不误折叠。
    """
    import ui.page_scaffold as scaffold
    from ui.pages.processing_page import ProcessingPage
    from ui.pages.spatial_page import SpatialPage

    real_threshold = scaffold._MIDDLE_MIN_PX
    assert real_threshold == 360

    # 真实阈值：1200px 宽（中栏 470 ≥ 360）不得自动折叠
    page0 = ProcessingPage()
    page0.resize(1200, 700)
    page0.show()
    _settle()
    assert not page0._left_panel.is_collapsed()
    assert not page0._right_panel.is_collapsed()
    page0.close()

    scaffold._MIDDLE_MIN_PX = 500
    try:
        page = ProcessingPage()
        page.resize(1300, 700)
        page.show()
        _settle()
        assert (not page._left_panel.is_collapsed()
                and not page._right_panel.is_collapsed())
        page.resize(1200, 700)          # 中栏 470 < 500 → 自动折叠左栏
        _settle(0.5)
        assert page._left_panel.is_collapsed(), '窄窗下左栏应自动折叠'
        assert not page._right_panel.is_collapsed(),             '折叠左栏后中栏已够用，右栏应保留（贪心单侧折叠）'
        assert page._auto_collapsed_sides == {'left'}
        _shot(page, 'processing_narrow_collapsed')

        page.resize(1300, 700)          # 恢复容纳宽度 → 自动展开
        _settle(0.5)
        assert not page._left_panel.is_collapsed(), '恢复宽度后左栏应自动展开'
        assert not page._right_panel.is_collapsed()
        assert page._auto_collapsed_sides == set(), '自动展开后痕迹应清空'
        page.close()

        # 手动折叠尊重：用户手动收左栏后，窄窗→恢复全程左栏保持折叠，
        # 右栏不受牵连
        page2 = ProcessingPage()
        page2.resize(1300, 700)
        page2.show()
        _settle()
        page2._left_panel.toggle()      # 模拟用户手动折叠
        _settle(0.5)
        assert page2._left_panel.is_collapsed()
        assert 'left' not in page2._auto_collapsed_sides,             '手动折叠不应留下自动痕迹'
        page2.resize(1200, 700)
        _settle(0.5)
        assert page2._left_panel.is_collapsed()
        assert not page2._right_panel.is_collapsed()
        page2.resize(1300, 700)
        _settle(0.5)
        assert page2._left_panel.is_collapsed(), '手动折叠的栏恢复宽度后仍应折叠'
        assert not page2._right_panel.is_collapsed()
        page2.close()

        # 窄窗下用户手动展开左栏 → 恢复宽度时尊重（room 足够不会被重折），
        # 右栏仍按自动痕迹展开
        page3 = ProcessingPage()
        page3.resize(1200, 700)
        page3.show()
        _settle(0.5)
        assert page3._auto_collapsed_sides == {'left'}
        page3._left_panel.toggle()      # 窄窗下用户手动展开左栏
        _settle(0.5)
        assert not page3._left_panel.is_collapsed()
        assert 'left' not in page3._auto_collapsed_sides
        page3.resize(1300, 700)
        _settle(0.5)
        assert not page3._left_panel.is_collapsed(), '用户手动展开的栏应保留'
        assert not page3._right_panel.is_collapsed(), '右栏未被自动折叠，应保持'
        page3.close()

        # spatial 同策略冒烟（同为 320/340 栏宽）
        spatial = SpatialPage()
        spatial.set_auto_prefetch_enabled(False)
        spatial.resize(1300, 700)
        spatial.show()
        _settle()
        spatial.resize(1200, 700)
        _settle(0.5)
        assert spatial._left_panel.is_collapsed()
        assert not spatial._right_panel.is_collapsed()
        _shot(spatial, 'spatial_narrow_collapsed')
        spatial.resize(1300, 700)
        _settle(0.5)
        assert not spatial._left_panel.is_collapsed()
        assert not spatial._right_panel.is_collapsed()
        spatial.close()
    finally:
        scaffold._MIDDLE_MIN_PX = real_threshold
    print('narrow collapse PASSED')


# ---------------------------------------------------------------- 3. 点表增量
def verify_points_table_incremental() -> None:
    from ui.pages.interpretation_page import InterpretationPage

    page = InterpretationPage()
    page.resize(1100, 700)
    page.show()
    _settle()
    page.set_session_active(True)
    emitted = []
    page.points_changed.connect(lambda pts: emitted.append(list(pts)))

    for i in range(5):
        page._on_point_picked(i * 3, i * 7)
    table = page._points_table
    assert table.rowCount() == 5, f'拾取 5 次应有 5 行, got {table.rowCount()}'
    assert page._points_count_label.text() == '5 个点'
    assert [table.item(r, 0).text() for r in range(5)] == \
        ['1', '2', '3', '4', '5'], '# 列应顺序编号'
    assert len(emitted) == 5, '每次拾取应发 points_changed'

    table.selectRow(1)
    page._on_remove_selected_point()
    assert table.rowCount() == 4, '删除单点应局部删一行'
    assert [table.item(r, 0).text() for r in range(4)] == \
        ['1', '2', '3', '4'], '删除后后续行号应重排'
    assert page._points_count_label.text() == '4 个点'
    assert page._points == [(0, 0), (6, 14), (9, 21), (12, 28)]

    page._on_clear_points()
    assert table.rowCount() == 0 and page._points_count_label.text() == '0 个点'
    page.close()
    print('points table incremental PASSED')


# ---------------------------------------------------------------- 4. B-scan
def verify_bscan() -> None:
    import numpy as np
    from ui.widgets.bscan_view import BScanDisplayMode, BScanView

    view = BScanView()
    view.resize(600, 400)
    view.show()
    _settle()
    view.set_matrix(np.random.randn(80, 60).astype(np.float32), -1.0, 1.0)
    view.set_display_mode(BScanDisplayMode.WIGGLE)
    _settle()
    # 用户放大到局部视野
    view._plot.vb.setRange(xRange=(10, 30), yRange=(20, 50), padding=0.0)
    _settle()
    before = view._plot.vb.viewRect()
    # 新数据到达（波形模式重渲染）不应重置视野
    view.set_matrix(np.random.randn(80, 60).astype(np.float32), -1.0, 1.0)
    _settle()
    after = view._plot.vb.viewRect()
    assert abs(after.left() - before.left()) < 1e-6 and \
        abs(after.right() - before.right()) < 1e-6, \
        f'新数据应保持用户缩放: {before} vs {after}'
    # 显示模式切换允许重置视野（不崩溃即可，autoRange 生效）
    view.set_display_mode(BScanDisplayMode.GRAYSCALE)
    view.set_display_mode(BScanDisplayMode.WIGGLE)
    _settle()

    # 十字读数节流：首次即时，连续移动合并到 30ms 定时器
    view2 = BScanView()
    view2.resize(600, 400)
    view2.show()
    _settle()
    view2.set_matrix(np.random.randn(80, 60).astype(np.float32), -1.0, 1.0)
    _settle()

    def _scene(x, y):
        return view2._plot.vb.mapViewToScene(QPointF(x + 0.5, y + 0.5))

    view2._on_mouse_moved(_scene(10, 20))
    assert '道 11' in view2._readout.text(), '首次移动读数应即时刷新'
    view2._on_mouse_moved(_scene(30, 40))
    assert '道 11' in view2._readout.text(), '30ms 内文本应合并未刷新'
    assert view2._pending_readout is not None, '节流窗口内应有挂起读数'
    _settle(0.08)
    assert '道 31' in view2._readout.text(), '定时器触发后文本应刷新'
    assert view2._pending_readout is None
    view.close()
    view2.close()
    print('bscan view PASSED')


# ---------------------------------------------------------------- 5. 地图缓存
def verify_map_cache() -> None:
    import ui.widgets.map_view as map_view_mod
    from ui.widgets.map_view import MapView

    view = MapView()
    view.resize(700, 500)
    view.show()
    _settle()

    calls = {'n': 0}
    original = map_view_mod._track_to_mercator_uncached

    def _counting(track, gcj02=False):
        calls['n'] += 1
        return original(track, gcj02=gcj02)

    map_view_mod._track_to_mercator_uncached = _counting
    try:
        tracks = [_track('L01'), _track('L02')]
        view.set_tracks(tracks, {'L01': '#ff0000', 'L02': '#00ff00'})
        _settle()
        first = calls['n']
        assert first == 2, f'首次渲染应做 2 次转换, got {first}'

        # 同数据重渲染（勾选变化触发）→ 缓存命中
        view.set_tracks(list(reversed(tracks)),
                        {'L01': '#ff0000', 'L02': '#00ff00'})
        _settle()
        assert calls['n'] == first, '同指纹重渲染应命中缓存'

        # 切底图（gcj02 标志变化）→ 重新转换；切回 → 缓存命中
        original_source = view.source_key()
        other = 'osm' if original_source != 'osm' else 'gaode_vec'
        view.set_source(other)
        _settle()
        assert calls['n'] == first + 2, '换底图应重转 2 条'
        view.set_source(original_source)
        _settle()
        assert calls['n'] == first + 2, '切回原底图应命中缓存'
    finally:
        map_view_mod._track_to_mercator_uncached = original
    view.close()
    print('map cache PASSED')


# ---------------------------------------------------------------- 6. 日志上界
def verify_log_cap() -> None:
    from ui.widgets.log_panel import LogPanel

    panel = LogPanel()
    panel.resize(400, 500)
    panel.show()
    _settle()
    limit = panel._log_edit.document().maximumBlockCount()
    assert limit == 5000, f'日志块数上限应为 5000, got {limit}'
    for i in range(5100):
        panel.append_log('INFO 验证日志行 %d' % i)
    _settle(0.5)
    blocks = panel._log_edit.document().blockCount()
    assert blocks <= 5000, f'超出上限后块数应被截断, got {blocks}'
    assert blocks > 4000, '截断不应误伤正常日志'
    panel.close()
    print('log cap PASSED')


# ---------------------------------------------------------------- 7. 深度切片
def verify_depth_slice() -> None:
    import numpy as np
    from ui.widgets.depth_slice_view import DepthSliceView

    view = DepthSliceView()
    view.resize(700, 500)
    view.show()
    _settle()
    matrix = np.arange(12, dtype=float).reshape(3, 4) + 5.0
    view.set_grid(matrix, x_origin_m=0.0, y_origin_m=10.0, cell_size_m=1.0,
                  attribute='界面深度切片')
    _settle()
    assert view._colorbar is not None, '应创建 ColorBarItem 色标'
    lo, hi = view._colorbar.levels()
    assert abs(lo - 5.0) < 1e-9 and abs(hi - 16.0) < 1e-9, \
        f'色标 levels 应与值域同步, got {lo}~{hi}'
    _shot(view, 'depth_slice_colorbar')
    view.apply_theme(True)
    view.apply_theme(False)
    _settle()
    view.clear_grid()
    _settle()
    view.close()
    print('depth slice colorbar PASSED')


if __name__ == '__main__':
    # 本机可能联网：瓦片下载线程在 close() 时会阻塞场景清理（QThreadPool
    # 等待在途任务，实测卡在 SSL 证书加载）。本验证与瓦片无关，全程禁用
    # 瓦片入队，保证离线确定性。
    import ui.widgets.map_view as _map_view_mod

    _tile_layer_cls = _map_view_mod.TileLayer
    _original_enqueue = _tile_layer_cls._enqueue

    def _no_enqueue(self, z, x, y):
        return None

    _tile_layer_cls._enqueue = _no_enqueue
    try:
        verify_empty_states()
        verify_narrow_collapse()
        verify_points_table_incremental()
        verify_bscan()
        verify_map_cache()
        verify_log_cap()
        verify_depth_slice()
    finally:
        _tile_layer_cls._enqueue = _original_enqueue
    print(f'ALL UX POLISH CHECKS PASSED (screenshots -> {SHOT_DIR})')
