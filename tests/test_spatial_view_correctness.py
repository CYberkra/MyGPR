# -*- coding: utf-8 -*-
"""空间页四个主视图的数据正确性回归测试（2026-10-07 审查 P0 批次）。

这些不是"补测试"，是**红线**：每一条都对应一个已用探针实测复现的缺陷，
修复前必失败、修复后必过。改动前先跑一遍确认它们是红的。

P0-1勾选变化 → 深度场不失效 → 静默导出已取消勾选的测线
    `spatial_page._on_line_check_changed` 只调 `_refresh_views()`（只刷轨迹
    散点），不碰 `_matrix` / `_depth_payload_line_ids`，也不重发预览请求。
    实测：填payload ['L1','L2'] → 取消勾选 L2 → save 仍回发 ['L1','L2']。

P0-2 set_depth_grid 先落地后校验 → TypeError + 混合态
    `spatial_page.py:609-614` 先 set_grid 写入视图，`:615-616` 才
    float(payload.get(...))。缺键抛 TypeError，且异常发生在
    `clear_depth_grid` 之前 → 矩阵是本次的、滑条/save 是上次的。

P0-3 NaN 被 marching squares 当成"高于 level" → 无数据区画伪等值线
    pyqtgraph 用 `data < level` 判格点，NaN 与任何值比较恒为 False → NaN 被
    归入高值侧。**填充类方案全部失败**（实测填最近有效值仍有 98% 节点落在空洞
    内；填 +inf 与 NaN 判同侧；填常数凭空造线）——marching squares 只有两态，
    无法表达"无数据"这个第三态。唯一正解是逐格判断：四角含非有限值就不画。
    实现见 `ui/widgets/marching_squares.py`。

P0-4 瓦片切源串源
    `map_view.set_source` 只清 _images/_failed，不清 _inflight 也无代次
    token；去重 key 与落库 key 都只有 (z,x,y)。实测：当前源 gaode_vec，
    _images 里躺着 osm 的图，且被去重锁死不再重下。

P0-5 fit_to_tracks 混合 mapped/unmapped 坐标 → 视野错位 12526×
    `tracks_bbox_lonlat` 有 `if not info.get('mapped'): continue`，
    `fit_to_tracks` 做完全相同的遍历却没有。实测视野跨度 13,982,180 m
    而轨迹本身 1,116 m。

P0-6 跨文件 .tmp 落盘竞态
    `map_view.py:198` 与 `trajectory_3d_view.py:224` 都用固定的
    `<path>.tmp`；gaode_img 源下两者路径逐字符相同，两个线程池并发写会
    损坏缓存 PNG。
"""
from __future__ import annotations

import os
import re
import time

import numpy as np
import pytest

pytest.importorskip('PyQt6')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def settle(qapp, wait_s: float = 0.35) -> None:
    end = time.perf_counter() + wait_s
    while time.perf_counter() < end:
        qapp.processEvents()


# ---------------------------------------------------------------- 夹具
class _Pt:
    def __init__(self, x, y, e=442.0):
        self.x, self.y, self.elevation_m = float(x), float(y), float(e)


class _Trk:
    """最小 SpatialTrack 鸭子类型。"""

    def __init__(self, line_id, xs, ys, crs='', n=50):
        self.line_id = line_id
        self.name = line_id
        self.rtk_status = 'fixed'
        self.coordinate_system = crs
        if not len(xs):
            self.points = []
        else:
            self.points = [_Pt(x, y) for x, y in zip(xs, ys)]


@pytest.fixture
def map_view(qapp):
    from ui.widgets.map_view import MapView
    view = MapView()
    view.resize(1200, 800)
    view.show()
    settle(qapp, 0.5)
    yield view
    view.deleteLater()
    qapp.processEvents()


@pytest.fixture
def spatial(qapp):
    from ui.pages.spatial_page import SpatialPage
    page = SpatialPage()
    page.resize(1600, 950)
    page.show()
    settle(qapp, 0.6)
    yield page
    page.deleteLater()
    qapp.processEvents()


@pytest.fixture(autouse=True)
def offline_tiles(monkeypatch):
    """把瓦片 worker 换成空实现——**单测绝不发真实网络请求**。

    ``TileLayer._enqueue`` 会 ``self._pool.start(_TileWorker(...))``，真跑一次
    ``urllib.request.urlopen``（timeout=10）。QObject 析构时 QThreadPool 要
    ``waitForDone()``，网络不通时这一等就是数十秒到挂死：pytest 收不到任何
    输出直接被 SIGTERM（实测：日志0 字节、连点行都没有）。

    autouse 而非按需：``MapView``/``SpatialPage`` 一旦 ``show()`` +进入事件
    循环就会自己触发瓦片请求，与测试是否显式调 ``_enqueue`` 无关——
    实测只给一条测试加禁网仍会卡在第 3 条（4 分钟无输出）。断言的是去重/
    落库/视野逻辑，与下载无关。
    """
    from PyQt6.QtCore import QRunnable
    from ui.widgets import map_view
    from ui.widgets import trajectory_3d_view

    class _NullWorker(QRunnable):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

        def run(self) -> None:
            pass

    monkeypatch.setattr(map_view, '_TileWorker', _NullWorker)
    # 三维地形 worker 同样要禁网：``SpatialPage`` 构造时地形来源默认 online，
    # 会在自己的 QThreadPool 里跑两个 ``urlopen(timeout=15)``
    # （terrarium 高程 + 高德影像）。只禁地图那一路时实测仍卡在第 3 条。
    monkeypatch.setattr(trajectory_3d_view, '_TerrainWorker', _NullWorker)


def _depth_payload(matrix=None, **over):
    payload = {
        'matrix': np.arange(100, dtype=float).reshape(10, 10)
        if matrix is None else matrix,
        'x_origin_m': 111.30,
        'y_origin_m': 30.12,
        'cell_size_m': 1.0,
        'attribute': '界面深度',
        'depth_min_m': 1.0,
        'depth_max_m': 5.0,
    }
    payload.update(over)
    return payload


# ================================================================ P0-5
class TestFitToTracksIgnoresUnmapped:
    """P0-5：未配准轨迹不得参与视野计算。"""

    def test_mixed_coords_view_zoomed_to_mapped_track(self, map_view):
        """一条已配准 + 一条 GK 投影米 → 视野必须贴合已配准那条。"""
        lon0, lat0 = 111.30, 30.12
        map_view.set_tracks([
            _Trk('L-mapped', np.linspace(lon0, lon0 + 0.01, 50),
                 np.linspace(lat0, lat0 + 0.008, 50), crs='EPSG:4326'),
            _Trk('L-unmapped', np.linspace(338_000, 338_500, 50),
                 np.linspace(3_415_000, 3_415_400, 50), crs=''),
        ])
        settle(qapp=None, wait_s=0.0)  # noqa: ARG005 - 占位，下面用 fixture 的
        info = {i['line_id']: i for i in map_view._track_summaries}
        assert info['L-mapped']['mapped'] is True
        assert info['L-unmapped']['mapped'] is False

        map_view.fit_to_tracks()
        vb = map_view._plot.getViewBox()
        x_lo, x_hi = vb.viewRange()[0]
        span = x_hi - x_lo
        true_span = float(np.ptp(info['L-mapped']['xs']))
        # 修复前：span ≈ 1.4e7 / true_span 1.116e3 ≈ 12526×
        assert span < true_span * 5, (
            f'视野跨度 {span:.0f} m 是已配准轨迹自身 {true_span:.0f} m 的 '
            f'{span / true_span:.0f} 倍——未配准坐标参与了视野计算')

    def test_all_unmapped_leaves_view_untouched(self, map_view):
        """全部未配准时不应乱设视野（宁可不动，也不要跳到原点）。"""
        map_view.set_tracks([
            _Trk('L1', np.linspace(338_000, 338_500, 50),
                 np.linspace(3_415_000, 3_415_400, 50), crs=''),
        ])
        vb = map_view._plot.getViewBox()
        before = list(vb.viewRange()[0])
        map_view.fit_to_tracks()
        after = list(vb.viewRange()[0])
        assert before == pytest.approx(after, abs=1.0)


# ================================================================ P0-4
class TestTileSourceSwitchNoCrossContamination:
    """P0-4：切源后旧源的回包不得写进新源的 _images。"""

    def test_inflight_cleared_on_source_switch(self, qapp):
        from ui.widgets.map_view import TILE_SOURCES, TileLayer
        layer = TileLayer()
        src_a, src_b = list(TILE_SOURCES)[:2]
        layer.set_source(src_b)
        layer.set_source(src_a)
        layer._images.clear()
        layer._failed.clear()
        layer._inflight.clear()
        layer._enqueue(14, 13000, 6800)
        assert (14, 13000, 6800) in layer._inflight

        layer.set_source(src_b)
        # 修复前：_inflight 仍含旧源 key
        assert (14, 13000, 6800) not in layer._inflight, (
            '切源未清 _inflight：新源请求同一瓦片会被去重挡掉，旧源回包'
            '将写进新源 _images')

    def test_stale_worker_result_dropped(self, qapp, tmp_path):
        """旧源 worker 完成时，当前源已变→ 必须丢弃而不是落库。"""
        from PyQt6.QtGui import QColor, QImage
        from ui.widgets.map_view import TILE_SOURCES, TileLayer
        layer = TileLayer()
        src_a, src_b = list(TILE_SOURCES)[:2]
        # 顺序要紧：先切到 src_a（旧源），再切到 src_b（当前源），
        # 随后才让 src_a 的 worker 回包 —— 这才是竞态的真实时序。
        layer.set_source(src_a)
        layer.set_source(src_b)
        assert layer.source_key() == src_b

        red = tmp_path / 'a.png'
        img = QImage(8, 8, QImage.Format.Format_ARGB32)
        img.fill(QColor(255, 0, 0))
        img.save(str(red))

        key = (14, 13000, 6800)
        layer._on_tile_finished(key[0], key[1], key[2], str(red),
                                source_key=src_a)
        # 当前源是 src_b，回包来自 src_a → 不得落库
        assert layer._images.get(key) is None, (
            '旧源瓦片被写进了新源的 _images')

    def test_current_source_worker_result_accepted(self, qapp, tmp_path):
        """同源回包必须正常落库（别把修复写成"一律丢弃"）。"""
        from PyQt6.QtGui import QColor, QImage
        from ui.widgets.map_view import TILE_SOURCES, TileLayer
        layer = TileLayer()
        src = list(TILE_SOURCES)[0]
        layer.set_source(src)
        blue = tmp_path / 'b.png'
        img = QImage(8, 8, QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 255))
        img.save(str(blue))

        key = (14, 13000, 6801)
        layer._on_tile_finished(key[0], key[1], key[2], str(blue),
                                source_key=src)
        assert layer._images.get(key) is not None, (
            '同源回包被误丢弃——底图会永远空白')


# ================================================================ P0-6
class TestTileTmpPathIsolated:
    """P0-6：临时文件名必须唯一，不能跨线程/跨文件撞车。"""

    def test_no_fixed_tmp_suffix(self):
        """两处落盘代码都不得使用固定的 `<path>.tmp`。"""
        offenders = []
        for rel in ('ui/widgets/map_view.py', 'ui/widgets/trajectory_3d_view.py'):
            path = os.path.join(_ROOT, rel.replace('/', os.sep))
            with open(path, encoding='utf-8') as fh:
                text = fh.read()
            for m in re.finditer(r"tmp_path\s*=\s*([^\n]+)", text):
                expr = m.group(1).strip()
                if 'unique_tmp_path' not in expr:
                    offenders.append(f'{rel}: tmp_path = {expr}')
        assert not offenders, (
            '落盘临时名必须走 unique_tmp_path（固定 <path>.tmp 会被并发写'
            f'互相截断，map 与 3D 在 gaode_img 源下路径逐字符相同）: '
            f'{offenders}')

    def test_tmp_name_is_unique_per_process_and_thread(self):
        """修复后的临时名必须含 pid + 线程 id，且同线程连调两次也不同。

        ⚠️ pid + 线程 id **不够**：线程池里一个 worker 串行处理多个瓦片，
        同线程两次调用会得到完全相同的名字 → 第二个截断第一个的 tmp。
        """
        import threading
        from ui.widgets.map_tiles import unique_tmp_path
        a = unique_tmp_path('/x/y/1.png')
        b = unique_tmp_path('/x/y/1.png')
        assert a != b, f'同线程两次调用得到相同临时名：{a}'
        assert '.png.' in a
        assert str(os.getpid()) in a
        assert str(threading.get_ident()) in a


# ================================================================ P0-3
class TestDepthSliceNaNHandling:
    """P0-3：NaN 空洞不得产生伪等值线。"""

    def test_isoline_avoids_nan_only_region(self, qapp):
        """等值线节点必须全部落在有数据格内。

        场景用「左半边连续有效 + 右半边全 NaN」：修复前NaN 被判成高于 level
        侧，横穿无数据区画出一条伪线（节点 100% 落在无数据格内）；修复后
        等值线在有效区内部成形，且不会越过数据边界。

        不用「零散孤立点」那种场——那种场四角全有效的格子一个都没有，
        **正确行为就是一条线都不画**，拿``elementCount() > 0`` 当断言等于
        把旧 bug 的行为写进契约（这坑踩过一次）。
        """
        from ui.widgets.depth_slice_view import DepthSliceView

        view = DepthSliceView()
        try:
            rng = np.random.default_rng(5)
            m = rng.uniform(0.5, 6.0, (40, 40))
            m[:, 20:] = np.nan          # 右半边无数据
            view.set_grid(m, x_origin_m=0.0, y_origin_m=0.0, cell_size_m=1.0)
            view.set_isoline(3.0)

            # 读**视图实际使用的**路径：``_rebuild_isoline`` 已用 NaN 感知的
            # marching squares 覆盖过 ``IsocurveItem`` 自己算的那份。
            # 直接调 ``iso.generatePath()`` 会拿到 pyqtgraph 的旧算法结果
            # （NaN 判为高于 level），测的就不是生产路径了。
            view._rebuild_isoline()
            path = view._isocurve.path
            assert path is not None and path.elementCount() > 0, (
                '有效区内应能画出等值线')
            data = view._matrix
            nrows, ncols = data.shape
            # 坐标口径（两处都容易搞反，踩过）：
            # ① ``_isocurve`` 是 ``axisOrder='row-major'``，``isoline_paths``
            #    同语义先转置再算，输出 **(x=列, y=行)**。
            # ② 坐标是**像素中心空间**，合法范围是 [0, n]（含边界），
            #    不是 [0, n-1]：`extendToEdge``clamp` 到 ``shape-2``，
            #    所以 y=40.0 表示"底边像素 39 的外沿"，**不是越界**。
            #    落在 n 那一侧的节点归到最后一格（``min(int(f), n-1)``）。
            for i in range(path.elementCount()):
                x = path.elementAt(i).x
                y = path.elementAt(i).y
                assert 0.0 <= x <= ncols and 0.0 <= y <= nrows, (
                    f'等值线节点 ({x:.1f}, {y:.1f}) 超出网格范围')
                col = min(int(np.floor(x)), ncols - 1)
                row = min(int(np.floor(y)), nrows - 1)
                assert np.isfinite(data[row, col]), (
                    f'等值线节点 ({x:.1f}, {y:.1f}) 落在无数据格'
                    f'（row={row}, col={col}）——NaN 被当成了高于 level 侧')
            # 数据边界必须被精确截断：NaN 从 col 20 开始，等值线不得越过
            for i in range(path.elementCount()):
                x = path.elementAt(i).x
                if int(np.floor(x)) < ncols:
                    assert int(np.floor(x)) < 20, (
                        f'等值线越过数据边界进入无数据区（col={x:.1f}）')

        finally:
            view.deleteLater()
            qapp.processEvents()

    def test_isoline_parity_with_pyqtgraph_on_finite_data(self, qapp):
        """全有限矩阵 → 节点集合必须与 pyqtgraph ``fn.isocurve`` 逐点一致。

        换算法不得改变**原本正确**的行为。只比节点集合、不比折线分组：链式
        连接的分组方式（哪条线从哪个点起走）允许不同，节点不能多/少/移位。
        """
        from pyqtgraph.functions import isocurve as pg_isocurve
        from ui.widgets.marching_squares import isoline_paths

        rng = np.random.default_rng(11)
        m = rng.uniform(0.0, 10.0, (30, 30))
        # 语义对齐：``IsocurveItem(axisOrder='row-major')`` 先 ``data.T`` 再算，
        # 所以传给 ``fn.isocurve`` 的必须是转置后的矩阵。
        mine = isoline_paths(m, 5.0, axis_order='row-major')
        theirs = pg_isocurve(m.T, 5.0, connected=True, extendToEdge=True)
        mine_pts = sorted((round(x, 6), round(y, 6))
                          for line in mine for x, y in line)
        theirs_pts = sorted((round(x, 6), round(y, 6))
                            for line in theirs for x, y in line)
        assert mine_pts == theirs_pts, (
            f'全有限矩阵下与 pyqtgraph 不一致：'
            f'本实现 {len(mine_pts)} 点 / pyqtgraph {len(theirs_pts)} 点')

    def test_internal_hole_draws_no_isoline(self, qapp):
        """内部空洞（四周有数据）不得凭空画出穿越空洞的等值线。"""
        from ui.widgets.depth_slice_view import DepthSliceView
        rng = np.random.default_rng(3)
        m = rng.uniform(0.0, 10.0, (40, 40))
        m[15:25, 15:25] = np.nan
        view = DepthSliceView()
        try:
            view.set_grid(m, x_origin_m=0.0, y_origin_m=0.0, cell_size_m=1.0)
            view.set_isoline(5.0)
            view._rebuild_isoline()
            path = view._isocurve.path
            assert path is not None
            nrows, ncols = m.shape
            for i in range(path.elementCount()):
                x = path.elementAt(i).x
                y = path.elementAt(i).y
                assert 0.0 <= x <= ncols and 0.0 <= y <= nrows, (
                    f'等值线节点 ({x:.1f}, {y:.1f}) 超出网格范围')
                # 坐标是像素中心空间，边界节点归到最后一格（见上一条测试注释）
                col = min(int(np.floor(x)), ncols - 1)
                row = min(int(np.floor(y)), nrows - 1)
                assert np.isfinite(m[row, col]), (
                    f'等值线节点 ({x:.1f}, {y:.1f}) 落在内部空洞里')
        finally:
            view.deleteLater()
            qapp.processEvents()


# ================================================================ P0-2
class TestSetDepthGridAtomicity:
    """P0-2：校验必须前置，失败即整体回退。"""

    def test_missing_depth_bounds_raises_nothing_and_clears(self, spatial):
        """缺 depth_min/max → 不抛异常，且清空到底（不留半应用状态）。"""
        payload = _depth_payload()
        payload.pop('depth_min_m')
        payload.pop('depth_max_m')
        spatial.set_depth_grid(payload, ['L1'], 1.0)
        assert spatial._depth_view._matrix is None, '缺键时仍写入了矩阵'
        assert spatial._depth_payload_line_ids == []
        assert not spatial._depth_save_btn.isEnabled()
        assert (spatial._depth_slider.minimum(),
                spatial._depth_slider.maximum()) == (0, 0)

    def test_no_mixed_state_after_failed_call(self, spatial):
        """先成功一次、再失败一次 → 不得留下「本次矩阵 + 上次滑条」。"""
        spatial.set_depth_grid(_depth_payload(), ['L1'], 1.0)
        first_range = (spatial._depth_slider.minimum(),
                       spatial._depth_slider.maximum())
        bad = _depth_payload(matrix=np.arange(64, dtype=float).reshape(8, 8))
        bad.pop('depth_min_m')
        spatial.set_depth_grid(bad, ['L2'], 1.0)
        assert spatial._depth_view._matrix is None, (
            '失败调用后视图里还留着矩阵')
        assert (spatial._depth_slider.minimum(),
                spatial._depth_slider.maximum()) != first_range or True
        assert (spatial._depth_slider.minimum(),
                spatial._depth_slider.maximum()) == (0, 0), (
            f'滑条仍停在 {first_range}——那是上一次成功调用的残留')

    def test_missing_origin_not_silently_zeroed(self, spatial):
        """缺 x/y_origin_m → 不得静默把网格放到世界原点。"""
        payload = _depth_payload()
        payload.pop('x_origin_m')
        payload.pop('y_origin_m')
        spatial.set_depth_grid(payload, ['L1'], 1.0)
        extent = spatial._depth_view._grid_extent
        if extent is not None:
            # 真实轨迹在东经 111.30 /北纬 30.12（米制 ~1.1e7）
            assert not (abs(extent[0]) < 100 and abs(extent[1]) < 100), (
                f'缺坐标时网格静默落到世界原点：{extent}')

    def test_valid_payload_still_applies(self, spatial):
        """正常路径不能被防御性代码挡住。"""
        spatial.set_depth_grid(_depth_payload(), ['L1', 'L2'], 1.0)
        assert spatial._depth_view._matrix is not None
        assert spatial._depth_payload_line_ids == ['L1', 'L2']
        assert spatial._depth_save_btn.isEnabled()
        assert spatial._depth_slider.minimum() == 100
        assert spatial._depth_slider.maximum() == 500


# ================================================================ P0-1
class TestDepthFieldInvalidatesOnCheckChange:
    """P0-1：勾选集合变化必须让深度场失效。"""

    @staticmethod
    def _checked_ids(page):
        from PyQt6.QtCore import Qt
        lw = page._line_list
        return [str(lw.item(i).text()) for i in range(lw.count())
                if lw.item(i).checkState() == Qt.CheckState.Checked]

    def test_unchecking_line_clears_stale_depth_field(self, spatial):
        from PyQt6.QtCore import Qt
        spatial.set_tracks([
            _Trk('L1', np.linspace(111.30, 111.31, 50),
                 np.linspace(30.12, 30.13, 50), crs='EPSG:4326'),
            _Trk('L2', np.linspace(111.32, 111.33, 50),
                 np.linspace(30.14, 30.15, 50), crs='EPSG:4326'),
        ])
        spatial.set_lines([{'line_id': 'L1', 'name': 'L1', 'rtk_status': 'fixed'},
                           {'line_id': 'L2', 'name': 'L2', 'rtk_status': 'fixed'}])
        spatial.set_depth_grid(_depth_payload(), ['L1', 'L2'], 1.0)
        assert spatial._depth_payload_line_ids == ['L1', 'L2']

        lw = spatial._line_list
        target = next((lw.item(i) for i in range(lw.count())
                       if str(lw.item(i).text()) == 'L2'), None)
        assert target is not None, '列表里找不到 L2'
        target.setCheckState(Qt.CheckState.Unchecked)

        assert 'L2' not in self._checked_ids(spatial)
        assert 'L2' not in spatial._depth_payload_line_ids, (
            '勾选集合已变但深度场仍含 L2')
        assert not spatial._depth_save_btn.isEnabled(), (
            '深度场已失效但「存为图层」仍可点')

    def test_save_never_exports_unchecked_line(self, spatial):
        """端到端红线：保存回发的 line_ids 不得含未勾选测线。"""
        from PyQt6.QtCore import Qt
        spatial.set_tracks([
            _Trk('L1', np.linspace(111.30, 111.31, 50),
                 np.linspace(30.12, 30.13, 50), crs='EPSG:4326'),
            _Trk('L2', np.linspace(111.32, 111.33, 50),
                 np.linspace(30.14, 30.15, 50), crs='EPSG:4326'),
        ])
        spatial.set_lines([{'line_id': 'L1', 'name': 'L1', 'rtk_status': 'fixed'},
                           {'line_id': 'L2', 'name': 'L2', 'rtk_status': 'fixed'}])
        spatial.set_depth_grid(_depth_payload(), ['L1', 'L2'], 1.0)

        lw = spatial._line_list
        target = next((lw.item(i) for i in range(lw.count())
                       if str(lw.item(i).text()) == 'L2'), None)
        target.setCheckState(Qt.CheckState.Unchecked)

        emitted: list[list[str]] = []
        spatial.save_depth_layer_requested.connect(
            lambda ids, cell: emitted.append(list(ids)))
        spatial._on_save_depth_layer_clicked()
        for ids in emitted:
            assert 'L2' not in ids, (
                f'导出了已取消勾选的测线：{ids}')