# -*- coding: utf-8 -*-
"""空间页「可见视图才重绘」回归测试。

回归背景（问题 1「打开项目后整窗闪一下」的真因）：
    ``SpatialPage._refresh_views`` 原实现无条件同步重绘四个 pyqtgraph /
    QOpenGL 画布。实测 6 条测线 / 10,394 个轨迹点的重绘成本：

        depth_view.set_tracks    ~640 ms（逐点 dict/颜色接口，见下）
        map_view.set_tracks      ~380 ms
        3d_view.set_tracks       ~130–470 ms
        profile_view.set_tracks  ~15 ms
        ─────────────────────────────────
        每次 _refresh_views 合计 ~0.9–1.3 s

    而「打开项目」会触发 ``set_tracks`` 两次（``on_project_opened`` 与
    ``on_lines_updated`` 各自 ``load_spatial_tracks()``），合计把主线程
    独占 **2 s 以上**。用户此时在主页，空间页根本不可见——为不可见的画布
    付 2 秒主线程代价，Windows 合成器长时间拿不到新帧，视觉上就是整窗
    「消失再出现」（无明显 Hide/Show 事件，offscreen 也不复现，与实测吻合）。

修复分两层：
    ① ``_refresh_views`` 只同步重绘「当前分段且页面可见」的视图，其余记入
       ``_dirty_views``，等 ``_switch_view`` / ``showEvent`` 时补齐；
    ② ``depth_slice_view`` 的散点改为「按颜色分组的 item 池」——pyqtgraph
       的 ScatterPlotItem 收逐点颜色会逐点建 QBrush 变体（10,394 点 426ms），
       只带单个 QBrush 时 35ms（12×）。

本测试钉住三个契约：
    - 页面不可见时 ``_refresh_views`` **不**重绘任何视图，只记脏；
    - 切到某分段时该分段被补齐（且只补一次，补后从脏集合移除）——
      2026-10-08 起 flush 延后一拍（QTimer.singleShot(0)），断言前需
      pump 事件循环；
    - 数据指纹没变时切段/刷新**不**重复 ``set_tracks``（三维视图切回
      不再全量重建——「每次点三维视图整窗消失数秒」的回归钉）。
"""
import pytest

pytest.importorskip('PyQt6')


@pytest.fixture
def spatial(qapp):
    """构造空间页（只需要页面自身，不启后端、不建主窗口）。"""
    from ui.pages.spatial_page import SpatialPage
    page = SpatialPage()
    yield page
    page.deleteLater()
    qapp.processEvents()


class _Track:
    """最小 SpatialTrack 鸭子类型。"""

    def __init__(self, line_id, n=50):
        self.line_id = line_id
        self.name = line_id
        self.points = [_Point(i, i * 0.5) for i in range(n)]


class _Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.elevation_m = 0.0


class TestDeferredRedraw:
    def test_hidden_page_defers_all_views(self, spatial):
        """页面不可见 → 四个视图全部只记脏，不发生实际 set_tracks。"""
        calls = []
        for key, view in spatial._views().items():
            if view is None:
                continue
            view.set_tracks = (
                lambda tracks, colors, _k=key: calls.append(_k))

        spatial._refresh_views()

        assert calls == [], f'页面不可见时不应重绘，实际重绘了 {calls}'
        assert spatial._dirty_views == set(spatial._views()), (
            '四个视图都应标记为脏')

    def test_switch_view_flushes_only_that_segment(self, spatial, qapp):
        """切到某分段 → 只补齐该分段，且补后从脏集合移除。

        flush 延后一拍（singleShot(0)）：先 pump 事件循环再断言。
        """
        from ui.pages import spatial_page as SP

        calls = []
        for key, view in spatial._views().items():
            if view is None:
                continue
            view.set_tracks = (
                lambda tracks, colors, _k=key: calls.append(_k))

        spatial._refresh_views()
        assert spatial._dirty_views == set(spatial._views())

        spatial._switch_view(SP._SEG_PROFILE)
        qapp.processEvents()  # 触发延后一拍的 flush
        assert calls == [SP._SEG_PROFILE], (
            f'切到剖面应只补剖面，实际 {calls}')
        assert SP._SEG_PROFILE not in spatial._dirty_views
        assert len(spatial._dirty_views) == 3

        # 再切一次同一分段不应重复重绘（已不在脏集合）
        calls.clear()
        spatial._switch_view(SP._SEG_PROFILE)
        qapp.processEvents()
        assert calls == [], '已补齐的分段不应重复重绘'

    def test_flush_is_idempotent(self, spatial):
        """显式 flush 已干净的分段是 no-op。"""
        calls = []
        for key, view in spatial._views().items():
            if view is None:
                continue
            view.set_tracks = (
                lambda tracks, colors, _k=key: calls.append(_k))

        spatial._dirty_views.clear()
        spatial._flush_dirty_view('planMap')
        assert calls == []

    def test_view_data_snapshot_kept_for_late_flush(self, spatial):
        """延后补齐用的是最近一次数据快照（切页后仍有数据可用）。

        ``_refresh_views`` 的快照来源是 ``_checked_tracks()``（勾选集合），
        所以这里直接打桩 ``_checked_tracks`` 提供数据。
        """
        from ui.pages import spatial_page as SP

        tracks = [_Track('L01'), _Track('L02')]
        colors = {'L01': '#ff0000', 'L02': '#00ff00'}
        spatial._checked_tracks = lambda: tracks
        # _refresh_views 的快照是 (tracks, self._colors)，
        # 所以颜色表必须落到实例上，不能只做局部变量。
        spatial._colors = colors

        applied = {}

        def fake_apply(view, t, c):
            applied['tracks'] = t
            applied['colors'] = c

        spatial._apply_tracks = fake_apply
        spatial._refresh_views()

        # 刷新时把数据写进快照（供后续 flush 使用）
        snapped_tracks, snapped_colors = spatial._view_data
        assert snapped_colors == colors
        assert [t.line_id for t in snapped_tracks] == ['L01', 'L02']

        # 手动 flush 一个分段，应拿到上面快照的数据
        spatial._dirty_views.add(SP._SEG_MAP)
        spatial._flush_dirty_view(SP._SEG_MAP)
        assert applied['colors'] == colors
        assert [t.line_id for t in applied['tracks']] == ['L01', 'L02']


class TestFingerprintDedup:
    """数据指纹去重：数据没变就不重复 set_tracks（2026-10-08）。

    回归背景（问题 2「每次点三维视图软件都消失几秒」）：脏标记是
    「切走就标脏、切回必全量 set_tracks」，即使数据一字未变。三维视图
    每次白付 GL 全量重建 + numpy 变换 + 4326→Mercator（130–470 ms），
    主线程阻塞期间 Windows 合成器拿不到新帧 → 整窗「消失再出现」。
    """

    def test_apply_tracks_if_changed_skips_unchanged_data(
            self, spatial, qapp):
        """同数据二次灌入被跳过；数据变化后重新灌入。"""
        from ui.pages import spatial_page as SP

        view = spatial._views()[SP._SEG_3D]
        calls = []
        view.set_tracks = lambda tracks, colors: calls.append(
            [t.line_id for t in tracks])

        tracks = [_Track('L01'), _Track('L02')]
        colors = {'L01': '#ff0000', 'L02': '#00ff00'}

        assert spatial._apply_tracks_if_changed(
            SP._SEG_3D, view, tracks, colors) is True
        assert len(calls) == 1

        # 同数据再灌：指纹一致，跳过
        assert spatial._apply_tracks_if_changed(
            SP._SEG_3D, view, tracks, colors) is False
        assert len(calls) == 1, '数据没变不应重复 set_tracks'

        # 数据变化（点数变）：重新灌入
        tracks.append(_Track('L03'))
        colors['L03'] = '#0000ff'
        assert spatial._apply_tracks_if_changed(
            SP._SEG_3D, view, tracks, colors) is True
        assert len(calls) == 2

    def test_refresh_skips_dirty_mark_when_fingerprint_unchanged(
            self, spatial):
        """视图已灌过同数据时，_refresh_views 不再把它标脏。"""
        from ui.pages import spatial_page as SP

        tracks = [_Track('L01')]
        spatial._checked_tracks = lambda: tracks
        spatial._colors = {'L01': '#ff0000'}

        # 手动给 3D 灌一次（成功后记指纹）
        view = spatial._views()[SP._SEG_3D]
        view.set_tracks = lambda tracks, colors: None
        assert spatial._apply_tracks_if_changed(
            SP._SEG_3D, view, tracks, spatial._colors) is True

        spatial._refresh_views()

        assert SP._SEG_3D not in spatial._dirty_views, (
            '指纹一致的视图不应被标脏')
        # 其余未灌过的视图仍标脏
        assert SP._SEG_MAP in spatial._dirty_views
        assert SP._SEG_PROFILE in spatial._dirty_views

    def test_switch_back_to_3d_without_data_change_skips_rebuild(
            self, spatial, qapp):
        """用户场景全链路：切三维(重建)→切走→刷新(同数据)→切回不重建。"""
        from ui.pages import spatial_page as SP

        calls = []
        for key, view in spatial._views().items():
            if view is None:
                continue
            view.set_tracks = (
                lambda tracks, colors, _k=key: calls.append(_k))

        spatial._refresh_views()                      # 全部标脏
        spatial._switch_view(SP._SEG_3D)              # 切到三维
        qapp.processEvents()
        assert calls == [SP._SEG_3D], '首次切三维应重建一次'

        spatial._switch_view(SP._SEG_PROFILE)         # 切走（刷剖面）
        qapp.processEvents()
        assert calls == [SP._SEG_3D, SP._SEG_PROFILE]

        spatial._refresh_views()                      # 数据没变的刷新
        spatial._switch_view(SP._SEG_3D)              # 再切回三维
        qapp.processEvents()
        assert calls == [SP._SEG_3D, SP._SEG_PROFILE], (
            '数据没变时切回三维不应重建 set_tracks')

    def test_switch_back_to_3d_after_data_change_rebuilds(
            self, spatial, qapp):
        """数据真的变了：切回三维必须重建（去重不得挡住真更新）。"""
        from ui.pages import spatial_page as SP

        tracks = [_Track('L01', 50), _Track('L02', 50)]
        spatial._checked_tracks = lambda: tracks

        calls = []
        for key, view in spatial._views().items():
            if view is None:
                continue
            view.set_tracks = (
                lambda t, c, _k=key: calls.append(_k))

        spatial._refresh_views()
        spatial._switch_view(SP._SEG_3D)
        qapp.processEvents()
        assert calls == [SP._SEG_3D]

        # 勾选集合变化 → 新指纹（点数不同）
        spatial._checked_tracks = lambda: [tracks[0]]
        spatial._refresh_views()
        spatial._switch_view(SP._SEG_3D)
        qapp.processEvents()
        assert calls == [SP._SEG_3D, SP._SEG_3D], (
            '数据变化后切回三维应重建')

    def test_apply_theme_invalidates_fingerprints(self, spatial):
        """主题切换清空指纹：下次刷新必须重新灌（视图内部条目被重建）。"""
        from ui.pages import spatial_page as SP

        view = spatial._views()[SP._SEG_3D]
        view.set_tracks = lambda tracks, colors: None
        spatial._apply_tracks_if_changed(
            SP._SEG_3D, view, [_Track('L01')], {})

        spatial.apply_theme(dark=True)

        assert spatial._views_applied == {}, '主题切换后指纹应清空'


class TestDepthScatterPool:
    """depth_slice_view：散点必须按颜色分组（单色快路径），不得逐点传色。"""

    def test_scatter_pool_one_item_per_color(self, qapp):
        from ui.widgets.depth_slice_view import DepthSliceView

        view = DepthSliceView()
        view.resize(600, 400)
        view.show()
        qapp.processEvents()

        tracks = [_Track('L01', 30), _Track('L02', 20), _Track('L03', 10)]
        colors = {'L01': '#ff0000', 'L02': '#00ff00', 'L03': '#0000ff'}
        view.set_tracks(tracks, colors)
        qapp.processEvents()

        pool = view._track_scatters
        assert set(pool.keys()) == {'#ff0000', '#00ff00', '#0000ff'}, (
            '每个颜色一个 scatter item（单色快路径）')
        counts = sorted(
            len(item.data['x']) for item in pool.values()
            if item.data['x'] is not None)
        assert counts == [10, 20, 30], f'点数按色分组应正确，实际 {counts}'
        # 缓存 (N,2) 供 _auto_range 使用
        assert view._track_xy is not None and len(view._track_xy) == 60

        view.deleteLater()
        qapp.processEvents()

    def test_clear_tracks_resets_pool(self, qapp):
        from ui.widgets.depth_slice_view import DepthSliceView

        view = DepthSliceView()
        view.resize(600, 400)
        view.show()
        qapp.processEvents()

        view.set_tracks([_Track('L01', 25)], {'L01': '#ff0000'})
        assert view._track_xy is not None
        view.clear_tracks()
        assert view._track_xy is None
        for item in view._track_scatters.values():
            assert item.data['x'] is None or len(item.data['x']) == 0

        view.deleteLater()
        qapp.processEvents()

    def test_same_color_reuses_item(self, qapp):
        """同色重复设置应复用 item 实例（不反复创建/销毁）。"""
        from ui.widgets.depth_slice_view import DepthSliceView

        view = DepthSliceView()
        view.resize(600, 400)
        view.show()
        qapp.processEvents()

        view.set_tracks([_Track('L01', 20)], {'L01': '#ff0000'})
        first = view._track_scatters['#ff0000']
        view.set_tracks([_Track('L01', 40)], {'L01': '#ff0000'})
        second = view._track_scatters['#ff0000']
        assert first is second, '同色应复用同一 item 实例'
        assert len(second.data['x']) == 40

        view.deleteLater()
        qapp.processEvents()
