# -*- coding: utf-8 -*-
"""B-Scan 海拔模式：整幅剖面重采样到共享海拔网格（用户选型）。

核心语义（改动即回归）：

1. **共享绝对海拔网格**：行 r 的海拔 = ``elev_axis[r]``（降序，所有道
   共用）。第 i 道第 j 个采样点画在海拔 ``ground[i] - depth[j]`` 处——
   每道用自己的地面高程，**不是**某个平均基准。
2. **顶边 = 地形线**：最高地面占据第 0 行，地面更低的道顶部自然留出
   NaN（pyqtgraph 渲染为透明），顶边随真实地貌起伏。
3. **显示层变换**：raw 矩阵一个字节不动；warp 结果按「同一份数据只算
   一次」缓存。
4. 退化：非有限逐道高程 → 该道整列 NaN；深度轴病态 / 高程全缺 →
   回落 (None, None)，显示层退回采样轴。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.widgets.bscan_axes import build_elevation_view  # noqa: E402
from ui.widgets.bscan_view import BScanView  # noqa: E402

# 合成数据：深度轴 0.5m 等距（网格命中精确、无插值误差），地形 4m 台阶
TRACES, SAMPLES = 60, 120
DEPTH_STEP = 0.5
DEPTH = np.arange(SAMPLES, dtype=np.float64) * DEPTH_STEP      # 0 .. 59.5 m
GROUND = np.where(np.arange(TRACES) < TRACES // 2, 500.0, 504.0)
TERRAIN_ROWS = 8                                               # 4m / 0.5m


def make_matrix() -> np.ndarray:
    """行号即值：warp 后逐行数值可直接对账（值 = 深度/步距）。"""
    return np.tile(np.arange(SAMPLES, dtype=np.float32)[:, None],
                   (1, TRACES))


# ============================================================ 纯函数内核
class TestBuildElevationView:
    def test_rows_cover_terrain_and_span(self):
        warped, axis = build_elevation_view(make_matrix(), GROUND, DEPTH)
        assert warped.shape == (SAMPLES + TERRAIN_ROWS, TRACES)
        assert axis.shape == (SAMPLES + TERRAIN_ROWS,)
        assert float(axis[0]) == pytest.approx(504.0)          # 最高地面
        step = float(axis[0]) - float(axis[1])
        assert step == pytest.approx(DEPTH_STEP)

    def test_flat_terrain_is_identity(self):
        ground = np.full(TRACES, 500.0)
        matrix = make_matrix()
        warped, axis = build_elevation_view(matrix, ground, DEPTH)
        assert warped.shape == matrix.shape
        np.testing.assert_allclose(warped, matrix, atol=1e-4)

    def test_top_row_is_highest_ground_surface(self):
        """第 0 行 = 最高地面：该道地表采样点原样出现在第 0 行。"""
        matrix = make_matrix()
        warped, _ = build_elevation_view(matrix, GROUND, DEPTH)
        high = TRACES - 1                       # 地面 504（= 网格顶）
        assert warped[0, high] == pytest.approx(matrix[0, high], abs=1e-4)

    def test_lower_ground_has_whitespace_on_top(self):
        """地面低 4m 的道顶部 8 行无数据（NaN），其下紧贴该道地表。"""
        matrix = make_matrix()
        warped, _ = build_elevation_view(matrix, GROUND, DEPTH)
        low = 0                                 # 地面 500
        assert np.isnan(warped[:TERRAIN_ROWS, low]).all()
        assert warped[TERRAIN_ROWS, low] == pytest.approx(matrix[0, low],
                                                           abs=1e-4)

    def test_values_follow_absolute_elevation(self):
        """同一海拔行上，不同道的值对应各自的深度（值 = 深度/步距）。"""
        matrix = make_matrix()
        warped, _ = build_elevation_view(matrix, GROUND, DEPTH)
        row = TERRAIN_ROWS + 10                 # 海拔 504 - 9m = 495
        assert warped[row, TRACES - 1] == pytest.approx(18.0, abs=1e-3)
        assert warped[row, 0] == pytest.approx(10.0, abs=1e-3)

    def test_nonfinite_ground_column_is_all_nan(self):
        ground = GROUND.copy()
        ground[3] = np.nan
        warped, _ = build_elevation_view(make_matrix(), ground, DEPTH)
        assert np.isnan(warped[:, 3]).all()
        assert np.isfinite(warped[:, 0]).any()   # 其余道不受影响

    def test_rows_capped(self):
        warped, _ = build_elevation_view(make_matrix(), GROUND, DEPTH,
                                         max_rows=10)
        assert warped.shape[0] == 10

    @pytest.mark.parametrize('ground,depth', [
        (None, DEPTH),
        (GROUND, None),
        (GROUND, np.array([0.0, float('nan'), 2.0])),
        (np.array([np.nan] * TRACES), DEPTH),
        (GROUND, np.linspace(5.0, 0.0, SAMPLES)),   # 步距为负 → 病态
    ])
    def test_degenerate_inputs_return_none(self, ground, depth):
        assert build_elevation_view(make_matrix(), ground, depth) == (None, None)

    def test_rejects_non_2d(self):
        assert build_elevation_view(np.zeros(10), GROUND, DEPTH) == (None, None)
        assert build_elevation_view(None, GROUND, DEPTH) == (None, None)


# ============================================================ 视图集成
@pytest.fixture
def view(qapp):
    from core.gui_rendering import make_preview_bundle

    widget = BScanView()
    matrix = make_matrix()
    widget.set_bundle(make_preview_bundle(
        matrix,
        sample_axis=DEPTH.copy(),
        sample_axis_label='深度 (m)',
        depth_axis_m=DEPTH.copy(),
        trace_elevation_m=GROUND.copy(),
    ))
    widget._raw_matrix_sentinel = matrix.copy()
    yield widget
    widget.close()


def switch_elevation(view: BScanView) -> None:
    view.set_y_axis_mode('elevation')


class TestElevationImageSwap:
    def test_switch_warps_image(self, view):
        switch_elevation(view)
        img = view._image_item.image
        assert img.shape == (SAMPLES + TERRAIN_ROWS, TRACES)
        assert view._showing_elevation is True
        assert view.effective_y_axis_mode() == 'elevation'

    def test_whitespace_and_surface_alignment(self, view):
        switch_elevation(view)
        img = view._image_item.image
        assert np.isnan(img[:TERRAIN_ROWS, 0]).all()      # 低地面道顶部留白
        assert np.isfinite(img[0, TRACES - 1])            # 最高地面顶行有数据

    def test_axis_label_and_ticks(self, view):
        switch_elevation(view)
        assert view._plot.getAxis('left').labelText == '海拔 (m)'
        rows = view._image_shape[1]
        ticks = view._convert_left_ticks([0, rows - 1], 1.0, 1.0)
        assert ticks is not None
        assert float(ticks[0]) == pytest.approx(504.0, abs=0.01)

    def test_switch_back_restores_raw(self, view):
        switch_elevation(view)
        view.set_y_axis_mode('sample')
        assert view._showing_elevation is False
        assert view._image_item.image.shape == (SAMPLES, TRACES)
        assert view._convert_left_ticks([0.0], 1.0, 1.0) is None

    def test_raw_matrix_untouched(self, view):
        """显示层变换零变异：无论怎么切轴，set_bundle 传入的矩阵不变。"""
        before = view._matrix.copy()
        switch_elevation(view)
        view.set_y_axis_mode('sample')
        switch_elevation(view)
        np.testing.assert_array_equal(view._matrix, before)
        np.testing.assert_array_equal(view._matrix,
                                      view._raw_matrix_sentinel)

    def test_warp_is_cached_per_data(self, view):
        switch_elevation(view)
        src = view._gain_applied(view._matrix)   # 增益是 warp 的原料（off=原样）
        first = view._warped_elevation(src)[0]
        view.set_y_axis_mode('sample')
        switch_elevation(view)
        assert view._warped_elevation(src)[0] is first   # 同数据不重算


class TestElevationCoordinates:
    def test_view_to_data_surface_hit(self, view):
        """海拔模式点击某道地表行 → 该道深度 0 采样点。"""
        switch_elevation(view)
        # 低地面道：地表在第 TERRAIN_ROWS 行
        t, s = view._view_to_data(0, TERRAIN_ROWS)
        assert (t, s) == (0, 0)
        # 高地面道：地表在第 0 行
        t, s = view._view_to_data(TRACES - 1, 0)
        assert (t, s) == (TRACES - 1, 0)

    def test_sample_to_row_round_trip(self, view):
        switch_elevation(view)
        for trace in (0, TRACES - 1):
            for sample in (0, 1, 50):
                row = view._sample_to_row(float(trace), float(sample))
                back = view._row_to_sample(trace, int(round(row)))
                assert back == sample

    def test_overlay_points_land_on_warped_image(self, view):
        """解释页标注（原始坐标）必须落在 warp 后图像的正确位置。"""
        switch_elevation(view)
        t, r = view._data_to_view(0, 0)
        assert t == pytest.approx(0.0)
        assert r == pytest.approx(TERRAIN_ROWS)          # 低地面道的地表行

    def test_sample_mode_conversions_unchanged(self, view):
        """采样轴模式坐标换算保持原语义（海拔改造不得扰动）。"""
        t, s = view._view_to_data(3, 7)
        assert (t, s) == (3, 7)
        t, r = view._data_to_view(3, 7)
        assert (t, r) == (3.0, 7.0)


class TestElevationWiggle:
    def test_wiggle_rows_follow_terrain(self, view):
        """海拔模式波形：每道从自己的地面高程起画（低地面道起点更靠下）。"""
        switch_elevation(view)
        ys_col, ys_bottom = view._wiggle_y_arrays(
            SAMPLES, SAMPLES, 1, TRACES, view._wiggle_y_context())
        assert ys_col.shape == (SAMPLES, TRACES)
        assert ys_col[0, 0] == pytest.approx(TERRAIN_ROWS)   # 地面 500
        assert ys_col[0, TRACES - 1] == pytest.approx(0.0)   # 地面 504
        assert (ys_bottom <= ys_col.max(axis=0) + 1e-9).all()

    def test_wiggle_rows_without_context_use_row_index(self, view):
        ys_col, ys_bottom = view._wiggle_y_arrays(
            SAMPLES, SAMPLES // 2, 2, TRACES, None)
        assert ys_col[0, 0] == 0.0
        assert ys_col[1, 0] == 2.0
        assert (ys_bottom == float(SAMPLES - 1)).all()


class TestToolbarTrim:
    """工具条退役（2026-09-23 评审）：4 钮与右键菜单缩放组完全重复。"""

    def test_toolbar_removed_fullscreen_button_floats(self, view):
        assert not hasattr(view, '_toolbar')
        assert not hasattr(view, '_toolbar_buttons')
        for name in ('_zoom_in_btn', '_zoom_out_btn', '_expand_btn',
                     '_fit_btn', '_square_btn', '_one_to_one_btn',
                     '_y_sample_btn', '_y_elevation_btn',
                     '_x_trace_btn', '_x_distance_btn'):
            assert not hasattr(view, name), f'{name} 应已退役'
        assert view._fullscreen_btn.parent() is view   # 悬浮画布左上角

    def test_capabilities_survive_in_context_menu(self, view):
        """删掉的是入口不是能力：右键菜单仍能切比例与轴单位。"""
        view.fit_square()
        assert view.aspect_mode() == 'square'
        view.fit_to_data()
        assert view.aspect_mode() == 'free'
        switch_elevation(view)
        assert view.y_axis_mode() == 'elevation'

    def test_expand_survives_as_context_menu_capability(self, view):
        """铺满钮退役后能力保留：宿主接管后仍可请求收起页面侧栏。"""
        got = []
        view.sig_expand_requested.connect(lambda: got.append(True))
        view.set_expand_enabled(True)
        view.request_expand()
        assert got == [True]
