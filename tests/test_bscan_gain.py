# -*- coding: utf-8 -*-
"""B-Scan 显示增益（SEC）：显示域逐行缩放，raw 零变异。

语义要点：

- 增益是显示链第一级：raw → 增益 → （海拔 warp）→ 图像/波形/色阶百分位；
  ``_matrix`` 本体永远保留 raw（十字读数契约）；
- SEC 曲线 g(d) = (d+d0)/d0 × 10^(αd/20)，限幅 1000×（60 dB）；
- 缓存按 (矩阵身份, α) 判重（与海拔 warp 缓存同一纪律）；
- 跟随既有偏好语义：右键改单视图 + notify 写盘，设置页改动全量下发。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过


def _make_view(qapp, **kwargs):
    from types import SimpleNamespace

    from ui.widgets.bscan_view import BScanView
    view = BScanView(**kwargs)
    rng = np.random.default_rng(7)
    n_samples, n_traces = 200, 60
    direct = 1.0 * np.exp(-np.arange(n_samples) / 6.0)[:, None]
    hyper = np.zeros((n_samples, n_traces), dtype=np.float64)
    for t0, amp in ((90, 0.05), (150, 0.02)):
        for trace in range(n_traces):
            tt = np.sqrt(max((trace - n_traces / 2) ** 2 + t0 ** 2, 0.0))
            hyper[int(tt), trace] += amp
    matrix = (direct + hyper
              + rng.normal(0, 0.004, (n_samples, n_traces))).astype('float32')
    bundle = SimpleNamespace(
        matrix=matrix, vmin=0.0, vmax=1.0, title='t',
        x_label='道数', y_label='双程走时 (ns)',
        trace_axis_m=np.arange(n_traces, dtype=float) * 0.5,
        sample_axis=np.linspace(0.0, 60.0, n_samples),
        sample_axis_label='双程走时 (ns)',
        trace_count=n_traces, sample_count=n_samples,
        trace_elevation_m=None, depth_axis_m=None)
    view.set_bundle(bundle)
    view.resize(400, 300)
    view.show()
    qapp.processEvents()
    return view, matrix


class TestSecGainOnView:
    def test_default_off(self, qapp):
        view, _ = _make_view(qapp)
        try:
            assert view.gain_mode() == 'off'
            assert view._gain_applied(view._matrix) is view._matrix
        finally:
            view.close()

    def test_sec_changes_display_keeps_raw(self, qapp):
        view, matrix = _make_view(qapp)
        try:
            view.set_gain('sec', alpha=1.0)
            assert view.gain_mode() == 'sec'
            shown = np.asarray(view._image_item.image, dtype=np.float64)
            raw = np.asarray(view._matrix, dtype=np.float64)
            assert not np.allclose(shown, raw), '增益开启后上屏值必须变化'
            np.testing.assert_allclose(view._matrix, matrix)   # raw 零变异
        finally:
            view.close()

    def test_off_restores_display(self, qapp):
        view, _ = _make_view(qapp)
        try:
            before = np.asarray(view._image_item.image, dtype=np.float64).copy()
            view.set_gain('sec', alpha=1.0)
            view.set_gain('off')
            after = np.asarray(view._image_item.image, dtype=np.float64)
            np.testing.assert_allclose(after, before)
            assert view._gain_applied(view._matrix) is view._matrix
        finally:
            view.close()

    def test_levels_follow_gain(self, qapp):
        """色阶在增益后矩阵上取百分位：开增益后 vmin/vmax 必须移动。"""
        view, _ = _make_view(qapp)
        try:
            levels_before = tuple(float(v) for v in view._image_item.levels)
            view.set_gain('sec', alpha=1.5)
            levels_after = tuple(float(v) for v in view._image_item.levels)
            assert levels_after != levels_before
        finally:
            view.close()

    def test_gain_cache_reuse_and_invalidate(self, qapp):
        view, _ = _make_view(qapp)
        try:
            view.set_gain('sec', alpha=1.0)
            first = view._gain_applied(view._matrix)
            assert view._gain_applied(view._matrix) is first   # 同参复用
            view.set_gain('sec', alpha=2.0)
            assert view._gain_applied(view._matrix) is not first
        finally:
            view.close()

    def test_notify_emits_signal(self, qapp):
        view, _ = _make_view(qapp)
        try:
            got = []
            view.sig_gain_changed.connect(lambda m: got.append(str(m)))
            view.set_gain('sec', notify=True)
            assert got == ['sec']
            view.set_gain('off')               # notify=False 不发
            assert got == ['sec']
        finally:
            view.close()

    def test_unknown_mode_raises(self, qapp):
        view, _ = _make_view(qapp)
        try:
            with pytest.raises(ValueError):
                view.set_gain('agc')
        finally:
            view.close()

    def test_tvg_changes_display_keeps_raw(self, qapp):
        view, matrix = _make_view(qapp)
        try:
            view.set_gain('tvg', db=36.0, power=1.0)
            shown = np.asarray(view._image_item.image, dtype=np.float64)
            raw = np.asarray(view._matrix, dtype=np.float64)
            assert not np.allclose(shown, raw)
            np.testing.assert_allclose(view._matrix, matrix)   # raw 零变异
        finally:
            view.close()

    def test_tvg_curve_shape(self, qapp):
        """g(0)=1、单调升、深端恰为 10^(db/20)、限幅 1000×。"""
        view, _ = _make_view(qapp)
        try:
            view.set_gain('tvg', db=36.0, power=1.0)
            gain = view._tvg_row_gain(200)
            assert gain.shape == (200,)
            assert gain[0] == pytest.approx(1.0)
            assert (np.diff(gain) >= 0).all()
            assert gain[-1] == pytest.approx(10.0 ** (36.0 / 20.0))
            assert gain.max() <= 1000.0
        finally:
            view.close()

    def test_tvg_power_concentrates_gain_deep(self, qapp):
        """power>1 时同深度的中段增益应低于线性（增益向深部集中）。"""
        view, _ = _make_view(qapp)
        try:
            view.set_gain('tvg', db=36.0, power=1.0)
            linear = view._tvg_row_gain(200)
            view.set_gain('tvg', db=36.0, power=2.0)
            curved = view._tvg_row_gain(200)
            assert curved[100] < linear[100]          # 中段被压低
            assert curved[-1] == pytest.approx(linear[-1])   # 深端不变
        finally:
            view.close()

    def test_tvg_params_notify_emits_signal(self, qapp):
        view, _ = _make_view(qapp)
        try:
            got = []
            view.sig_gain_params_changed.connect(
                lambda db, power: got.append((float(db), float(power))))
            view.set_gain('tvg', db=30.0, power=1.5, notify=True)
            assert got == [(30.0, 1.5)]
            view.set_gain('tvg', db=40.0, power=1.5)   # notify=False 不发
            assert got == [(30.0, 1.5)]
        finally:
            view.close()

    def test_sec_curve_shape(self, qapp):
        """曲线浅端=1、单调升、限幅 1000×：物理近似的硬边界。"""
        view, _ = _make_view(qapp)
        try:
            gain = view._sec_row_gain(200)
            assert gain.shape == (200,)
            assert gain[0] == pytest.approx(1.0)
            assert (np.diff(gain) >= 0).all()
            assert gain.max() <= 1000.0
        finally:
            view.close()
