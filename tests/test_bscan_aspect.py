# -*- coding: utf-8 -*-
"""B-Scan 显示比例策略测试（offscreen 控件 + 设置持久化）。

背景（2026-09-22）：用户报告「bscan 目前总是有很多占空的地方」。实测三种
比例模式的数据占画布面积比：

    square  43.3%（留白 56.7%，x/y 拉伸比 0.50 → 把 4:1 剖面压成 1:1）
    free    85.9%（留白 14.1%，x/y 拉伸比 1.09 → 几乎不畸变）
    cell    93.7%（留白  6.3%）

旧默认 'square' 在「占空」与「保真」两项都最差，故改为 'free'。本文件锁定
该默认值与三模式切换 API，防止后续重构悄悄改回。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.widgets.bscan_view import BScanView  # noqa: E402


@pytest.fixture
def view(qapp):
    v = BScanView()
    yield v
    v.close()


class TestDefaultAspect:
    """默认比例必须铺满（回归防护：改回 square 会让用户重新看到大片留白）。"""

    def test_default_is_free(self, view):
        assert view.aspect_mode() == 'free'

    def test_default_not_square(self, view):
        assert view.aspect_mode() != 'square', (
            'square 占空比仅 43.3%（free 85.9%），不应作为默认')

    def test_explicit_default_respected(self, qapp):
        v = BScanView(default_aspect='square')
        assert v.aspect_mode() == 'square'
        v.close()

    def test_illegal_default_falls_back_to_free(self, qapp):
        v = BScanView(default_aspect='nonsense')
        assert v.aspect_mode() == 'free'
        v.close()


class TestSetAspectMode:
    """按名切换 API：三模式各归位，非法名不改动现状。"""

    @pytest.mark.parametrize('mode', ['free', 'square', 'cell'])
    def test_roundtrip(self, view, mode):
        view.set_aspect_mode(mode)
        assert view.aspect_mode() == mode

    def test_illegal_mode_is_noop(self, view):
        view.set_aspect_mode('square')
        view.set_aspect_mode('bogus')
        assert view.aspect_mode() == 'square'


class TestAspectSignal:
    """sig_aspect_changed：用户切换发射、数据驱动重排不发射。"""

    def test_user_switch_emits(self, view):
        seen = []
        view.sig_aspect_changed.connect(seen.append)
        view.fit_square()
        view.fit_to_data()
        view.reset_1to1()
        assert seen == ['square', 'free', 'cell']

    def test_restore_does_not_emit(self, view):
        """恢复持久化设置时必须 notify=False，否则「读设置→写设置」回环。"""
        seen = []
        view.sig_aspect_changed.connect(seen.append)
        view.set_aspect_mode('square', notify=False)
        assert seen == []
        assert view.aspect_mode() == 'square'

    def test_data_driven_refit_does_not_emit(self, view):
        """新数据到达时的重排属于数据驱动，不该触发持久化写盘。"""
        import numpy as np
        seen = []
        view.sig_aspect_changed.connect(seen.append)
        view.set_matrix(np.random.rand(80, 160).astype('float32'), 0.0, 1.0)
        assert seen == []
        assert view.aspect_mode() == 'free'


class TestAspectPersistence:
    """设置键 bscan_aspect_mode 的默认值与合法性。"""

    def test_settings_key_default(self, tmp_path):
        from ui.settings_manager import SettingsManager
        sm = SettingsManager(str(tmp_path / 's.json'))
        assert sm.get('bscan_aspect_mode') == 'free'

    def test_settings_roundtrip(self, tmp_path):
        from ui.settings_manager import SettingsManager
        path = str(tmp_path / 's.json')
        sm = SettingsManager(path)
        sm.set('bscan_aspect_mode', 'square')
        sm.save()
        assert SettingsManager(path).get('bscan_aspect_mode') == 'square'
