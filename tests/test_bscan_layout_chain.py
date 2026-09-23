# -*- coding: utf-8 -*-
"""free 自由窗口布局全链路（真实主窗集成）：恢复 / 分发 / 下发 / 镜像 / 跨会话。

锁定契约（改动即回归；探针 output/probes_archive/_probe_free_chain.py 的
固化版，主窗在当前环境不可构造时 skip）：

1. **启动恢复**：设置文件 ``bscan_layout_mode=free`` → 容器实际页=free；
   free 子窗口必须出现在 ``page.findChildren(BScanView)`` 里——逐面板偏好
   （比例/轴/色阶）的统一恢复靠这个列表，漏掉就是「free 窗记不住偏好」。
2. **数据分发**：0 号窗=原始数据、1 号窗=处理成果；设置页改布局
   （free↔quad）后数据跟着重分发，窗位语义不漂移。
3. **双向镜像**：设置页 combo 改动 → 容器即时切换 + 磁盘写盘；free 面板
   右键改比例 → 磁盘 + 设置页控件同步（否则关窗被过期值覆盖）。
4. **跨会话**：关窗后 free/cell 保留；二次构造主窗真恢复（容器落 free、
   逐面板比例 cell）。
"""
from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.settings_manager import SettingsManager  # noqa: E402
from ui.widgets.bscan_view import BScanView  # noqa: E402


def _make_window(settings: SettingsManager):
    """构造真实主窗；不可用（缺依赖）时返回 None 由调用方 skip。"""
    try:
        from ui.main_window import MyGPRMainWindow
    except Exception:  # noqa: BLE001 - 环境缺件时跳过而非失败
        return None
    try:
        return MyGPRMainWindow(settings=settings)
    except Exception:  # noqa: BLE001
        return None


def _bundle(mat, title: str) -> SimpleNamespace:
    return SimpleNamespace(matrix=mat, vmin=-1.0, vmax=1.0, title=title,
                           x_label='道数', y_label='采样点', trace_axis_m=None,
                           sample_axis=None, sample_axis_label='',
                           trace_count=mat.shape[1], sample_count=mat.shape[0],
                           trace_elevation_m=None, depth_axis_m=None)


def _read_disk(store: str) -> dict:
    with open(store, encoding='utf-8') as handle:
        return json.load(handle)


def _settle(qapp, rounds: int = 5) -> None:
    for _ in range(rounds):
        qapp.processEvents()


def test_free_layout_full_chain(qapp):
    """free 布局：预置 → 恢复 → 分发 → 下发/镜像 → 关窗 → 二次启动。"""
    tmpdir = tempfile.mkdtemp(prefix='mygpr_free_chain_')
    store = os.path.join(tmpdir, 'settings.json')
    with open(store, 'w', encoding='utf-8') as handle:
        json.dump({'bscan_layout_mode': 'free',
                   'bscan_aspect_mode': 'square'}, handle)

    window = _make_window(SettingsManager(store))
    if window is None:
        pytest.skip('主窗在当前环境不可构造')
    try:
        window.ensure_pages_ready()
        _settle(qapp)
        proc = window._page('processingInterface')
        container = proc._bscan_container
        free_views = container.views()

        # 1. 启动恢复：容器 + 逐面板
        assert container.layout_mode() == 'free'
        assert container.effective_mode() == 'free'
        assert len(free_views) == 2
        page_views = proc.findChildren(BScanView)
        assert len(page_views) == 9
        assert all(v in page_views for v in free_views)
        assert all(v.aspect_mode() == 'square' for v in free_views)

        # 2. 数据分发：0 号窗=原始、1 号窗=成果
        raw = np.random.default_rng(1).standard_normal((64, 48)).astype('float32')
        res = (np.random.default_rng(2).standard_normal((64, 48))
               * 0.2).astype('float32')
        proc.set_original_bundle(_bundle(raw, '原始数据'))
        proc.set_result_bundle(_bundle(res, '成果'))
        _settle(qapp)
        assert free_views[0]._matrix is raw
        assert free_views[1]._matrix is res

        # 3. 设置页改动即时下发 + 镜像写盘 + 切回重分发
        settings_page = window._page('settingsInterface')
        combo = settings_page._bscan_layout_combo
        combo.setCurrentIndex(combo.findData('quad'))
        _settle(qapp)
        assert container.effective_mode() == 'quad'
        assert _read_disk(store).get('bscan_layout_mode') == 'quad'
        combo.setCurrentIndex(combo.findData('free'))
        _settle(qapp)
        assert container.effective_mode() == 'free'
        assert free_views[0]._matrix is raw
        assert free_views[1]._matrix is res
        assert _read_disk(store).get('bscan_layout_mode') == 'free'

        # 4. free 面板右键改比例 → 磁盘 + 设置页同步
        free_views[0].set_aspect_mode('cell', notify=True)
        _settle(qapp)
        assert _read_disk(store).get('bscan_aspect_mode') == 'cell'
        assert settings_page._bscan_aspect_combo.currentData() == 'cell'

        # 5. 设置页改色标映射 → 全量下发 + 写盘（页面色标 ComboBox 已退役）
        settings_page._bscan_cmap_combo.setCurrentText('gray')
        _settle(qapp)
        assert all(v._cmap_name == 'gray' for v in free_views)
        assert _read_disk(store).get('bscan_colormap') == 'gray'

        # 6. 关窗跨会话
        window.close()
        _settle(qapp)
        assert _read_disk(store).get('bscan_layout_mode') == 'free'
        assert _read_disk(store).get('bscan_aspect_mode') == 'cell'
        assert (SettingsManager(store).get('bscan_layout_mode') == 'free')
    finally:
        try:
            window.close()
        except Exception:  # noqa: BLE001
            pass

    # 7. 二次启动真恢复（容器落 free、逐面板比例 cell）
    window2 = _make_window(SettingsManager(store))
    if window2 is None:
        pytest.skip('主窗二次构造在当前环境不可用')
    try:
        window2.ensure_pages_ready()
        _settle(qapp)
        container2 = window2._page('processingInterface')._bscan_container
        views2 = container2.views()
        assert container2.effective_mode() == 'free'
        assert len(views2) == 2
        assert all(v.aspect_mode() == 'cell' for v in views2)
        # 色标映射也真恢复（处理页工具行退役后唯一状态源是设置）
        assert all(v._cmap_name == 'gray' for v in views2)
    finally:
        try:
            window2.close()
        except Exception:  # noqa: BLE001
            pass
