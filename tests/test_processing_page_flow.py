# -*- coding: utf-8 -*-
"""处理页 v2 完整用户路径回归（真机走查固化，2026-09-26）。

路径：选测线 → 输入卡出现 → 加算法×3（画面不动）→ 运行（spinner）→
完成（按序出图）→ 改参数（琥珀提示）→ 排序（结果不丢）→ 再运行（新序
刷新）。每步断言——交互后状态是自动化断言，不是截图目检。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")


def _bundle(tag: float):
    matrix = np.full((60, 80), float(tag), dtype=np.float32)
    return SimpleNamespace(
        matrix=matrix, vmin=0.0, vmax=float(tag), title=f'data{tag}',
        x_label='道数', y_label='采样点', trace_axis_m=None, sample_axis=None,
        sample_axis_label='', trace_count=80, sample_count=60,
        trace_elevation_m=None, depth_axis_m=None)


def _artifact(aid, method, group, step, kind='intermediate'):
    return SimpleNamespace(
        artifact_id=aid, line_id='L01', name=f'run_{aid}',
        method_id=method, method_name=method,
        created_at=f'2026-09-26T1{step}:0{step}:00',
        manifest={'params': {'artifact_kind': kind, 'run_group_id': group,
                             'run_step_index': step}})


def _titles(page):
    return [c.title_label.text() for c in page._result_grid.cards()]


def _run_and_feed(page, group, aids_methods, tag_base):
    page.set_running(True)
    page.set_artifacts([_artifact(a, m, group, i + 1,
                                  kind=('processing' if i == len(aids_methods) - 1
                                        else 'intermediate'))
                        for i, (a, m) in enumerate(aids_methods)])
    for aid, _m in aids_methods:
        page.set_artifact_bundle(aid, _bundle(tag_base))
    page.set_running(False, success=True)


@pytest.fixture
def page(qapp):
    from ui.pages.processing_page import ProcessingPage
    page = ProcessingPage()
    yield page
    page.close()


class TestUserPathRegression:
    """完整用户路径（2026-09-26 真机走查固化）。"""

    def test_full_path(self, page):
        # 1. 选测线：仅输入卡
        page.set_original_bundle(_bundle(1.0))
        assert len(page._result_grid.cards()) == 1
        input_card = page._result_grid.cards()[0]
        assert input_card.view._matrix is not None

        # 2. 加算法 ×3：画面不动、无脏提示
        page._pipeline_list.add_step('dewow', '零时校正 (Dewow)', {})
        page._pipeline_list.add_step('sec_gain', 'SEC 增益 (AGC)', {})
        page._pipeline_list.add_step('agc', '自动增益控制 (AGC)', {})
        assert len(page._result_grid.cards()) == 1
        assert page._result_grid.cards()[0] is input_card
        assert not page._chain_strip._dirty_label.isVisibleTo(page._chain_strip)
        assert page._chain_strip._list.count() == 4

        # 3-4. 运行完成：按序出图 + 脏清除
        _run_and_feed(page, 'G1', [('R1', 'dewow'), ('R2', 'sec_gain'),
                                   ('R3', 'agc')], 10.0)
        assert _titles(page) == ['输入', '1 dewow', '2 sec_gain', '3 agc']
        assert all(c.view._matrix is not None for c in page._result_grid.cards())
        assert page._result_grid.cards()[0] is input_card
        assert not page._chain_strip._dirty_label.isVisibleTo(page._chain_strip)

        # 5. 改参数：琥珀提示、画面不动
        page._pipeline_list.update_step_params(1, {'alpha': 0.4})
        assert page._chain_strip._dirty_label.isVisibleTo(page._chain_strip)
        assert all(c.view._matrix is not None for c in page._result_grid.cards())

        # 6. 排序：结果仍为上次运行（点运行才更新）
        page._pipeline_list._move_step_to(0, 2)
        assert _titles(page) == ['输入', '1 dewow', '2 sec_gain', '3 agc']
        assert page._chain_strip._dirty_label.isVisibleTo(page._chain_strip)

        # 7. 再运行（新顺序）：按新 run_group 刷新
        _run_and_feed(page, 'G2', [('T1', 'sec_gain'), ('T2', 'dewow'),
                                   ('T3', 'agc')], 20.0)
        assert _titles(page) == ['输入', '1 sec_gain', '2 dewow', '3 agc']
        assert page._result_grid.cards()[0] is input_card
        assert not page._chain_strip._dirty_label.isVisibleTo(page._chain_strip)
