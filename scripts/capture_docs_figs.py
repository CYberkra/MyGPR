#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成文档配图：B-scan 处理前后对比（教程/处理指南用）。

用营山真实数据（或自带样例）跑 robust_imaging 前 3 步，
渲染原始 vs 处理后双联图。产物 docs/user/images/processing_*.png。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / 'docs' / 'user' / 'images'

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor


def _load_data():
    """优先营山真实数据，回退自带样例。"""
    import os
    yingshan = os.environ.get('MYGPR_YINGSHAN_DATA')
    if yingshan:
        from core.gpr_data_model import load_gpr_dataset
        candidates = sorted(Path(yingshan).glob('Line*origin(36).csv'))
        if candidates:
            ds = load_gpr_dataset(candidates[0])
            return np.asarray(ds.matrix, dtype=np.float32), candidates[0].name
    from core.gpr_data_model import load_gpr_dataset
    ds = load_gpr_dataset(ROOT / 'sample_data' / 'gui_sidecar_all_data_main.csv')
    return np.asarray(ds.matrix, dtype=np.float32), 'sample_data'


def main() -> int:
    raw, name = _load_data()
    # 取前 800 道便于渲染
    raw = raw[:, :800]
    header = {'total_time_ns': 700.0}

    executor = NativeProcessingExecutor()
    current = raw
    steps = [('dewow', {'window': 23}), ('subtracting_average_2D', {'ntraces': 21})]
    for method_id, params in steps:
        result = executor.execute(
            ProcessingRequest(data=current, method_id=method_id,
                              params=params, header_info=header))
        current = np.asarray(result.data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    # 共享色标：两面板用同一 vmax（联合 98 分位），保证视觉可比
    vmax = float(np.percentile(np.abs(np.concatenate([raw.ravel(), current.ravel()])), 98))
    for ax, (data, title) in zip(axes, ((raw, '原始 B-scan'), (current, '处理后（dewow → 背景消除）'))):
        ax.imshow(data, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('道（沿测线）')
        ax.set_ylabel('采样点（时间）' if ax is axes[0] else '')
    fig.suptitle(f'MyGPR 处理对比 — {name}（共享色标 ±{vmax:.1f}）', fontsize=14)
    out = OUT_DIR / 'processing_before_after.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'[docs-figs] {out} ({out.stat().st_size // 1024} KB)')

    # AutoTune 概念图：处理链示意——颜色按功能类别着色（颜色携带意义）
    fig2, ax2 = plt.subplots(figsize=(12, 3.0), constrained_layout=True)
    # (算法, 功能类别)；类别色与 UI 语义色一致
    category_of = {
        'set_zero_time': '矫正', 'dewow': '矫正',
        'subtracting_average_2D': '背景消除',
        'fk_filter': '滤波', 'sec_gain': '增益',
        'svd_subspace': '去噪',
    }
    category_color = {
        '矫正': '#3b82f6', '背景消除': '#22c55e', '滤波': '#0ea5e9',
        '增益': '#f59e0b', '去噪': '#9467bd',
    }
    chain = ['set_zero_time', 'dewow', 'subtracting_average_2D',
             'fk_filter', 'sec_gain', 'svd_subspace']
    ax2.set_xlim(0, len(chain) * 2.2)
    ax2.set_ylim(0, 2.6)
    ax2.axis('off')
    seen_categories = set()
    for i, step in enumerate(chain):
        category = category_of[step]
        color = category_color[category]
        x = i * 2.2 + 0.2
        ax2.add_patch(plt.Rectangle((x, 1.2), 1.8, 0.8, facecolor=color,
                                    edgecolor='none', alpha=0.85, zorder=2))
        ax2.text(x + 0.9, 1.6, step, ha='center', va='center',
                 fontsize=9.5, color='white', zorder=3)
        if category not in seen_categories:
            seen_categories.add(category)
            ax2.add_patch(plt.Rectangle((x, 0.3), 0.35, 0.28, facecolor=color,
                                        edgecolor='none', alpha=0.85))
            ax2.text(x + 0.45, 0.44, category, ha='left', va='center',
                     fontsize=9.5, color='#374151')
        if i < len(chain) - 1:
            ax2.annotate('', xy=(x + 2.15, 1.6), xytext=(x + 1.85, 1.6),
                         arrowprops=dict(arrowstyle='->', color='#6b7280', lw=1.5))
    ax2.set_title('预设档 robust_imaging 的处理链（颜色 = 功能类别）', fontsize=13)
    out2 = OUT_DIR / 'processing_chain.png'
    fig2.savefig(out2, dpi=120)
    plt.close(fig2)
    print(f'[docs-figs] {out2} ({out2.stat().st_size // 1024} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
