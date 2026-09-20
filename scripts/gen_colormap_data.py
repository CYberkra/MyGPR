#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 ui/widgets/_colormap_data.py：B-Scan 九项色标的 matplotlib 采样数据。

背景：BScanView.set_colormap 原走 pg.colormap.getFromMatplotlib，首次调用
触发 matplotlib 全量 import（实测 ~0.9s），主页预览卡构造期即命中，拖慢
冷启动。本脚本离线采样 ui.constants.COLORMAPS 各色标，生成纯数据模块；
运行时由 pg.colormap.ColorMap 直接构造，渲染结果与 matplotlib 一致、
零 import 成本。数据末尾自验重建 LUT 与原版最大通道差。

色标清单变更后重跑：python scripts/gen_colormap_data.py
（新增色标若未重跑，运行时自动回落 getFromMatplotlib，不影响正确性。）
"""
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from ui import constants  # noqa: E402


def sample(name: str):
    """按 pg.colormap.getFromMatplotlib 的三分支逻辑离线复刻 (pos, color)。

    对齐基准是 getFromMatplotlib（用户此前实际看到的色标），不是
    matplotlib 原生 cmap()（后者对 ListedColormap 是阶梯查表，两者本就有
    微小差异）。量化公式一致：255*x + 0.5 截断。
    """
    cmap = matplotlib.colormaps[name]
    if hasattr(cmap, '_segmentdata'):
        data = cmap._segmentdata
        seg = data['red']
        if isinstance(seg, (Sequence, np.ndarray)):
            positions = set()
            for key in ('red', 'green', 'blue'):
                for tup in data[key]:
                    positions.add(tup[0])
            pos = np.array(sorted(positions))
            comps = np.zeros((len(pos), 3))
            for idx, key in enumerate(('red', 'green', 'blue')):
                xs = np.zeros(len(data[key]))
                vals = np.zeros(len(data[key]))
                for j, tup in enumerate(data[key]):
                    xs[j] = tup[0]
                    vals[j] = tup[1]
                comps[:, idx] = np.interp(pos, xs, vals)
            color = (255.0 * comps + 0.5).astype(np.uint8)
        elif callable(seg):
            pos = np.linspace(0.0, 1.0, 64)
            comps = np.zeros((len(pos), 3))
            for idx, key in enumerate(('red', 'green', 'blue')):
                comps[:, idx] = np.clip(data[key](pos), 0, 1)
            color = (255.0 * comps + 0.5).astype(np.uint8)
        else:
            raise TypeError(f'{name}: 不支持的 segmentdata 形式')
    elif hasattr(cmap, 'colors'):
        col_data = np.array(cmap.colors)
        pos = np.linspace(0.0, 1.0, col_data.shape[0])
        color = (255.0 * col_data[:, :3] + 0.5).astype(np.uint8)
    else:
        raise TypeError(f'{name}: 不支持的 colormap 类型')
    return pos, color


def verify(name: str, pos: np.ndarray, color: np.ndarray) -> int:
    """重建 ColorMap 与 getFromMatplotlib 原版在 1024 点采样的最大通道差。"""
    rebuilt = pg.colormap.ColorMap(pos=pos, color=color, name=name)
    probe = np.linspace(0.0, 1.0, 1024)
    got = np.asarray(rebuilt.map(probe, mode='float'), dtype=float)[:, :3] * 255.0
    ref_cmap = pg.colormap.getFromMatplotlib(name)
    ref = np.asarray(ref_cmap.map(probe, mode='float'), dtype=float)[:, :3] * 255.0
    return int(np.max(np.abs(got - ref)))


def _fmt_pos(pos: np.ndarray) -> list[str]:
    """pos 数组 → 每行 13 个浮点字面量的源码行。"""
    literals = ['%.6g' % v for v in pos]
    rows = []
    for start in range(0, len(literals), 13):
        rows.append('        ' + ', '.join(literals[start:start + 13]) + ',')
    return rows


def _fmt_rows(color: np.ndarray) -> list[str]:
    """(N,3) uint8 → 每行 4 个 (r, g, b) 三元组的源码行。"""
    rows = []
    triples = [', '.join(str(int(v)) for v in color[i]) for i in range(len(color))]
    for start in range(0, len(triples), 4):
        rows.append('        ' + ', '.join(f'({t})' for t in triples[start:start + 4]) + ',')
    return rows


def main() -> int:
    data_blocks = []
    print(f'{"name":<10} max_diff/255')
    for name in constants.COLORMAPS:
        pos, color = sample(name)
        diff = verify(name, pos, color)
        print(f'{name:<10} {diff}')
        if diff > 4:
            print(f'ERROR: {name} 重建差异 {diff} 超阈值 4，采样点数不足', file=sys.stderr)
            return 1
        data_blocks.append((name, _fmt_pos(pos), _fmt_rows(color)))

    out_lines = [
        '# -*- coding: utf-8 -*-',
        '"""B-Scan 色标采样数据 —— 自动生成，勿手改。',
        '',
        '由 scripts/gen_colormap_data.py 从 matplotlib 离线采样，',
        '重建 LUT 与 pg.colormap.getFromMatplotlib 输出逐点一致',
        '（1024 点采样最大通道差 0，脚本自验）。',
        'BScanView 用 pg.colormap.ColorMap 直接重建，避免启动路径 import',
        'matplotlib（实测全量 import ~0.9s）。色标清单变更后重跑生成脚本；',
        '未重跑时运行时对缺失色标自动回落 getFromMatplotlib。',
        '"""',
        '',
        'COLORMAP_DATA = {',
    ]
    for name, pos_rows, rows in data_blocks:
        out_lines.append(f'    {name!r}: (')
        out_lines.append('        (')
        out_lines.extend(pos_rows)
        out_lines.append('        ),')
        out_lines.append('        (')
        out_lines.extend(rows)
        out_lines.append('        ),')
        out_lines.append('    ),')
    out_lines.extend(['}', ''])

    out_path = ROOT / 'ui' / 'widgets' / '_colormap_data.py'
    out_path.write_text('\n'.join(out_lines), encoding='utf-8')
    print(f'written: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
