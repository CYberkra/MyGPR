#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NaN 感知的 marching squares 等值线（替代 pyqtgraph ``fn.isocurve``）。

为什么需要它
------------
pyqtgraph 的 ``functions.isocurve`` 用 ``mask = data < level`` 判格点在 level
的哪一侧（``functions.py:2380``），而 **NaN 与任何值比较恒为 False** —— NaN
被归入"高于level"侧。于是无数据空洞的边界被当成真实跃变，横穿空洞画出伪
等值线。生产矩阵绝大多数格子是 NaN（``grid/service.py:240`` 用 NaN 填空
cell），所以这是常态而非 corner case。

实测（修复前）
    20×20 全 NaN + 唯一点，level 超值域→ 1 节点折线，**100% 落在无数据格内**
    40×40 稀疏场（8个有效点）→同样 100%

为什么"填值"治不了
-----------------
试过把 NaN 填成最近有效值（``scipy.ndimage.distance_transform_edt``），
**失败**：40×40 稀疏场仍有 98% 节点落在空洞内。原因是最近邻填充是 Voronoi
式外推，空洞内部并不平（相邻格取到不同邻居），照样跨 level。
填常数（0 / min / +inf）也不行—— ``+inf < level`` 同样恒为 False，与 NaN
**判同侧**，一条不少；而填0 会在 level < 0 时凭空造出另一批伪线。
根因是marching squares 只有"高于/ 低于"两态，**无法表达"无数据"这个第三态**，
任何在进算法前消掉 NaN 的做法都只是把伪影挪位置。

正确做法
--------
按 marching squares 的定义逐格判断：**格子的 4 个角只要有一个不是有限值，
这个格子就不产生任何线段**（不做插值、不跨空洞）。这样等值线只画在
"四角皆有数据"的格子上，既不横穿空洞、也不沿空洞边界凭空出现。

线段连接沿用 pyqtgraph 的 ``gridKey`` 机制（``i + (1 if edge==2 else 0)``,
``j + (1 if edge==3 else 0)``, ``edge % 2``），保证同一位置的相邻线段能接上，
与 ``connected=True`` 的行为一致。

对外只暴露 :func:`isoline_paths`，返回 ``list[list[tuple[float, float]]]``；
调用方（DepthSliceView）把它们moveTo/lineTo 进一个 QPainterPath。
"""
from __future__ import annotations

import numpy as np

__all__ = ['isoline_paths']

# 与 pyqtgraph functions.isocurve 相同的查表（顶点编号 Bourk 方案）
_SIDE_TABLE = (
    (), (0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (0, 1, 2, 3),
    (2, 3), (2, 3), (0, 1, 2, 3), (1, 3), (0, 3), (0, 2), (1, 2), (0, 1), (),
)

_EDGE_KEY = (
    ((0, 1), (0, 0)),
    ((0, 0), (1, 0)),
    ((1, 0), (1, 1)),
    ((1, 1), (0, 1)),
)


def isoline_paths(data: np.ndarray, level: float,
                  extend_to_edge: bool = True,
                  axis_order: str = 'row-major'
                  ) -> list[list[tuple[float, float]]]:
    """生成等值线折线，**跳过含非有限角点的格子**。

    参数
    ----
    data : 2D float array
    level : 等值线场值
    extend_to_edge : 与 pyqtgraph 同名参数同义——把边缘值向外扩一格，使等值线
        能延伸到数据边界。无效角的格子依然被跳过，所以不会画出越界的线。
    axis_order : ``'row-major'``（默认，与 ``pg.IsocurveItem(axisOrder=...)``
        的默认值一致）表示传入数组行序为 y 降序，内部先转置再算，输出坐标是
        ``(列, 行)``；``'col-major'`` 则不转置。与 pyqtgraph 的
        ``IsocurveItem.generatePath`` 行为逐字对应，替换它时坐标才一致。

    返回
    ----
    每条折线是 ``[(x, y), ...]``，坐标在 **格点索引空间**（像素 i 的中心在
    i+0.5），由调用方套 QTransform 转成米。
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.size == 0:
        return []
    level = float(level)
    if axis_order == 'row-major':
        values = values.T
    elif axis_order != 'col-major':
        raise ValueError(f'axis_order 只能是 row-major / col-major，'
                         f'收到 {axis_order!r}')

    if extend_to_edge:
        values = _pad_to_edge(values)

    finite = np.isfinite(values)
    below = values < level          # NaN -> False，与 pyqtgraph 同解
    nrows, ncols = below.shape
    if nrows < 2 or ncols < 2:
        return []

    # 逐格 4 角。位序按 pyqtgraph 的 ``vertIndex = i + 2*j``：
    #   i=0,j=0 -> bit 0 = [ :-1, :-1]     i=0,j=1 -> bit 2 = [ :-1, 1: ]
    #   i=1,j=0 -> bit 1 = [ 1:, :-1]     i=1,j=1 -> bit 3 = [ 1:, 1: ]
    # 写反会让sideTable 选错边——全有限矩阵都会与 pyqtgraph 不一致。
    v00 = below[:-1, :-1]
    v01 = below[:-1, 1:]
    v10 = below[1:, :-1]
    v11 = below[1:, 1:]
    f00 = finite[:-1, :-1]
    f01 = finite[:-1, 1:]
    f10 = finite[1:, :-1]
    f11 = finite[1:, 1:]
    # NaN 感知的核心：四角全有限才画。缺这一句 NaN 会被当成"高于 level"。
    cell_ok = f00 & f01 & f10 & f11

    index = (v00.astype(np.uint8)
             | (v10.astype(np.uint8) << 1)
             | (v01.astype(np.uint8) << 2)
             | (v11.astype(np.uint8) << 3))
    index = np.where(cell_ok, index, 0).astype(np.uint8)
    return _chain(_segments(values, index, level, extend_to_edge))


def _pad_to_edge(values: np.ndarray) -> np.ndarray:
    """把边缘值向外扩一格（复刻 pyqtgraph ``extendToEdge=True``）。

    四个角的取值照抄上游（``functions.py:2344``）。注意右上角取 ``[1, -1]``
    （行 1 同列）而不是 ``[0, -2]``（同行相邻列）—— 看着像上游手误，但**必须
    照抄**，否则扩展出来的角值不同、边缘格子插值结果就不一致（实测 x=0那条
    边的节点与上游无一重合）。
    """
    padded = np.empty((values.shape[0] + 2, values.shape[1] + 2), dtype=float)
    padded[1:-1, 1:-1] = values
    padded[0, 1:-1] = values[0]
    padded[-1, 1:-1] = values[-1]
    padded[1:-1, 0] = values[:, 0]
    padded[1:-1, -1] = values[:, -1]
    padded[0, 0] = padded[0, 1]
    padded[0, -1] = padded[1, -1]
    padded[-1, 0] = padded[-1, 1]
    padded[-1, -1] = padded[-1, -2]
    return padded


def _segments(values: np.ndarray, index: np.ndarray, level: float,
              extend_to_edge: bool) -> list:
    """逐格插值出线段，每段带一对用于首尾相接的 gridKey。"""
    segments: list[tuple[tuple[float, float], tuple, tuple]] = []
    rows, cols = np.nonzero(index)
    x_hi = values.shape[0] - 2
    y_hi = values.shape[1] - 2
    for r, c in zip(rows.tolist(), cols.tolist()):
        sides = _SIDE_TABLE[index[r, c]]
        for k in range(0, len(sides), 2):
            edges = sides[k:k + 2]
            pts = []
            for m in (0, 1):
                p1 = _EDGE_KEY[edges[m]][0]
                p2 = _EDGE_KEY[edges[m]][1]
                v1 = values[r + p1[0], c + p1[1]]
                v2 = values[r + p2[0], c + p2[1]]
                denom = v2 - v1
                f = 0.5 if denom == 0.0 else (level - v1) / denom
                fi = 1.0 - f
                x = p1[0] * fi + p2[0] * f + r + 0.5
                y = p1[1] * fi + p2[1] * f + c + 0.5
                if extend_to_edge:
                    x = min(x_hi, max(0, x - 1))
                    y = min(y_hi, max(0, y - 1))
                grid_key = (r + (1 if edges[m] == 2 else 0),
                            c + (1 if edges[m] == 3 else 0),
                            edges[m] % 2)
                pts.append(((x, y), grid_key))
            segments.append((pts[0], pts[1]))
    return segments


def _chain(segments):
    """把首尾相接的线段接成折线（复刻 pyqtgraph 的 connected=True 行为）。

    以「线段」为最小单位消费：同一位置的相邻线段通过 ``gridKey`` 首尾相接，
    从任一未消费线段出发向两端走到底。逐段标记 ``used`` 而非标记节点——
    同一个 gridKey 上可能挂着多条线段（分叉/鞍点），按节点标记会漏边。
    """
    incident: dict[tuple, list[int]] = {}
    for idx, ((pa, ka), (pb, kb)) in enumerate(segments):
        incident.setdefault(ka, []).append(idx)
        incident.setdefault(kb, []).append(idx)

    used = [False] * len(segments)
    lines: list[list[tuple[float, float]]] = []

    def walk(start_idx: int, start_key: tuple) -> list:
        """从 start_idx 出发、沿 start_key 这一端向远端走，返回点序列。"""
        pts = []
        idx, key = start_idx, start_key
        while True:
            used[idx] = True
            (pa, ka), (pb, kb) = segments[idx]
            pts.append(pb if key == ka else pa)
            nxt_key = kb if key == ka else ka
            nxt = None
            for cand in incident.get(nxt_key, ()):
                if not used[cand]:
                    nxt = cand
                    break
            if nxt is None:
                return pts
            idx, key = nxt, nxt_key

    for start_idx in range(len(segments)):
        if used[start_idx]:
            continue
        (pa, ka), (pb, kb) = segments[start_idx]
        # 先从 ka 端向 ka 方向走，再把 kb 端反向走的结果接在前面
        head = walk(start_idx, ka)
        tail = walk(start_idx, kb)
        if tail:
            head = list(reversed(tail)) + head
        if len(head) >= 2:
            lines.append(head)
    return lines
