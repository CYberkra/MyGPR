# -*- coding: utf-8 -*-
"""B-Scan 物理单位轴 + 海拔剖面重采样（共享海拔网格）。

两组可切换单位：

- 横轴：道 ↔ 距离（m），取 PreviewBundle.trace_axis_m；
- 纵轴：采样轴（时间 ns / 深度 m，随 bundle）↔ 海拔（m）。

海拔模式 = **整幅剖面重采样到统一海拔网格**（用户选型）：第 i 道第 j 个
采样点画在海拔 ``ground_elevation_m[i] - depth_axis_m[j]`` 处，逐道用
**各自的地面高程**。所有道共享同一条绝对海拔轴（:func:`build_elevation_view`
返回的 ``elev_axis``），因此地表以上自然留白（NaN → pyqtgraph 渲染为
透明，已实测 0.14.0），顶边呈现真实地貌起伏——这正是海拔模式的语义：
不是换刻度文字，而是图像本身落进海拔坐标系。

工程约束：

- **显示层变换，原始数据零变异**——warp 只发生在 BScanView 拿到预览矩阵
  之后，raw 矩阵/存储一字节不动；
- 性能实测（output/probes_archive/_probe_elevation_resample.py）：预览上限
  900×1800 约 50ms、全尺寸 1200×3800 约 70ms，为换轴/换线时的一次性成本，
  且 BScanView 按「同一份数据只算一次」缓存，不进渲染热路径；
- 退化策略：任一道地面高程非有限 → 该道整列 NaN（无数据，不猜值）；
  深度轴非有限/非递增、或逐道高程全部缺失 → 返回 (None, None)，调用方
  回落采样轴显示。

索引显示约定（pick / overlay / 降采样换算的基础）：

- 采样/距离模式下显示坐标 = 数据索引，与之前一致；
- 海拔模式下纵轴显示坐标 = **海拔网格行号**（不再是采样点号），行 r 的
  海拔 = ``elev_axis[r]``。BScanView 负责行号 ↔ 采样点的双向换算，
  AxisItem 刻度由 :func:`elevation_tick_strings` 直接查海拔轴。
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pyqtgraph as pg

__all__ = [
    'ELEVATION_UNAVAILABLE_HINT',
    'IndexAxis',
    'X_AXIS_MODES',
    'Y_AXIS_MODES',
    'build_elevation_view',
    'distance_available',
    'distance_tick_strings',
    'elevation_available',
    'elevation_tick_strings',
    'sample_unit_label',
    'tick_index',
]

# 横轴/纵轴可用单位（持久化键取值必须落在这两个元组里）
X_AXIS_MODES = ('trace', 'distance')
Y_AXIS_MODES = ('sample', 'elevation')

ELEVATION_UNAVAILABLE_HINT = (
    '本条数据缺少逐道地面高程，或缺少可换算为深度的介电常数，'
    '暂时无法按海拔显示纵轴（检查是否已导入/同步 RTK 导航数据）'
)

# 海拔网格行数上限：防止病态深度轴（极小步距）把内存吃爆。
# 8192 行 × 2000 道 × 4B = 64MB，已远超任何合理预览规模。
MAX_ELEVATION_ROWS = 8192


def tick_index(value: float, count: int) -> int:
    """轴刻度值（像素边界坐标）→ 数据索引，越界夹到 [0, count-1]。

    ImageItem 的第 i 列占据 [i, i+1)，所以 floor 后再夹取恰好是该列的采样
    /道号；pyqtgraph 传进来的刻度值可能是越界的负数或 n+ε（范围是
    [-0.5, n+0.5] 附近），故必须夹取。
    """
    if count <= 0:
        return 0
    index = int(math.floor(float(value)))
    return max(0, min(int(count) - 1, index))


def sample_unit_label(sample_axis_label: str) -> str:
    """由 bundle 的采样轴标签得到短单位名（右键菜单文字用）。

    ``'时间 (ns)'`` → ``'时间'``；``'深度 (m)'`` → ``'深度'``；
    无物理轴（直接 set_matrix）时回落到 ``'采样点'``。
    """
    label = str(sample_axis_label or '')
    for unit in ('时间', '深度'):
        if unit in label:
            return unit
    return '采样点'


def elevation_available(ground_elevation_m: Any, depth_axis_m: Any,
                        trace_count: int, sample_count: int) -> bool:
    """纵轴能否切成海拔：逐道地面高程与深度轴必须同时齐备且长度匹配。"""
    if ground_elevation_m is None or depth_axis_m is None:
        return False
    return (len(ground_elevation_m) >= max(int(trace_count), 1)
            and len(depth_axis_m) >= max(int(sample_count), 1))


def distance_available(trace_axis_m: Any, trace_count: int) -> bool:
    """横轴能否切成距离：需要与显示矩阵等长的逐道里程轴。"""
    if trace_axis_m is None or trace_count <= 0:
        return False
    return len(trace_axis_m) >= int(trace_count)


def build_elevation_view(matrix: Any, ground_elevation_m: Any,
                         depth_axis_m: Any,
                         *, max_rows: int = MAX_ELEVATION_ROWS):
    """把整幅剖面重采样到共享海拔网格。

    :param matrix: (n_samples, n_traces) 预览矩阵（显示网格，未变形）。
    :param ground_elevation_m: 逐道地面高程（m），长度 ≥ n_traces；
        允许个别道非有限（该道输出整列 NaN）。
    :param depth_axis_m: 地表以下深度轴（m，随显示矩阵、严格对应），
        要求有限；按中位差分视为等距。
    :return: ``(warped, elev_axis)``；warped 为 (rows, n_traces) float32，
        地表以上/最深处以下为 NaN；elev_axis 为降序海拔轴（行 r ↔
        elev_axis[r]）。输入不足或病态时返回 ``(None, None)``。

    网格行数 = (最高地面 − (最低地面 − 深度跨度)) / 深度步距 + 1，
    即行距恰为一个深度采样间隔——重采样不制造也不丢失纵向分辨率；
    地形起伏只把网格整体拉长（顶部留白行），夹在 max_rows 内。
    """
    if matrix is None or ground_elevation_m is None or depth_axis_m is None:
        return None, None
    mat = np.asarray(matrix)
    if mat.ndim != 2:
        return None, None
    n_samples, n_traces = mat.shape
    if n_samples < 2 or n_traces < 1:
        return None, None
    depth = np.asarray(depth_axis_m, dtype=np.float64).ravel()
    ground = np.asarray(ground_elevation_m, dtype=np.float64).ravel()
    if depth.size < n_samples or ground.size < n_traces:
        return None, None
    depth = depth[:n_samples]
    ground = ground[:n_traces]
    if not np.all(np.isfinite(depth)):
        return None, None
    step = float(np.median(np.diff(depth)))
    if not np.isfinite(step) or step <= 0:
        return None, None
    finite_ground = np.isfinite(ground)
    if not finite_ground.any():
        return None, None

    span = float(depth[-1] - depth[0])
    top = float(ground[finite_ground].max())
    bottom = float(ground[finite_ground].min()) - span
    rows = int(np.ceil((top - bottom) / step)) + 1
    rows = max(2, min(rows, int(max_rows)))
    elev_axis = top - step * np.arange(rows, dtype=np.float64)   # 降序

    lo, hi = float(depth[0]), float(depth[-1])
    inv_step = 1.0 / step
    # 一次性转置为 C 连续 (n_traces, n_samples)：np.interp 的 fp 参数
    # 连续时走快路径，比逐列切非连续列快数倍（实测见 probes_archive）。
    data64 = np.ascontiguousarray(mat.T, dtype=np.float64)
    warped = np.full((rows, n_traces), np.nan, dtype=np.float32)
    for i in range(n_traces):
        if not finite_ground[i]:
            continue                      # 该道无高程 → 整列无数据
        g = ground[i]
        # 有效行带：lo ≤ g − elev_axis[r] ≤ hi ⇔ 两条边界行号之间的行。
        # 带外（地表以上/最深以下）保持 NaN，无需逐列掩码。
        first = max(int(math.ceil((top - g + lo) * inv_step)), 0)
        last = min(int(math.floor((top - g + hi) * inv_step)), rows - 1)
        if first > last:
            continue
        target_depth = g - elev_axis[first:last + 1]   # 升序，且已落在 [lo, hi]
        warped[first:last + 1, i] = np.interp(target_depth, depth, data64[i])
    return warped, elev_axis


def distance_tick_strings(values, trace_axis_m: Any) -> list[str] | None:
    """把索引刻度换算成里程（m）；无里程轴返回 None（回落默认刻度）。"""
    if trace_axis_m is None or not len(trace_axis_m):
        return None
    return [f'{float(trace_axis_m[tick_index(value, len(trace_axis_m))]):.4g}'
            for value in values]


def elevation_tick_strings(values, elev_axis: Any) -> list[str] | None:
    """海拔模式纵轴刻度：行号直接查共享海拔轴；无轴返回 None（回落）。"""
    if elev_axis is None or not len(elev_axis):
        return None
    return [f'{float(elev_axis[tick_index(value, len(elev_axis))]):.4g}'
            for value in values]


class IndexAxis(pg.AxisItem):
    """索引坐标 → 物理单位 的刻度格式化轴（BScanView 横轴/纵轴共用）。

    ``converter`` 签名为 ``(values, scale, spacing) -> list[str] | None``：
    返回 ``None`` 表示「本轮不换算」，回落到 pyqtgraph 默认刻度文本
    （轴在该模式下依然是索引数，例如纵轴为时间/采样点时不需要换算）。

    另一个职责是**兜住 pyqtgraph 0.14 的空刻度崩溃**（已实测定位，非本
    项目的自定义轴引起——未接入 IndexAxis 的旧版 BScanView 同样会抛）：
    轴条短于约 10px 时 ``tickSpacing`` 仍返回两级（主刻度 50、次刻度同样
    50），``tickValues`` 去重后次刻度那一级的 values 变成空列表，而
    ``generateDrawSpecs`` 末尾的 ``min(map(min, tickPositions))`` 遇到全
    None 的空列表就抛 ``ValueError: min() iterable argument is empty``。

    复现路径（实测）：控件还没被布局、或刚从全屏窗口改嫁回原位的那一刻，
    纵轴轴条高度会是 **2px**，于是每次首帧都必然抛一次；正常尺寸（≥100px）
    下不会出现。这里在 ``tickValues`` 里丢掉空级，从根上不给出会让
    pyqtgraph 踩空的结构——比按尺寸打补丁稳，也不影响正常刻度。
    """

    def __init__(self, orientation: str,
                 converter: Callable[[Any, float, float], list[str] | None]):
        super().__init__(orientation)
        self._converter = converter

    def tickValues(self, minVal, maxVal, size):  # noqa: N802, N803 - pyqtgraph 命名
        """父类结果里剔除「空 values」的刻度级（见类 docstring 的崩溃成因）。"""
        levels = super().tickValues(minVal, maxVal, size)
        return [(spacing, values) for spacing, values in levels if values]

    def invalidate(self) -> None:
        """丢弃缓存的刻度画面并请求重绘（单位切换 / 数据更新后调用）。

        AxisItem 把刻度画进 ``QPicture`` 缓存，``update()`` 只会重播旧画面；
        必须先把 picture 置空才能拿到新刻度文本（pyqtgraph 内部换也是这么走的）。
        """
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):  # noqa: N802 - Qt/pyqtgraph 命名
        texts = self._converter(values, scale, spacing) if self._converter else None
        if texts is None:
            return super().tickStrings(values, scale, spacing)
        return texts
