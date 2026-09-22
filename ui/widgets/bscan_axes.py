# -*- coding: utf-8 -*-
"""B-Scan 物理单位轴：把「道/采样」索引坐标换算成物理量后显示刻度。

BScanView 的显示坐标始终是**索引**（x = 列号、y = 行号），pick / overlay /
strided 降采样换算全部建立在这个约定上。若改用 ``ImageItem.setRect`` 把坐标
系换成物理量，三处换算都要跟着改。本模块改为在**显示层**（AxisItem 的刻度
文本）做单位换算，索引坐标系一动不动，功能上足够（用户要的是「读数与刻度
是物理量」），代价最小。

两组可切换单位：

- 横轴：道 ↔ 距离（m），取 PreviewBundle.trace_axis_m；
- 纵轴：采样轴（时间 ns / 深度 m，随 bundle）↔ 海拔（m）。

海拔 = **逐道地面高程 − 该道下方深度**。地形起伏时「某一行对应什么海拔」
逐道不同，而纵轴刻度是一维的，只能以某个参考高程为基准（见
:func:`reference_elevation`）；精确值由十字光标按当前道实时给出（见
:func:`elevation_at`）。这不是近似糊弄：替代方案是把整幅剖面重采样到统一
海拔网格，既改变了像元值（等于悄悄改数据），又要在预览路径上加一次全矩阵
插值，代价远大于收益——真要做到「地面拉平」应做成独立的时深/高程校正处理
方法，而不是预览显示的后处理。
"""

from __future__ import annotations

import math
from typing import Any, Callable

import pyqtgraph as pg

__all__ = [
    'ELEVATION_UNAVAILABLE_HINT',
    'IndexAxis',
    'X_AXIS_MODES',
    'Y_AXIS_MODES',
    'distance_available',
    'distance_tick_strings',
    'elevation_at',
    'elevation_available',
    'elevation_tick_strings',
    'reference_elevation',
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
    """由 bundle 的采样轴标签得到短单位名（按钮文字用）。

    ``'时间 (ns)'`` → ``'时间'``；``'深度 (m)'`` → ``'深度'``；
    无物理轴（直接 set_matrix）时回落到 ``'采样点'``。
    """
    label = str(sample_axis_label or '')
    for unit in ('时间', '深度'):
        if unit in label:
            return unit
    return '采样点'


def reference_elevation(series: Any) -> float | None:
    """纵轴刻度的参考地面高程：有限值的算术平均；无有效值返回 None。"""
    if series is None:
        return None
    try:
        raw_values = list(series)
    except TypeError:
        return None
    values = []
    for raw in raw_values:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def elevation_at(ground_elevation_m: Any, depth_axis_m: Any,
                 trace: int, sample: int) -> float | None:
    """指定道/采样处的海拔（地面高程 − 深度）；任一输入缺失返回 None。"""
    if ground_elevation_m is None or depth_axis_m is None:
        return None
    trace_count = len(ground_elevation_m)
    sample_count = len(depth_axis_m)
    if not (0 <= trace < trace_count and 0 <= sample < sample_count):
        return None
    ground = float(ground_elevation_m[trace])
    depth = float(depth_axis_m[sample])
    if not (math.isfinite(ground) and math.isfinite(depth)):
        return None
    return ground - depth


def distance_available(trace_axis_m: Any, trace_count: int) -> bool:
    """横轴能否切成距离：需要与显示矩阵等长的逐道里程轴。"""
    if trace_axis_m is None or trace_count <= 0:
        return False
    return len(trace_axis_m) >= int(trace_count)


def elevation_available(ground_elevation_m: Any, depth_axis_m: Any,
                        trace_count: int, sample_count: int) -> bool:
    """纵轴能否切成海拔：逐道地面高程与深度轴必须同时齐备且长度匹配。"""
    if ground_elevation_m is None or depth_axis_m is None:
        return False
    return (len(ground_elevation_m) >= max(int(trace_count), 1)
            and len(depth_axis_m) >= max(int(sample_count), 1))


def distance_tick_strings(values, trace_axis_m: Any) -> list[str] | None:
    """把索引刻度换算成里程（m）；无里程轴返回 None（回落默认刻度）。"""
    if trace_axis_m is None or not len(trace_axis_m):
        return None
    return [f'{float(trace_axis_m[tick_index(value, len(trace_axis_m))]):.4g}'
            for value in values]


def elevation_tick_strings(values, ground_elevation_m: Any,
                           depth_axis_m: Any) -> list[str] | None:
    """把索引刻度换算成海拔（m）；基准不足返回 None（回落默认刻度）。"""
    base = reference_elevation(ground_elevation_m)
    if base is None or depth_axis_m is None or not len(depth_axis_m):
        return None
    count = len(depth_axis_m)
    return [f'{base - float(depth_axis_m[tick_index(value, count)]):.4g}'
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
