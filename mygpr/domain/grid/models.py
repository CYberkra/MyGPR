#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure domain models for track grouping and attribute gridding."""
from __future__ import annotations

from dataclasses import dataclass

# 测线分组证据 schema 版本（登记 config/schema_catalog.json）。
# 结构变更必须升版本号，不得原地修改字段语义。
LINE_GROUPS_SCHEMA = "mygpr.line_groups.v1"


@dataclass(frozen=True, slots=True)
class LineGroup:
    """一条轨迹聚类分组：代表点相邻（≤ tolerance_m）的测线集合。

    representative 点是成员轨迹代表点的算术平均（米制投影坐标），
    成员间最大间距 max_pair_distance_m 供 UI 展示与审计。
    """

    group_id: str
    line_ids: tuple[str, ...]
    representative_x_m: float
    representative_y_m: float
    track_count: int
    max_pair_distance_m: float


@dataclass(frozen=True, slots=True)
class TrackGrouping:
    """一次轨迹分组的完整输出（确定性：组间按代表点字典序排序）。"""

    tolerance_m: float
    groups: tuple[LineGroup, ...]
    ungrouped_line_ids: tuple[str, ...]

    def group_of(self, line_id: str) -> LineGroup | None:
        for group in self.groups:
            if line_id in group.line_ids:
                return group
        return None


@dataclass(frozen=True, slots=True)
class AttributeGridRequest:
    """网格化请求：属性点 + 网格几何（application 层负责采集属性点）。

    value_missing_ok 控制含 NaN 属性点的处理：True 时剔除后继续，
    False 时直接报错。
    """

    x_m: tuple[float, ...]
    y_m: tuple[float, ...]
    values: tuple[float, ...]
    attribute_name: str
    cell_size_m: float = 1.0
    value_missing_ok: bool = True

    def __post_init__(self) -> None:
        if len(self.x_m) != len(self.y_m) or len(self.x_m) != len(self.values):
            raise ValueError("x_m/y_m/values 长度必须一致")
        if not self.x_m:
            raise ValueError("至少需要一个属性点")
        cell = float(self.cell_size_m)
        if not cell > 0.0:
            raise ValueError("cell_size_m 必须为正数")
        name = str(self.attribute_name).strip()
        if not name:
            raise ValueError("attribute_name 不能为空")


@dataclass(frozen=True, slots=True)
class AttributeGrid:
    """规则网格化结果（cell 中心坐标，行序 = y 降序，列序 = x 升序）。

    mask[i, j] = False 表示该 cell 无有效样本。
    """

    attribute_name: str
    x_origin_m: float
    y_origin_m: float
    cell_size_m: float
    ncols: int
    nrows: int
    values: tuple[tuple[float | None, ...], ...]

    @property
    def valid_count(self) -> int:
        return sum(1 for row in self.values for v in row if v is not None)


__all__ = [
    "LINE_GROUPS_SCHEMA",
    "AttributeGrid",
    "AttributeGridRequest",
    "LineGroup",
    "TrackGrouping",
]
