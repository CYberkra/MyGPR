# -*- coding: utf-8 -*-
"""测线树分组逻辑（纯函数，不依赖 Qt）。

一期形态：从 ``ProjectLine`` 现有字段派生分组键（采集日期，取
``updated_at`` 的 ISO 日期前缀），**不落库、不改 schema**。任何一行解析
不出日期就归"未分组"；若整个项目都归"未分组"则退化为平铺（不产生分组
层级），避免为一个无意义的单组节点多一次展开。

二三期若引入 ``ProjectLine.group_id``（测区/批次落库），只需替换本模块
的键派生函数，树的构建与接线全部不动。
"""
from __future__ import annotations

import re
from typing import Any, Sequence

_DATE_PREFIX = re.compile(r'^(\d{4}-\d{2}-\d{2})')
_UNGROUPED = '未分组'


def _date_key(updated_at: Any) -> str:
    """``updated_at`` → 'YYYY-MM-DD'；解析失败返回空串。"""
    text = str(updated_at or '').strip()
    match = _DATE_PREFIX.match(text)
    return match.group(1) if match else ''


def group_lines(lines: Sequence[Any]) -> list[tuple[str, list]]:
    """按采集日期分组，返回 ``[(组键, 测线列表), ...]``。

    - 组键空串 ``''`` 表示"整体平铺，不要分组层级"；
    - 组按日期倒序（最新在前），组内按 ``line_id`` 升序；
    - 只有部分测线有日期时，无日期的归"未分组"组，同样参与排序。
    """
    dated: dict[str, list] = {}
    fallback: list = []
    for line in lines or []:
        key = _date_key(getattr(line, 'updated_at', ''))
        if key:
            dated.setdefault(key, []).append(line)
        else:
            fallback.append(line)

    def _sort_lines(bucket: list) -> list:
        return sorted(
            bucket,
            key=lambda ln: str(getattr(ln, 'line_id', '') or ''),
        )

    if not dated:
        # 全部无日期：平铺，不产生分组节点
        return [('', _sort_lines(fallback))]

    groups = [(key, _sort_lines(bucket)) for key, bucket in dated.items()]
    if fallback:
        groups.append((_UNGROUPED, _sort_lines(fallback)))
    # 日期倒序；"未分组"无真实日期，按字典序排到末尾
    groups.sort(key=lambda kv: (kv[0] == _UNGROUPED, kv[0]), reverse=True)
    return groups


def group_stats(lines: Sequence[Any]) -> str:
    """分组节点副标题：条数 + 总长度（米，取整）。"""
    lines = list(lines or [])
    total_m = sum(float(getattr(ln, 'length_m', 0.0) or 0.0) for ln in lines)
    if total_m > 0:
        return f'{len(lines)} 条 · {total_m:.0f} m'
    return f'{len(lines)} 条'
