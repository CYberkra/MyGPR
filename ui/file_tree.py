# -*- coding: utf-8 -*-
"""文件树分组逻辑（纯函数，不依赖 Qt）。

一期形态：从 ``ProjectLine`` 现有字段派生分组键（采集日期，取
``updated_at`` 的 ISO 日期前缀），**不落库、不改 schema**。任何一行解析
不出日期就归"未分组"；若整个项目都归"未分组"则退化为平铺（不产生分组
层级），避免为一个无意义的单组节点多一次展开。

二三期若引入 ``ProjectLine.group_id``（测区/批次落库），只需替换本模块
的键派生函数，树的构建与接线全部不动。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------- 节点模型（Provider 层）
# 文件树面板只渲染 TreeNode、发信号；节点由本模块纯函数装配。
# 「测线｜成果｜文件」分段视图 = 各视图一个装配器：
# build_tree_model（测线）/ build_artifacts_model（成果）/ 文件视图待阶段 3。


@dataclass(frozen=True)
class TreeNode:
    """文件树节点（纯数据）。

    kind:
    - ``group``    分组行（不可选，仅展开）
    - ``line``     测线叶子，payload = line_id
    - ``artifact`` 处理成果叶子，payload = artifact_id，aux = line_id
    - ``spatial``  空间成果叶子，payload = result_id
    - ``report``   项目报告叶子，payload = package_dir
    icon 为语义键（success/info/disabled，供状态点查表），仅 line 用。
    suffix 为行尾灰色角标（第二列右对齐小字）。
    """
    key: str
    kind: str
    text: str
    payload: str = ''
    suffix: str = ''
    icon: str = ''
    tooltip: str = ''
    aux: str = ''
    children: tuple = field(default_factory=tuple)


def line_suffix(line: Any) -> str:
    """测线行尾角标：成果✓ / 标N / 界面✓（全部派生自 ProjectLine 现有字段，
    不新增后端查询）。"""
    parts = []
    if str(getattr(line, 'processed_result', '') or ''):
        parts.append('成果✓')
    targets = int(getattr(line, 'target_count', 0) or 0)
    if targets > 0:
        parts.append(f'标{targets}')
    if int(getattr(line, 'interface_keypoint_count', 0) or 0) > 0:
        parts.append('界面✓')
    return ' '.join(parts)


def _line_tooltip(line: Any) -> str:
    status = str(getattr(line, 'processing_status', '') or '未处理')
    lines = [f'状态：{status}']
    length = float(getattr(line, 'length_m', 0.0) or 0.0)
    if length > 0:
        lines.append(f'长度：{length:.1f} m')
    updated = str(getattr(line, 'updated_at', '') or '')[:10]
    if updated:
        lines.append(f'更新：{updated}')
    return '\n'.join(lines)


def _line_node(line: Any) -> TreeNode:
    line_id = str(getattr(line, 'line_id', '') or '')
    name = str(getattr(line, 'name', '') or '')
    text = line_id if name in ('', line_id) else f'{line_id} · {name}'
    status = str(getattr(line, 'processing_status', '') or '未处理')
    return TreeNode(
        key=f'line:{line_id}', kind='line', text=text, payload=line_id,
        suffix=line_suffix(line), icon=status, tooltip=_line_tooltip(line))


def _spatial_node(result: Any) -> TreeNode:
    result_id = str(getattr(result, 'result_id', '') or '')
    name = str(getattr(result, 'name', '') or result_id)
    revision = int(getattr(result, 'revision', 0) or 0)
    created = str(getattr(result, 'created_at', '') or '')[:10]
    n_lines = len(getattr(result, 'line_ids', ()) or ())
    tip = [f'状态：{getattr(result, "status", "") or "—"}',
           f'测线：{n_lines} 条']
    if getattr(result, 'stale', False):
        tip.append('已有更新的数据（陈旧）')
    return TreeNode(
        key=f'spatial:{result_id}', kind='spatial',
        text=f'{name} v{revision}' if revision else name,
        payload=result_id, suffix=created, tooltip='\n'.join(tip))


def _report_node(package: Any) -> TreeNode:
    package_dir = str(getattr(package, 'package_dir', '') or '')
    name = os.path.basename(package_dir.rstrip('/\\')) or package_dir
    generated = str(getattr(package, 'generated_at', '') or '')[:10]
    count = int(getattr(package, 'file_count', 0) or 0)
    return TreeNode(
        key=f'report:{package_dir}', kind='report', text=name,
        payload=package_dir, suffix=generated,
        tooltip=f'文件：{count} 个\n目录：{package_dir}')


def build_tree_model(lines: Sequence[Any], spatial_results: Sequence[Any] = (),
                     reports: Sequence[Any] = ()) -> list[TreeNode]:
    """测线视图装配器：测线按日期分组（沿用 :func:`group_lines` 规则）。

    空间成果/项目报告已移入「成果」视图（:func:`build_artifacts_model`），
    本函数保留 spatial/reports 形参仅为旧调用签名兼容，传入即忽略。
    """
    nodes: list[TreeNode] = []
    for key, bucket in group_lines(lines):
        children = tuple(_line_node(ln) for ln in bucket)
        if not key:
            nodes.extend(children)
        else:
            nodes.append(TreeNode(
                key=f'group:{key}', kind='group', text=key,
                suffix=group_stats(bucket), children=children))
    return nodes


def _artifact_node(artifact: Any) -> TreeNode:
    artifact_id = str(getattr(artifact, 'artifact_id', '') or '')
    line_id = str(getattr(artifact, 'line_id', '') or '')
    name = str(getattr(artifact, 'name', '') or artifact_id)
    method = str(getattr(artifact, 'method_name', '') or
                 getattr(artifact, 'method_id', '') or '')
    created = str(getattr(artifact, 'created_at', '') or '')[:16].replace('T', ' ')
    shape = getattr(artifact, 'shape', ()) or ()
    tip = [f'测线：{line_id}']
    if method:
        tip.append(f'方法：{method}')
    if created:
        tip.append(f'创建：{created}')
    if shape:
        tip.append(f'形状：{"×".join(str(v) for v in shape)} '
                   f'{getattr(artifact, "dtype", "") or ""}'.strip())
    return TreeNode(
        key=f'artifact:{artifact_id}', kind='artifact', text=name,
        payload=artifact_id, aux=line_id, suffix=method,
        tooltip='\n'.join(tip))


def build_artifacts_model(artifacts: Sequence[Any],
                          spatial_results: Sequence[Any] = (),
                          reports: Sequence[Any] = ()) -> list[TreeNode]:
    """成果视图装配器：处理成果按测线分组平铺 + 空间成果 + 项目报告。

    - 处理成果：按 ``line_id`` 升序分组（组标题=测线号），组内按创建时间
      倒序（最新在前）；无成果/无空间/无报告的分组不产生节点；
    - 空间成果按创建时间倒序，报告按生成时间倒序。
    """
    nodes: list[TreeNode] = []
    by_line: dict[str, list] = {}
    for art in artifacts or []:
        by_line.setdefault(str(getattr(art, 'line_id', '') or ''), []).append(art)
    for line_id in sorted(by_line):
        bucket = sorted(
            by_line[line_id],
            key=lambda a: str(getattr(a, 'created_at', '') or ''), reverse=True)
        nodes.append(TreeNode(
            key=f'group:line-artifacts:{line_id}', kind='group', text=line_id,
            suffix=f'{len(bucket)} 项',
            children=tuple(_artifact_node(a) for a in bucket)))

    spatial = sorted(
        (r for r in spatial_results or []),
        key=lambda r: str(getattr(r, 'created_at', '') or ''), reverse=True)
    if spatial:
        nodes.append(TreeNode(
            key='group:spatial', kind='group', text='空间成果',
            suffix=f'{len(spatial)} 项',
            children=tuple(_spatial_node(r) for r in spatial)))

    pkgs = sorted(
        (p for p in reports or []),
        key=lambda p: str(getattr(p, 'generated_at', '') or ''), reverse=True)
    if pkgs:
        nodes.append(TreeNode(
            key='group:report', kind='group', text='项目报告',
            suffix=f'{len(pkgs)} 份',
            children=tuple(_report_node(p) for p in pkgs)))
    return nodes
