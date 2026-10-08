# -*- coding: utf-8 -*-
"""文件树节点装配（纯函数，不依赖 Qt）。

测线视图**平铺**列出（line_id 升序）：旧版曾按 ``updated_at`` 日期分组，
但组键是最后修改时间而非采集时间（``ProjectLine`` 无采集日期字段），
同批测线只要有一条被重新处理就被拆散，且与行业惯例（EKKO_Project
Project Explorer 平铺列线、无日期分组）相悖，分组层级已移除。

二三期若引入 ``ProjectLine.group_id``（测区/批次落库），在
:func:`build_tree_model` 里按语义键分组即可，树的构建与接线全部不动。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence


# ---------------------------------------------------------------- 节点模型（Provider 层）
# 文件树面板只渲染 TreeNode、发信号；节点由本模块纯函数装配。
# build_tree_model（测线）/ build_artifacts_model（成果）/
# build_files_model（文件，单层懒加载），经 build_project_model
# 装配为单棵三分类树（测线/成果/文件 一级常驻）。


@dataclass(frozen=True)
class TreeNode:
    """文件树节点（纯数据）。

    kind:
    - ``group``    分组行（不可选，仅展开）
    - ``line``     测线叶子，payload = line_id
    - ``artifact`` 处理成果叶子，payload = artifact_id，aux = line_id
    - ``spatial``  空间成果叶子，payload = result_id
    - ``report``   项目报告叶子，payload = package_dir
    - ``dir``      目录（文件视图），payload = 绝对路径，children 为空表示待懒加载
    - ``file``     文件（文件视图），payload = 绝对路径，suffix = 大小
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
    """测线视图装配器：平铺列出全部测线，line_id 升序（与日期无关）。

    空间成果/项目报告已移入「成果」视图（:func:`build_artifacts_model`），
    本函数保留 spatial/reports 形参仅为旧调用签名兼容，传入即忽略。
    """
    ordered = sorted(
        (ln for ln in lines or ()),
        key=lambda ln: str(getattr(ln, 'line_id', '') or ''))
    return [_line_node(ln) for ln in ordered]


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


# ---------------------------------------------------------------- 文件视图
# 项目根目录只读浏览（os.scandir 单层扫描，面板在展开目录时逐层懒加载）。

# 项目根下的内部实现条目（任何深度同名都隐藏）
HIDDEN_ENTRY_NAMES = frozenset({
    'cache', 'metadata', '.transactions', '.trash',
    'catalog.sqlite', 'catalog.sqlite-wal', 'catalog.sqlite-shm',
})


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if value < 1024 or unit == 'GB':
            return f'{value:.0f} {unit}' if unit == 'B' else f'{value:.1f} {unit}'
        value /= 1024
    return f'{size} B'


def build_files_model(dir_path: Any) -> list[TreeNode]:
    """单层目录扫描 → 节点列表（目录在前，各自按名称排序；忽略内部条目）。

    - ``dir`` 节点 payload = 目录绝对路径，无 children（面板展开时懒加载）；
    - ``file`` 节点 payload = 文件绝对路径，suffix = 人类可读大小；
    - 无权限/路径不存在返回空列表。
    """
    path = str(dir_path or '')
    if not path:
        return []
    try:
        entries = [e for e in os.scandir(path)
                   if e.name not in HIDDEN_ENTRY_NAMES]
    except OSError:
        return []
    dirs: list[TreeNode] = []
    files: list[TreeNode] = []
    for entry in entries:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if is_dir:
            dirs.append(TreeNode(
                key=f'dir:{entry.path}', kind='dir', text=entry.name,
                payload=entry.path, tooltip=entry.path))
        else:
            files.append(TreeNode(
                key=f'file:{entry.path}', kind='file', text=entry.name,
                payload=entry.path, suffix=_human_size(stat.st_size),
                tooltip=entry.path))
    by_name = lambda n: n.text.lower()  # noqa: E731
    return sorted(dirs, key=by_name) + sorted(files, key=by_name)


# ---------------------------------------------------------------- 项目总树
def build_project_model(lines: Sequence[Any], artifacts: Sequence[Any],
                        spatial_results: Sequence[Any] = (),
                        reports: Sequence[Any] = (),
                        files_root: Any = None) -> list[TreeNode]:
    """单棵三分类树（真树形）：测线 / 成果 / 文件 作为一级节点常驻。

    取代旧「顶部分段切换三视图」结构。分类行 ``kind='category'``，
    ``suffix`` = 该类条目数——空分类折叠后 ``(0)`` 计数自明，不再需要
    「尚无成果」之类的空态文案；有内容的分类由面板默认展开。
    文件分类仅在给出 ``files_root``（项目根）时参与计数与子节点
    （单层扫描，子目录沿用面板懒加载）。
    """
    file_nodes = build_files_model(files_root) if files_root else []
    categories = (
        ('测线', len(lines), build_tree_model(lines)),
        ('成果', len(artifacts) + len(spatial_results) + len(reports),
         build_artifacts_model(artifacts, spatial_results, reports)),
        ('文件', len(file_nodes), file_nodes),
    )
    return [
        TreeNode(key=f'category:{name}', kind='category', text=name,
                 suffix=str(count), children=tuple(children))
        for name, count, children in categories
    ]
