"""Shared preflight and execution contract for permanent resource deletion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from mygpr.application.project.errors import ProjectApplicationError


@dataclass(frozen=True)
class DeletionPlan:
    kind: str
    project_id: str
    root: str
    line_id: str
    targets: tuple[str, ...]
    names: tuple[str, ...]
    fingerprint: str
    parents: tuple[str, ...] = ()


class PermanentDeletionService:
    def __init__(self, projects):
        self.projects = projects

    def preview(self, kind, project_id, targets=(), *, line_id="", root=""):
        temporary = False
        if kind == 'project' and root:
            existing = {p.project_id for p in self.projects.list_open_projects()}
            summary = self.projects.open_project(root)
            project_id = summary.project_id
            temporary = project_id not in existing
        try:
            with self.projects.exclusive_delete(project_id) as session:
                return self._plan(session, kind, tuple(dict.fromkeys(targets)), line_id)
        finally:
            if temporary:
                self.projects.close_project(project_id)

    def _plan(self, session, kind, targets, line_id):
        summary = session.summary
        parents = ()
        names: tuple[str, ...]
        records: tuple[Any, ...]
        if kind == 'project':
            names = (summary.name, f'{summary.line_count} 条测线 · MyGPR 产物与导入副本')
            records = (summary, tuple(session.list_lines()), tuple(session.list_artifacts(None)))
            targets = (summary.project_id,)
        elif kind == 'lines':
            by_id = {line.line_id: line for line in session.list_lines()}
            if not targets or any(t not in by_id for t in targets):
                raise ProjectApplicationError('测线已变化，请重新选择')
            names = tuple(f'{t} · {by_id[t].name}' for t in targets)
            records = tuple((by_id[t], tuple(session.list_artifacts(t))) for t in targets)
        elif kind == 'artifacts':
            targets, names, records, parents = self._plan_artifacts(session, targets, line_id)
            document = session.read_markers(line_id)
            affected = tuple(item for item in document.records if any(
                anchor and anchor.source_id in {'artifact:' + uid for uid in targets}
                for anchor in (getattr(item, 'anchor', None), getattr(item, 'start', None), getattr(item, 'end', None))))
            if affected:
                names += (f'{len(affected)} 条标记或区间引用这些成果；保留记录及来源，位置将无法定位',)
            records += (document,)
        elif kind == 'spatial':
            by_id = {r.result_id: r for r in session.list_spatial_results()}
            if len(targets) != 1 or targets[0] not in by_id:
                raise ProjectApplicationError('空间成果已变化，请重新选择')
            record = by_id[targets[0]]
            names = (f'{record.name} · {record.result_id}',)
            current = session.current_spatial_result_id()
            if current == record.result_id:
                remaining = [r.result_id for r in by_id.values() if r.result_id != current]
                names += (f'当前版本将切换为：{remaining[0] if remaining else "无"}',)
            records = (record, current)
        else:
            raise ProjectApplicationError('未知删除类型')
        disk = session.validate_deletion(kind, targets, line_id)
        return DeletionPlan(kind, summary.project_id, summary.root_path, line_id,
                            targets, names, repr((records, disk)), parents)

    def _plan_artifacts(self, session, targets, line_id):
        by_id = {a.artifact_id: a for a in session.list_artifacts(line_id)}
        closure = dict.fromkeys(aid for t in targets for aid in session.list_artifact_descendants(line_id, t))
        if not targets or any(t not in by_id for t in targets):
            raise ProjectApplicationError('成果已变化，请重新选择')
        targets = tuple(closure)
        names = tuple(by_id[t].method_name or by_id[t].name or t for t in targets)

        def parent_of(artifact):
            params = artifact.manifest.get('params') or {}
            return str(params.get('step_input_artifact_id') or '') if params.get('lineage_version') == 2 else str(artifact.parent_artifact_id or '')

        parents = tuple(parent_of(by_id[t]) for t in targets)
        edges = dict(zip(targets, parents))
        for target in targets:
            seen, node = set(), target
            while node in edges:
                if node in seen:
                    raise ProjectApplicationError('处理成果依赖存在循环，无法删除')
                seen.add(node)
                node = edges[node]
        return targets, names, tuple(by_id[t] for t in targets), parents

    def execute(self, plan: DeletionPlan):
        if plan.kind == 'project':
            def validate(session):
                if self._plan(session, plan.kind, plan.targets, plan.line_id) != plan:
                    raise ProjectApplicationError('删除范围已变化，请重新确认')
            return self.projects.delete_project_permanently(plan.root, expected_project_id=plan.project_id, validate=validate)
        completed = []
        with self.projects.exclusive_delete(plan.project_id) as session:
            current = self._plan(session, plan.kind, plan.targets, plan.line_id)
            if current != plan:
                raise ProjectApplicationError('删除范围已变化，请重新确认')
            try:
                if plan.kind == 'artifacts':
                    return session.delete_artifacts(plan.line_id, plan.targets, reason='用户永久删除成果')
                for target in plan.targets:
                    if plan.kind == 'lines':
                        session.delete_line(target, reason='用户永久删除测线')
                    else:
                        session.delete_spatial_result(target)
                    completed.append(target)
            except Exception as exc:
                raise ProjectApplicationError(
                    f'已删除 {len(completed)} 项，当前项未完成，'
                    f'{len(plan.targets)-len(completed)-1} 项未执行。{exc}'
                ) from exc
        return tuple(completed)
