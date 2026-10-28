"""Contracts and saved-edge selection for portable processing result packages."""
from dataclasses import dataclass
from collections import Counter
from typing import Mapping, Sequence

from mygpr.domain.project.models import ProjectArtifact

RESULT_H5_SCHEMA = 'mygpr.processing_result_hdf5.v1'
PACKAGE_SCHEMA = 'mygpr.processing_result_package.v1'


@dataclass(frozen=True, slots=True)
class ProcessingResultPackage:
    archive_path: str
    line_id: str
    artifact_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    verified: bool


def parameters(artifact: ProjectArtifact) -> dict:
    value = artifact.manifest.get('params')
    return dict(value if isinstance(value, Mapping) else artifact.manifest)


def direct_parent(artifact: ProjectArtifact) -> str | None:
    params = parameters(artifact)
    if params.get('lineage_version') != 2 or 'step_input_artifact_id' not in params:
        return None
    return str(params['step_input_artifact_id'] or '')


def select_saved_steps(artifacts: Sequence[ProjectArtifact], target: str = ''):
    """Use only persisted v2 edges; unknown legacy lineage is never guessed."""
    by_id = {a.artifact_id: a for a in artifacts if a.status in ('committed', 'completed', 'success')}
    if target and target not in by_id:
        raise ValueError(f'所选成果不存在或未保存：{target}')
    if not by_id:
        raise ValueError('该测线没有已保存的处理结果。')
    selected = set(by_id) if not target else set()
    cursor = target
    while cursor:
        if cursor in selected:
            raise ValueError('处理来源存在循环，无法打包。')
        selected.add(cursor)
        parent = direct_parent(by_id[cursor])
        cursor = parent if parent in by_id else ''
    warnings = []
    # Iterative DFS keeps deep processing chains below Python's recursion limit.
    ordered, done, active = [], set(), set()
    for artifact in sorted((by_id[k] for k in selected), key=lambda a: (a.created_at, a.artifact_id)):
        trail, cursor = [], artifact.artifact_id
        while cursor and cursor not in done:
            if cursor in active:
                raise ValueError('处理来源存在循环，无法打包。')
            active.add(cursor)
            trail.append(cursor)
            parent = direct_parent(by_id[cursor])
            if parent is None:
                warnings.append(f'{cursor}：缺少可靠的直接来源，仅包含实际保存的结果。')
            elif parent and parent not in by_id:
                warnings.append(f'{cursor}：前序成果 {parent} 未保存或已移除。')
            cursor = parent if parent in selected else ''
        for key in reversed(trail):
            ordered.append(by_id[key])
            done.add(key)
            active.remove(key)
    saved_per_run = Counter(str(parameters(a).get('run_group_id') or '') for a in ordered)
    for artifact in ordered:
        params = parameters(artifact)
        run = params.get('run_group_id')
        index = int(params.get('run_step_index') or 0)
        saved = saved_per_run[str(run or '')]
        if run and index > saved:
            warnings.append(f'{artifact.artifact_id}：该运行的部分中间步骤没有保存；不会重新执行算法补齐。')
    return tuple(ordered), tuple(dict.fromkeys(warnings))
