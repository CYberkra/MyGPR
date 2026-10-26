"""Truthful result titles and executed history; no matrix or filesystem access."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re

from ui.desktop_backend_facade import METHOD_METADATA


def artifact_parameters(artifact):
    manifest = getattr(artifact, "manifest", None) or {}
    if not isinstance(manifest, Mapping):
        return {}
    params = manifest.get("params")
    return dict(params if isinstance(params, Mapping) else manifest)


def method_name(method_id):
    return str(METHOD_METADATA.get(method_id, {}).get("display_name") or method_id)


@dataclass(frozen=True)
class ResultPresentation:
    title: str
    tooltip: str
    path: str
    complete: bool
    steps: tuple
    entries: tuple = ()


def _step_source(item, by_id, meta, lineage, index):
    if index == len(lineage):
        return str(item.artifact_id)
    run = meta.get("run_group_id")
    if not run:
        return ""
    matches = []
    for aid, other in by_id.items():
        data = artifact_parameters(other)
        if (
            getattr(other, "line_id", "") == getattr(item, "line_id", "")
            and data.get("run_group_id") == run
            and data.get("lineage_version") == 2
            and data.get("run_step_index") == index
            and data.get("lineage") == lineage[:index]
        ):
            matches.append(aid)
    return matches[0] if len(matches) == 1 else ""


def _executed(artifact, by_id):
    line_id = str(getattr(artifact, "line_id", ""))
    chunks, seen, complete = [], set(), True
    item = artifact
    while item is not None:
        aid = str(getattr(item, "artifact_id", ""))
        if aid in seen or len(seen) >= 4096 or str(getattr(item, "line_id", "")) != line_id:
            complete = False
            break
        seen.add(aid)
        meta = artifact_parameters(item)
        lineage = meta.get("lineage")
        if (
            meta.get("lineage_version") != 2
            or not isinstance(lineage, (list, tuple))
            or not lineage
            or any(
                not isinstance(s, Mapping) or not isinstance(s.get("method_id"), str) or not s["method_id"]
                for s in lineage
            )
        ):
            complete = False
            break
        try:
            count = int(meta.get("run_step_index", 0))
        except (ValueError, TypeError):
            count = 0
        if count != len(lineage) or "run_input_artifact_id" not in meta:
            complete = False
            break
        chunks.append(
            [
                dict(step, _artifact_id=_step_source(item, by_id, meta, lineage, index))
                for index, step in enumerate(lineage, 1)
            ]
        )
        parent = str(meta.get("run_input_artifact_id") or "")
        if not parent:
            break
        item = by_id.get(parent)
        if item is None:
            complete = False
    return tuple(step for chunk in reversed(chunks) for step in chunk), complete


def describe_artifact(artifact, artifacts=()):
    by_id = artifacts if isinstance(artifacts, Mapping) else {str(getattr(a, "artifact_id", "")): a for a in artifacts}
    steps, complete = _executed(artifact, by_id)
    names = [method_name(step["method_id"]) for step in steps]
    line_id = str(getattr(artifact, "line_id", ""))
    meta = artifact_parameters(artifact)
    name = str(getattr(artifact, "name", "") or "").strip()
    generic = not name or bool(re.fullmatch(r"处理结果(?:[_\s-]+.*)?", name)) or name.startswith("Processing pipeline")
    custom = str(meta["result_label"] or "") if "result_label" in meta else ("" if generic else name)
    count_label = "累计" if complete else "已知"
    summary = " → ".join(names) if len(names) <= 3 else f"{names[-1]}（{count_label}{len(names)}步）"
    if not complete:
        summary = (summary + " · " if summary else "") + "处理记录不完整"
    # Preserve explicit historical labels; incomplete origin is stated in details.
    title = custom if custom and not steps else f"{line_id} · " + (custom + " · " if custom else "") + summary
    path = ("原始数据" if complete else "处理记录不完整") + (" → " + " → ".join(names) if names else "")
    current_count = len(meta.get("lineage") or []) if isinstance(meta.get("lineage"), (list, tuple)) else 0
    tip = [
        f"测线：{line_id}",
        f"处理流程：{path}",
        f"本次新增：{current_count}步",
        f"生成时间：{getattr(artifact, 'created_at', '')}",
        f"结果号：{getattr(artifact, 'artifact_id', '')}",
    ]
    for index, step in enumerate(steps, 1):
        tip.append(
            f"{index}. {method_name(step['method_id'])}  "
            + json.dumps(step.get("params") or {}, ensure_ascii=False, default=str)
        )
    entries = [("original" if complete else "", "原始输入" if complete else "处理记录不完整", "")]
    for step in steps:
        aid = step.get("_artifact_id", "")
        entries.append(
            (
                "artifact:" + aid if aid else "",
                method_name(step["method_id"]),
                str(getattr(by_id.get(aid), "sha256", "") or ""),
            )
        )
    if not steps:
        entries.append(("artifact:" + str(artifact.artifact_id), title, str(getattr(artifact, "sha256", "") or "")))
    return ResultPresentation(title, "\n".join(tip), path, complete, steps, tuple(entries))
