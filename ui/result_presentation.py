"""Truthful result titles and executed history; no matrix or filesystem access."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re

from ui.desktop_backend_facade import METHOD_METADATA
from ui.processing_effects import describe_step


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
    tab_title: str = ""
    effect_summary: str = ""
    run_summary: str = ""
    warning_count: int = 0


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
    effects = [describe_step(step) for step in steps]
    line_id = str(getattr(artifact, "line_id", ""))
    meta = artifact_parameters(artifact)
    name = str(getattr(artifact, "name", "") or "").strip()
    generic = (not name or name == str(getattr(artifact, "artifact_id", ""))
               or bool(re.fullmatch(r"处理结果(?:[_\s-]+.*)?", name))
               or name.startswith("Processing pipeline"))
    custom = str(meta["result_label"] or "") if "result_label" in meta else ("" if generic else name)
    count_label = "累计" if complete else "已知"
    summary = " → ".join(names) if len(names) <= 3 else f"{names[-1]}（{count_label}{len(names)}步）"
    if not complete:
        summary = (summary + " · " if summary else "") + "处理记录不完整"
    title = custom if custom and not steps else f"{line_id} · " + (custom + " · " if custom else "") + summary
    if len(effects) <= 3:
        compact = " → ".join(effect.action for effect in effects)
    else:
        categories = tuple(dict.fromkeys(effect.category for effect in effects))
        compact = " + ".join(categories) + f"（{len(effects)}步）"
    if not complete:
        compact = (compact + " · " if compact else "") + "记录不完整"
    tab_title = f"{custom or line_id}｜{compact}" if compact else title

    path = ("原始数据" if complete else "处理记录不完整") + (" → " + " → ".join(names) if names else "")
    current_steps = meta.get("lineage") or []
    if not isinstance(current_steps, (list, tuple)):
        current_steps = []
    current_effects = [describe_step(step) for step in current_steps if isinstance(step, Mapping)]
    current_count = len(current_steps)
    effect_parts = []
    for effect in current_effects:
        text = effect.action + (f"({effect.parameter_summary})" if effect.parameter_summary else "")
        if effect.outcome != "已执行":
            text += f"[{effect.outcome}]"
        effect_parts.append(text)
    effect_summary = " → ".join(effect_parts) if effect_parts else "无可靠的本轮处理记录"
    warning_count = sum(len(effect.warnings) for effect in effects)
    run_summary = f"本次新增{current_count}步｜累计{len(steps)}步"
    if warning_count:
        run_summary += f"｜警告{warning_count}"
    tip = [
        f"测线：{line_id}",
        f"处理流程：{path}",
        f"本次新增：{current_count}步",
        f"生成时间：{getattr(artifact, 'created_at', '')}",
        f"结果号：{getattr(artifact, 'artifact_id', '')}",
    ]
    for index, (step, effect) in enumerate(zip(steps, effects), 1):
        tip.append(
            f"{index}. {method_name(step['method_id'])}  "
            + json.dumps(step.get("params") or {}, ensure_ascii=False, default=str)
            + f"｜{effect.outcome}"
        )
        for warning in effect.warnings:
            tip.append(f"   警告：{warning}")
    entries = [("original" if complete else "", "原始输入" if complete else "处理记录不完整", "")]
    for step, effect in zip(steps, effects):
        aid = step.get("_artifact_id", "")
        details = {
            'method': method_name(step['method_id']),
            'params': dict(step.get('params') or {}),
            'outcome': effect.outcome,
            'warnings': list(effect.warnings),
        }
        entries.append(
            (
                "artifact:" + aid if aid else "",
                effect.action,
                str(getattr(by_id.get(aid), "sha256", "") or ""),
                json.dumps(details, ensure_ascii=False, default=str),
            )
        )
    if not steps:
        entries.append(("artifact:" + str(artifact.artifact_id), title, str(getattr(artifact, "sha256", "") or "")))
    return ResultPresentation(
        title=title,
        tooltip="\n".join(tip),
        path=path,
        complete=complete,
        steps=steps,
        entries=tuple(entries),
        tab_title=tab_title,
        effect_summary=effect_summary,
        run_summary=run_summary,
        warning_count=warning_count,
    )
