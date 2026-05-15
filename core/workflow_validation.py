#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI-free validation helpers for MyGPR Workflow Studio."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from core.methods_registry import PROCESSING_METHODS


PREVIEW_NODE_ID = "__workflow_preview__"


@dataclass(frozen=True)
class WorkflowValidationIssue:
    """A single workflow validation finding."""

    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    node_id: str = ""
    link_key: str = ""


@dataclass
class WorkflowValidationReport:
    """Structured validation report for UI and tests."""

    issues: list[WorkflowValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[WorkflowValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[WorkflowValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def infos(self) -> list[WorkflowValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "info"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        node_id: str = "",
        link_key: str = "",
    ) -> None:
        self.issues.append(
            WorkflowValidationIssue(
                severity=severity,
                code=code,
                message=message,
                node_id=node_id,
                link_key=link_key,
            )
        )

    def summary(self) -> str:
        if not self.issues:
            return "Validate: OK"
        return (
            f"Validate: {len(self.errors)} errors, "
            f"{len(self.warnings)} warnings, {len(self.infos)} info"
        )

    def to_text(self) -> str:
        labels = {"error": "ERROR", "warning": "WARN", "info": "INFO"}
        lines = [self.summary()]

        if not self.issues:
            lines.append("No workflow graph issues detected.")
            return "\n".join(lines)

        for issue in self.issues:
            suffix = ""
            if issue.node_id:
                suffix += f" node={issue.node_id}"
            if issue.link_key:
                suffix += f" link={issue.link_key}"
            lines.append(
                f"[{labels.get(issue.severity, issue.severity.upper())}] "
                f"{issue.code}: {issue.message}{suffix}"
            )
        return "\n".join(lines)


def _get_suggestion(issue: WorkflowValidationIssue, sidecar_status: dict[str, bool] | None = None) -> str:
    """Generate user-friendly suggestion for a validation issue."""
    code = issue.code

    suggestions = {
        "missing_sidecar": "导入对应的 sidecar 文件，或在当前工作流中停用相关节点。",
        "graph_order_mismatch": "当前工作流仍按节点顺序执行。确认节点顺序后继续，或使用「自动布局」恢复线性链。",
        "isolated_node": "检查节点连线是否正确，或将其连接到工作流中。",
        "preview_without_input": "B-scan Preview 节点需要连接到最后一个处理节点。",
        "cycle_detected": "检测到循环连接，请移除循环边。",
        "duplicate_link": "移除重复的连线。",
        "empty_link_endpoint": "检查连线两端是否正确连接到节点。",
        "missing_from_node": "连线源节点不存在，删除此连线。",
        "missing_to_node": "连线目标节点不存在，删除此连线。",
        "self_link": "节点不能连接到自己，删除此连线。",
        "multiple_data_inputs": "每个节点只能有一个数据输入，移除多余的连线。",
        "unknown_link_kind": "忽略未知类型的连线，或将其删除。",
        "empty_workflow": "工作流为空，请添加处理节点。",
        "duplicate_or_missing_node_id": "工作流内部错误，请重新加载模板。",
    }

    base = suggestions.get(code, "请根据上述信息调整工作流。")

    if code == "missing_sidecar" and sidecar_status:
        req_map = {
            "RTK": ("rtk", "RTK"),
            "IMU": ("imu", "IMU"),
            "AGL": ("agl", "高度计"),
        }
        req_str = issue.message.replace("节点缺少所需的辅助数据：", "").replace("、", ",")
        for key, (status_key, label) in req_map.items():
            if key in req_str:
                if sidecar_status.get(status_key):
                    base = f"已加载 {label} 数据，忽略此警告。"
                else:
                    base = f"导入 {label} sidecar 文件，或停用需要 {label} 的节点。"
                break

    return base


def to_text_with_suggestions(
    report: WorkflowValidationReport,
    *,
    sidecar_status: dict[str, bool] | None = None,
    execution_mode: str = "order",
) -> str:
    """Generate validation report text with user-friendly suggestions."""
    labels = {"error": "ERROR", "warning": "WARN", "info": "INFO"}
    lines = [f"Validate: {len(report.errors)} errors, {len(report.warnings)} warnings, {len(report.infos)} info"]

    if execution_mode == "order":
        lines.append("执行模式：按节点顺序（画布连线仅供参考）")

    if not report.issues:
        lines.append("No workflow graph issues detected.")
        return "\n".join(lines)

    for issue in report.issues:
        suffix = ""
        if issue.node_id:
            suffix += f" node={issue.node_id}"
        if issue.link_key:
            suffix += f" link={issue.link_key}"

        suggestion = _get_suggestion(issue, sidecar_status)
        lines.append(
            f"[{labels.get(issue.severity, issue.severity.upper())}] "
            f"{issue.code}: {issue.message}{suffix}"
        )
        if issue.severity in ("error", "warning"):
            lines.append(f"  建议：{suggestion}")

    return "\n".join(lines)


def validate_workflow_config(
    config: Any,
    *,
    preview_node_id: str = PREVIEW_NODE_ID,
    execution_mode: str = "order",
    sidecar_status: dict[str, bool] | None = None,
) -> WorkflowValidationReport:
    """Validate a WorkflowConfig-like object.

    Use execution_mode="order" while the backend still executes by
    WorkflowMethod.order rather than graph topology.
    """

    methods = list(getattr(config, "methods", []) or [])
    links = list(getattr(config, "canvas_links", []) or [])
    report = WorkflowValidationReport()

    if not methods:
        report.add("warning", "empty_workflow", "Workflow has no method nodes.")
        return report

    node_by_id = {}
    duplicate_or_missing = False
    for method in methods:
        node_id = str(getattr(method, "node_id", ""))
        if not node_id or node_id in node_by_id:
            duplicate_or_missing = True
            continue
        node_by_id[node_id] = method

    method_node_ids = set(node_by_id)
    valid_node_ids = set(method_node_ids)
    valid_node_ids.add(preview_node_id)

    if duplicate_or_missing or len(node_by_id) != len(methods):
        report.add(
            "error",
            "duplicate_or_missing_node_id",
            "Some workflow methods have missing or duplicate node_id values.",
        )

    order_index = {
        str(getattr(method, "node_id", "")): int(getattr(method, "order", index))
        for index, method in enumerate(methods)
        if str(getattr(method, "node_id", ""))
    }

    valid_links = []
    seen_link_keys: set[str] = set()
    incoming: dict[str, list[Any]] = defaultdict(list)
    outgoing: dict[str, list[Any]] = defaultdict(list)
    data_input_count: dict[tuple[str, str], int] = defaultdict(int)

    for link in links:
        from_node = str(getattr(link, "from_node", ""))
        to_node = str(getattr(link, "to_node", ""))
        from_port = str(getattr(link, "from_port", "output"))
        to_port = str(getattr(link, "to_port", "input"))
        kind = str(getattr(link, "kind", "data"))
        key = link_key(link)

        if key in seen_link_keys:
            report.add("warning", "duplicate_link", "Duplicate canvas link detected.", link_key=key)
            continue
        seen_link_keys.add(key)

        if not from_node or not to_node:
            report.add("error", "empty_link_endpoint", "A canvas link has an empty endpoint.", link_key=key)
            continue

        if from_node not in valid_node_ids:
            report.add("error", "missing_from_node", "Link source node does not exist.", node_id=from_node, link_key=key)
            continue

        if to_node not in valid_node_ids:
            report.add("error", "missing_to_node", "Link target node does not exist.", node_id=to_node, link_key=key)
            continue

        if from_node == to_node:
            report.add("error", "self_link", "A node cannot connect to itself.", node_id=from_node, link_key=key)
            continue

        if from_port != "output" or to_port != "input":
            report.add("warning", "unsupported_port_pair", "Only output -> input links are currently supported.", link_key=key)
            continue

        if kind not in {"data", "metadata", "preview", "export"}:
            report.add("warning", "unknown_link_kind", f"Unknown link kind '{kind}'.", link_key=key)

        valid_links.append(link)
        outgoing[from_node].append(link)
        incoming[to_node].append(link)

        if kind == "data" and to_port == "input":
            data_input_count[(to_node, to_port)] += 1

    for (node_id, port), count in sorted(data_input_count.items()):
        if count > 1:
            report.add(
                "error",
                "multiple_data_inputs",
                f"Input port '{port}' has {count} data links; only one is supported.",
                node_id=node_id,
            )

    preview_inputs = [
        link for link in valid_links
        if str(getattr(link, "to_node", "")) == preview_node_id
    ]
    if not preview_inputs:
        report.add(
            "warning",
            "preview_without_input",
            "B-scan Preview node has no input link.",
            node_id=preview_node_id,
        )

    for method in methods:
        node_id = str(getattr(method, "node_id", ""))
        if not node_id or bool(getattr(method, "hidden", False)) or not bool(getattr(method, "enabled", True)):
            continue

        if not incoming.get(node_id) and not outgoing.get(node_id):
            report.add(
                "warning",
                "isolated_node",
                "Enabled visible node is isolated from the workflow graph.",
                node_id=node_id,
            )

    cycle = find_cycle(method_node_ids, valid_links)
    if cycle:
        report.add(
            "error",
            "cycle_detected",
            "Canvas links contain a cycle: " + " -> ".join(cycle),
            node_id=cycle[0],
        )

    for link in valid_links:
        from_node = str(getattr(link, "from_node", ""))
        to_node = str(getattr(link, "to_node", ""))

        if from_node in method_node_ids and to_node in method_node_ids:
            if order_index.get(from_node, -1) >= order_index.get(to_node, -1):
                report.add(
                    "warning",
                    "graph_order_mismatch",
                    (
                        "Canvas link points against the current execution order. "
                        "Execution may not match the visual graph."
                    ),
                    link_key=link_key(link),
                )

    if execution_mode == "order" and valid_links:
        report.add(
            "info",
            "execution_uses_order",
            (
                "Current workflow execution still follows method order. "
                "Canvas links are validated and displayed, but do not yet fully "
                "define execution order."
            ),
        )

    if sidecar_status is not None:
        rtk_loaded = sidecar_status.get("rtk", False)
        imu_loaded = sidecar_status.get("imu", False)
        agl_loaded = sidecar_status.get("agl", False)

        for method in methods:
            if method.hidden or not method.enabled:
                continue
            method_key = method.method_id
            if method_key not in PROCESSING_METHODS:
                continue
            meta = PROCESSING_METHODS[method_key]
            missing_requirements = []
            if meta.get("requires_rtk") and not rtk_loaded:
                missing_requirements.append("RTK")
            if meta.get("requires_imu") and not imu_loaded:
                missing_requirements.append("IMU")
            if meta.get("requires_agl") and not agl_loaded:
                missing_requirements.append("AGL")
            if missing_requirements:
                req_str = "、".join(missing_requirements)
                report.add(
                    "warning",
                    "missing_sidecar",
                    f"节点缺少所需的辅助数据：{req_str}",
                    node_id=method.node_id,
                )

    if not report.issues:
        report.add("info", "ok", "Workflow graph looks consistent.")

    return report


def link_key(link: Any) -> str:
    return "|".join(
        [
            str(getattr(link, "from_node", "")),
            str(getattr(link, "from_port", "output")),
            str(getattr(link, "to_node", "")),
            str(getattr(link, "to_port", "input")),
            str(getattr(link, "kind", "data")),
        ]
    )


def find_cycle(node_ids: Iterable[str], links: Iterable[Any]) -> list[str]:
    """Return one cycle path if present, otherwise an empty list."""

    node_set = set(node_ids)
    graph: dict[str, list[str]] = defaultdict(list)

    for link in links:
        from_node = str(getattr(link, "from_node", ""))
        to_node = str(getattr(link, "to_node", ""))
        kind = str(getattr(link, "kind", "data"))
        if kind not in {"data", "metadata"}:
            continue
        if from_node in node_set and to_node in node_set:
            graph[from_node].append(to_node)

    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> list[str]:
        if node in visiting:
            try:
                start = stack.index(node)
                return stack[start:] + [node]
            except ValueError:
                return [node, node]
        if node in visited:
            return []

        visiting.add(node)
        stack.append(node)
        for target in graph.get(node, []):
            found = visit(target)
            if found:
                return found
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return []

    for node in sorted(node_set):
        found = visit(node)
        if found:
            return found
    return []


def topological_method_order(config: Any) -> list[str]:
    """Return method node_ids in graph topological order, or [] for cycles."""

    methods = list(getattr(config, "methods", []) or [])
    links = list(getattr(config, "canvas_links", []) or [])
    method_ids = [
        str(getattr(method, "node_id", ""))
        for method in methods
        if str(getattr(method, "node_id", ""))
    ]
    method_set = set(method_ids)

    graph: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {node_id: 0 for node_id in method_ids}

    for link in links:
        from_node = str(getattr(link, "from_node", ""))
        to_node = str(getattr(link, "to_node", ""))
        kind = str(getattr(link, "kind", "data"))
        if kind != "data":
            continue
        if from_node not in method_set or to_node not in method_set:
            continue
        graph[from_node].append(to_node)
        indegree[to_node] += 1

    queue = deque([node_id for node_id in method_ids if indegree[node_id] == 0])
    ordered: list[str] = []

    while queue:
        node = queue.popleft()
        ordered.append(node)
        for target in graph.get(node, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if len(ordered) != len(method_ids):
        return []
    return ordered
