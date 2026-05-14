#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from core.workflow_data import WorkflowLink, build_default_workflow_config
from core.workflow_validation import (
    topological_method_order,
    validate_workflow_config,
)


def _codes(report):
    return {issue.code for issue in report.issues}


def test_validate_default_workflow_has_execution_mode_info():
    config = build_default_workflow_config("high_quality_uav_gpr")
    report = validate_workflow_config(config, execution_mode="order")

    assert "execution_uses_order" in _codes(report)
    assert not report.errors


def test_validate_reports_missing_link_endpoint():
    config = build_default_workflow_config("high_quality_uav_gpr")
    first = config.methods[0]
    config.canvas_links.append(WorkflowLink(first.node_id, "missing_node"))

    report = validate_workflow_config(config)

    assert "missing_to_node" in _codes(report)
    assert report.errors


def test_validate_reports_multiple_data_inputs():
    config = build_default_workflow_config("high_quality_uav_gpr")
    a, b, c = config.methods[:3]
    config.canvas_links = [
        WorkflowLink(a.node_id, c.node_id),
        WorkflowLink(b.node_id, c.node_id),
    ]

    report = validate_workflow_config(config)

    assert "multiple_data_inputs" in _codes(report)
    assert report.errors


def test_validate_reports_cycle():
    config = build_default_workflow_config("high_quality_uav_gpr")
    a, b, c = config.methods[:3]
    config.canvas_links = [
        WorkflowLink(a.node_id, b.node_id),
        WorkflowLink(b.node_id, c.node_id),
        WorkflowLink(c.node_id, a.node_id),
    ]

    report = validate_workflow_config(config)

    assert "cycle_detected" in _codes(report)
    assert report.errors


def test_validate_reports_graph_order_mismatch():
    config = build_default_workflow_config("high_quality_uav_gpr")
    a, b = config.methods[:2]
    config.canvas_links = [WorkflowLink(b.node_id, a.node_id)]

    report = validate_workflow_config(config)

    assert "graph_order_mismatch" in _codes(report)


def test_validate_reports_isolated_node():
    config = build_default_workflow_config("high_quality_uav_gpr")
    config.canvas_links = []

    report = validate_workflow_config(config)

    assert "isolated_node" in _codes(report)


def test_topological_method_order_returns_graph_order_for_dag():
    config = build_default_workflow_config("high_quality_uav_gpr")
    a, b, c = config.methods[:3]
    config.canvas_links = [
        WorkflowLink(a.node_id, b.node_id),
        WorkflowLink(b.node_id, c.node_id),
    ]
    config.methods = [c, a, b]

    ordered = topological_method_order(config)

    assert ordered == [a.node_id, b.node_id, c.node_id]


def test_topological_method_order_returns_empty_for_cycle():
    config = build_default_workflow_config("high_quality_uav_gpr")
    a, b = config.methods[:2]
    config.canvas_links = [
        WorkflowLink(a.node_id, b.node_id),
        WorkflowLink(b.node_id, a.node_id),
    ]

    assert topological_method_order(config) == []
