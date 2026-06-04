#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Page operation contracts for MyGPR GUI.

The GUI uses these contracts to keep page responsibilities explicit:
processing pages may mutate data, display pages may only change presentation,
quality pages summarize/check/export, and spatial pages handle metadata/spatial
products.  Tests assert these boundaries so future UI changes do not mix real
processing buttons into display-only pages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageOperationContract:
    page: str
    allowed_operation_types: frozenset[str]
    mutates_data: bool
    summary: str


PAGE_OPERATION_CONTRACTS: dict[str, PageOperationContract] = {
    "processing": PageOperationContract(
        page="processing",
        allowed_operation_types=frozenset({"import", "processing", "lineage"}),
        mutates_data=True,
        summary="导入数据、执行真实处理算法、维护处理链路。",
    ),
    "autotune": PageOperationContract(
        page="autotune",
        allowed_operation_types=frozenset({"autotune", "processing_recipe", "report"}),
        mutates_data=True,
        summary="生成可执行推荐流程与参数；运行方案时通过处理链路改变数据。",
    ),
    "display": PageOperationContract(
        page="display",
        allowed_operation_types=frozenset({"display_only", "compare", "screenshot_export"}),
        mutates_data=False,
        summary="只改变主图显示和对比视图，不改变处理数组和处理链路。",
    ),
    "quality": PageOperationContract(
        page="quality",
        allowed_operation_types=frozenset({"qc", "record", "report"}),
        mutates_data=False,
        summary="显示质量指标、处理记录、运行摘要和报告导出入口。",
    ),
    "spatial": PageOperationContract(
        page="spatial",
        allowed_operation_types=frozenset({"spatial", "metadata", "three_d_export"}),
        mutates_data=False,
        summary="显示航迹、地形、C-scan、三维和空间成果。",
    ),
}


def get_page_contract(page: str) -> PageOperationContract:
    key = str(page or "").strip().lower()
    if key not in PAGE_OPERATION_CONTRACTS:
        raise KeyError(f"Unknown page operation contract: {page!r}")
    return PAGE_OPERATION_CONTRACTS[key]


def assert_page_allows(page: str, operation_type: str) -> None:
    contract = get_page_contract(page)
    op = str(operation_type or "").strip()
    if op not in contract.allowed_operation_types:
        allowed = ", ".join(sorted(contract.allowed_operation_types))
        raise ValueError(f"Page {page!r} does not allow operation {op!r}; allowed: {allowed}")
