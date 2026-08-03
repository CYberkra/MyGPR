#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DesktopBackendFacade — ui/ 唯一允许接触的 core/ 与跨层导入通道。

所有 ``ui/`` 模块必须通过本文件获取 core/ 数据模型、GUI 渲染辅助、
方法注册表元数据、文件格式过滤器及 job 状态类型；禁止在 ui/ 内直接
``import core.*`` 或 ``import mygpr.domain.*`` / ``mygpr.application.*``。

Rationale: architecture_policy.toml 将逐步收紧 ui/ 导入白名单；
本文件是当前 ui→core / ui→domain / ui→application 直接依赖的
集中式迁移垫片。
"""
from __future__ import annotations

# Data models
from core.gpr_data_model import GPRDataSet

# GUI rendering helpers
from core.gui_rendering import bundle_from_dataset, compute_levels

# File format dialog filter (project page import filter)
from core.gpr_format_registry import supported_file_dialog_filter

# Method registry display metadata
from core.method_registry_metadata import (
    METHOD_CATEGORY_LABELS,
    METHOD_METADATA,
    METHOD_TAGS,
    PREFERRED_METHOD_ORDER,
)
from core.methods_registry import PROCESSING_METHODS

# Job state types used by BackendController / JobBridge
from mygpr.application.jobs.models import JobEventType, JobResultSummary, JobSnapshot

# Processing pipeline types used by ProcessingController
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

__all__ = [
    "GPRDataSet",
    "bundle_from_dataset",
    "compute_levels",
    "supported_file_dialog_filter",
    "METHOD_CATEGORY_LABELS",
    "METHOD_METADATA",
    "METHOD_TAGS",
    "PREFERRED_METHOD_ORDER",
    "PROCESSING_METHODS",
    "JobEventType",
    "JobResultSummary",
    "JobSnapshot",
    "PipelineDefinition",
    "PipelineStep",
]
