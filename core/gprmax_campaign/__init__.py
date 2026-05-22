#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gprMax campaign dry-run loader and validator (backend only)."""

from core.gprmax_campaign.campaign_loader import load_campaign_yaml
from core.gprmax_campaign.pairing import (
    PairedOutputSpec,
    PairedOutputValidationResult,
    TargetResponseResult,
    generate_target_response,
    validate_paired_outputs,
)
from core.gprmax_campaign.schema import (
    Campaign,
    CampaignScene,
    CampaignValidationResult,
    GprMaxRunResult,
    GprMaxTaskSpec,
    SceneValidationResult,
    ValidationIssue,
    VALIDATION_INVALID,
    VALIDATION_READY,
    VALIDATION_WARNING,
)
from core.gprmax_campaign.runner import run_gprmax_task
from core.gprmax_campaign.validator import validate_campaign

__all__ = [
    "Campaign",
    "CampaignScene",
    "CampaignValidationResult",
    "GprMaxRunResult",
    "GprMaxTaskSpec",
    "SceneValidationResult",
    "ValidationIssue",
    "VALIDATION_INVALID",
    "VALIDATION_READY",
    "VALIDATION_WARNING",
    "load_campaign_yaml",
    "PairedOutputSpec",
    "PairedOutputValidationResult",
    "TargetResponseResult",
    "validate_paired_outputs",
    "generate_target_response",
    "run_gprmax_task",
    "validate_campaign",
]
