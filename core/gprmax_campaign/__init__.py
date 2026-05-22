#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gprMax campaign dry-run loader and validator (backend only)."""

from core.gprmax_campaign.campaign_loader import load_campaign_yaml
from core.gprmax_campaign.schema import (
    Campaign,
    CampaignScene,
    CampaignValidationResult,
    SceneValidationResult,
    ValidationIssue,
    VALIDATION_INVALID,
    VALIDATION_READY,
    VALIDATION_WARNING,
)
from core.gprmax_campaign.validator import validate_campaign

__all__ = [
    "Campaign",
    "CampaignScene",
    "CampaignValidationResult",
    "SceneValidationResult",
    "ValidationIssue",
    "VALIDATION_INVALID",
    "VALIDATION_READY",
    "VALIDATION_WARNING",
    "load_campaign_yaml",
    "validate_campaign",
]
