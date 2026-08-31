#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application errors for automatic parameter tuning."""
from __future__ import annotations

from mygpr.domain.common.errors import MyGPRError


class AutoTuneError(MyGPRError):
    """Raised when AutoTune configuration or execution fails."""

    error_code = "MYGPR_AUTOTUNE_ERROR"
    category = "autotune"
    default_hint = "确认处理方法、候选参数、关注范围和输入 B-scan 有效。"


class AutoTuneCancelled(AutoTuneError):
    """Raised when AutoTune is cancelled by the caller."""

    error_code = "MYGPR_AUTOTUNE_CANCELLED"
    category = "autotune"
    default_hint = "任务已按请求停止；可以调整范围或参数后重新执行。"


__all__ = ["AutoTuneCancelled", "AutoTuneError"]
