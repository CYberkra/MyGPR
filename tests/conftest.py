#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pytest collection markers for controllable MyGPR test subsets.

The project now contains GUI, gprMax, integration and long-running validation
coverage in one tests/ tree.  These markers keep local/CI commands explicit
without editing every historical test file.
"""

from __future__ import annotations

from pathlib import Path

GUI_NAME_HINTS = (
    "gui",
    "daily_processing",
    "result_dialog",
    "no_prior_ui",
    "wiggle_ui",
    "workbench",
    "roi_picker",
    "app_sidecar_gui",
    "import_export_report",
    "shared_state_sync",
)

SLOW_NAME_HINTS = (
    "runner",
    "benchmark",
    "validation",
    "multi_scene",
    "native",
    "package",
    "motion_pipeline",
    "e2e",
    "no_zerotime",
    "post_zero_time",
    "risk_flag",
    "signal_loss",
    "demo",
)


# Broad sweep/regression files that are too expensive for the fast unit baseline
# despite generic filenames.
EXPLICIT_SLOW_FILES = {
    "test_auto_tune.py",
}

INTEGRATION_NAME_HINTS = (
    "cli",
    "sidecar",
    "evidence",
    "export",
    "workflow",
    "pipeline",
    "dashboard",
    "campaign",
    "package",
    "runner",
    "validation",
    "report",
)


def pytest_collection_modifyitems(config, items):  # noqa: D401 - pytest hook
    """Auto-apply high-level subset markers from test filenames."""
    for item in items:
        name = Path(str(item.fspath)).name.lower()
        markers = {mark.name for mark in item.iter_markers()}

        if any(hint in name for hint in GUI_NAME_HINTS):
            item.add_marker("gui")
            markers.add("gui")

        if "gprmax" in name or name.startswith("test_gx_"):
            item.add_marker("gprmax")
            item.add_marker("integration")
            markers.update({"gprmax", "integration"})

        if "wavelet" in name:
            item.add_marker("wavelet")
            markers.add("wavelet")

        if name in EXPLICIT_SLOW_FILES or any(hint in name for hint in SLOW_NAME_HINTS):
            item.add_marker("slow")
            markers.add("slow")

        if any(hint in name for hint in INTEGRATION_NAME_HINTS):
            item.add_marker("integration")
            markers.add("integration")

        if not ({"gui", "integration", "slow", "gprmax"} & markers):
            item.add_marker("unit")
