#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for ChatGPT patch exchange helpers."""

from __future__ import annotations

from tools.apply_chatgpt_patch import _extract_base_commit, _parse_touched_files


def test_extract_base_commit_accepts_common_headers():
    patch_text = """Base-Commit: 1c33b11abcdef1234567890abcdef1234567890
diff --git a/foo.py b/foo.py
"""

    assert _extract_base_commit(patch_text) == "1c33b11abcdef1234567890abcdef1234567890"


def test_parse_touched_files_from_unified_diff():
    patch_text = """Base-Commit: 1c33b11
diff --git a/ui/workflow_canvas_cards.py b/ui/workflow_canvas_cards.py
--- a/ui/workflow_canvas_cards.py
+++ b/ui/workflow_canvas_cards.py
diff --git a/tests/test_workflow_page_editor.py b/tests/test_workflow_page_editor.py
--- a/tests/test_workflow_page_editor.py
+++ b/tests/test_workflow_page_editor.py
"""

    assert _parse_touched_files(patch_text) == [
        "tests/test_workflow_page_editor.py",
        "ui/workflow_canvas_cards.py",
    ]
