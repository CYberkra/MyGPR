#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for core.command_contract — capability-based workbench command policy."""

from __future__ import annotations

import pytest

from core.command_contract import STATE_CAPABILITIES, assert_command_allowed


# ── STATE_CAPABILITIES structure ────────────────────────────────────────────

class TestStateCapabilities:
    def test_empty_state_allows_project_create(self) -> None:
        assert "project_create" in STATE_CAPABILITIES["empty"]

    def test_empty_state_allows_project_open(self) -> None:
        assert "project_open" in STATE_CAPABILITIES["empty"]

    def test_empty_state_allows_import(self) -> None:
        assert "import" in STATE_CAPABILITIES["empty"]

    def test_empty_state_does_not_allow_processing(self) -> None:
        assert "processing" not in STATE_CAPABILITIES["empty"]

    def test_formal_ready_allows_processing(self) -> None:
        assert "processing" in STATE_CAPABILITIES["formal_ready"]

    def test_formal_ready_allows_interpretation(self) -> None:
        assert "interpretation" in STATE_CAPABILITIES["formal_ready"]

    def test_formal_ready_allows_spatial(self) -> None:
        assert "spatial" in STATE_CAPABILITIES["formal_ready"]

    def test_formal_ready_allows_report(self) -> None:
        assert "report" in STATE_CAPABILITIES["formal_ready"]

    def test_formal_ready_allows_display(self) -> None:
        assert "display" in STATE_CAPABILITIES["formal_ready"]

    def test_formal_ready_allows_qc(self) -> None:
        assert "qc" in STATE_CAPABILITIES["formal_ready"]

    def test_formal_ready_allows_evidence_export(self) -> None:
        assert "evidence_export" in STATE_CAPABILITIES["formal_ready"]

    def test_qc_blocked_allows_display(self) -> None:
        assert "display" in STATE_CAPABILITIES["qc_blocked"]

    def test_qc_blocked_disallows_processing(self) -> None:
        assert "processing" not in STATE_CAPABILITIES["qc_blocked"]

    def test_qc_review_required_allows_evidence_export(self) -> None:
        assert "evidence_export" in STATE_CAPABILITIES["qc_review_required"]

    def test_temporary_preview_allows_formalize(self) -> None:
        assert "formalize" in STATE_CAPABILITIES["temporary_preview"]

    def test_all_known_states_have_required_fields(self) -> None:
        expected_states = {"empty", "temporary_preview", "qc_review_required",
                           "qc_blocked", "formal_ready"}
        assert set(STATE_CAPABILITIES.keys()) == expected_states

    def test_all_capability_sets_are_frozensets(self) -> None:
        for caps in STATE_CAPABILITIES.values():
            assert isinstance(caps, frozenset)


# ── assert_command_allowed ──────────────────────────────────────────────────

class TestAssertCommandAllowed:
    def test_known_command_in_correct_state_passes(self) -> None:
        assert_command_allowed("formal_ready", "processing")

    def test_unknown_command_raises_permission_error(self) -> None:
        with pytest.raises(PermissionError, match="not allowed"):
            assert_command_allowed("empty", "processing")

    def test_unknown_state_treated_as_no_capabilities(self) -> None:
        with pytest.raises(PermissionError, match="not allowed"):
            assert_command_allowed("nonexistent_state", "display")

    def test_empty_string_state_treated_as_unknown(self) -> None:
        with pytest.raises(PermissionError, match="not allowed"):
            assert_command_allowed("", "display")

    def test_numeric_state_treated_as_unknown(self) -> None:
        with pytest.raises(PermissionError, match="not allowed"):
            assert_command_allowed(123, "display")  # type: ignore[arg-type]

    def test_formal_ready_allows_all_its_commands(self) -> None:
        for cmd in STATE_CAPABILITIES["formal_ready"]:
            assert_command_allowed("formal_ready", cmd)

    def test_empty_rejects_all_but_its_own_commands(self) -> None:
        allowed = STATE_CAPABILITIES["empty"]
        for cmd in STATE_CAPABILITIES["formal_ready"]:
            if cmd not in allowed:
                with pytest.raises(PermissionError):
                    assert_command_allowed("empty", cmd)

    def test_error_message_includes_command_name(self) -> None:
        with pytest.raises(PermissionError, match="processing"):
            assert_command_allowed("empty", "processing")

    def test_error_message_includes_state_name(self) -> None:
        with pytest.raises(PermissionError, match="empty"):
            assert_command_allowed("empty", "processing")
