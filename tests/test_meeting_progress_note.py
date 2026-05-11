#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for meeting-progress note generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scripts.meeting_progress_note import append_meeting_progress_note, build_note_entry


def test_first_run_creates_meeting_progress_note(tmp_path: Path):
    vault = tmp_path / "vault"

    note_path = append_meeting_progress_note(
        summary="feat: checkpoint",
        progress=["完成数据上下文分流"],
        results=["形成可回滚 tag"],
        risks=["真实传感器数据仍缺失"],
        next_steps=["继续完善自动选参"],
        commit="abc1234",
        tag="v2026-05-11-demo",
        archive_note=str(vault / "40-归档与历史" / "版本快照" / "demo.md"),
        vault_path=vault,
        timestamp=datetime(2026, 5, 11, 19, 30),
    )

    assert note_path == vault / "10-项目" / "组会进展" / "组会进展记录.md"
    text = note_path.read_text(encoding="utf-8")
    assert "# 组会进展记录" in text
    assert "## 2026-05-11 19:30 - feat: checkpoint" in text
    assert "- 完成数据上下文分流" in text
    assert "- 形成可回滚 tag" in text
    assert "- 真实传感器数据仍缺失" in text
    assert "- 继续完善自动选参" in text
    assert "Commit: `abc1234`" in text
    assert "Tag: `v2026-05-11-demo`" in text
    assert "[[40-归档与历史/版本快照/demo]]" in text


def test_multiple_runs_append_without_overwriting(tmp_path: Path):
    vault = tmp_path / "vault"

    append_meeting_progress_note(
        summary="first",
        vault_path=vault,
        timestamp=datetime(2026, 5, 11, 10, 0),
    )
    append_meeting_progress_note(
        summary="second",
        vault_path=vault,
        timestamp=datetime(2026, 5, 12, 10, 0),
    )

    text = (vault / "10-项目" / "组会进展" / "组会进展记录.md").read_text(
        encoding="utf-8"
    )
    assert text.count("## 2026-05-11 10:00 - first") == 1
    assert text.count("## 2026-05-12 10:00 - second") == 1


def test_minimal_entry_uses_summary_and_fallbacks():
    text = build_note_entry(
        summary="feat: minimal checkpoint",
        commit="abc1234",
        tag="v2026-05-11-minimal",
        timestamp=datetime(2026, 5, 11, 20, 0),
    )

    assert "- feat: minimal checkpoint" in text
    assert "- 已形成可回看的重要检查点。" in text
    assert "- 暂无新增风险记录。" in text
    assert "- 按下一轮任务继续推进。" in text
    assert "- Commit: `abc1234`" in text
    assert "- Tag: `v2026-05-11-minimal`" in text


def test_dry_run_does_not_create_note(tmp_path: Path):
    vault = tmp_path / "vault"

    target = append_meeting_progress_note(
        summary="dry",
        vault_path=vault,
        dry_run=True,
    )

    assert target == vault / "10-项目" / "组会进展" / "组会进展记录.md"
    assert not target.exists()
