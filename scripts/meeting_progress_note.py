#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append concise meeting progress notes into the UAV-GPR Obsidian vault."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence


DEFAULT_VAULT_PATH = Path(r"D:\ClawX-Data\Obsidian\uav_gpr")
DEFAULT_NOTE_PATH = Path("10-项目/组会进展/组会进展记录.md")


def _clean_items(items: Sequence[str] | None) -> list[str]:
    return [str(item).strip() for item in (items or []) if str(item).strip()]


def _format_bullets(items: Sequence[str], fallback: str) -> str:
    cleaned = _clean_items(items)
    if not cleaned:
        return f"- {fallback}"
    return "\n".join(f"- {item}" for item in cleaned)


def _obsidian_link_for_vault_path(path_text: str, vault_path: Path) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        rel = path.resolve().relative_to(vault_path.resolve())
    except Exception:
        return path_text
    return "[[" + str(rel).replace("\\", "/").removesuffix(".md") + "]]"


def build_note_entry(
    *,
    summary: str,
    progress: Sequence[str] | None = None,
    results: Sequence[str] | None = None,
    risks: Sequence[str] | None = None,
    next_steps: Sequence[str] | None = None,
    commit: str = "",
    tag: str = "",
    archive_note: str = "",
    vault_path: Path = DEFAULT_VAULT_PATH,
    timestamp: datetime | None = None,
) -> str:
    """Build one concise meeting-progress entry."""
    timestamp = timestamp or datetime.now()
    title = summary.strip() or "组会进展更新"
    archive_display = _obsidian_link_for_vault_path(archive_note, vault_path)
    relations = []
    if commit:
        relations.append(f"Commit: `{commit}`")
    if tag:
        relations.append(f"Tag: `{tag}`")
    if archive_display:
        relations.append(f"版本归档: {archive_display}")

    return f"""
## {timestamp.strftime('%Y-%m-%d %H:%M')} - {title}

### 本次主要进展
{_format_bullets(progress, title)}

### 可展示成果
{_format_bullets(results, "已形成可回看的重要检查点。")}

### 当前问题/风险
{_format_bullets(risks, "暂无新增风险记录。")}

### 下一步
{_format_bullets(next_steps, "按下一轮任务继续推进。")}

### 关联
{_format_bullets(relations, "暂无关联提交或归档。")}
""".rstrip() + "\n"


def _initial_note_content() -> str:
    return """---
type: meeting-progress
domain: uav-gpr
section: 项目
status: active
llm_ready: true
tags:
  - 组会
  - 进展
  - MyGPR
---

# 组会进展记录

这里记录每次重要检查点或组会后的简洁进展，方便快速回忆成果、可展示内容和下一步。
"""


def append_meeting_progress_note(
    *,
    summary: str,
    progress: Sequence[str] | None = None,
    results: Sequence[str] | None = None,
    risks: Sequence[str] | None = None,
    next_steps: Sequence[str] | None = None,
    commit: str = "",
    tag: str = "",
    archive_note: str = "",
    vault_path: Path = DEFAULT_VAULT_PATH,
    note_path: Path = DEFAULT_NOTE_PATH,
    timestamp: datetime | None = None,
    dry_run: bool = False,
) -> Path:
    """Append one progress entry and return the target note path."""
    vault_path = Path(vault_path)
    note_path = Path(note_path)
    target = note_path if note_path.is_absolute() else vault_path / note_path
    entry = build_note_entry(
        summary=summary,
        progress=progress,
        results=results,
        risks=risks,
        next_steps=next_steps,
        commit=commit,
        tag=tag,
        archive_note=archive_note,
        vault_path=vault_path,
        timestamp=timestamp,
    )

    if dry_run:
        print(f"[dry-run] meeting progress note: {target}")
        print(entry)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        content = existing.rstrip() + "\n\n" + entry
    else:
        content = _initial_note_content().rstrip() + "\n\n" + entry
    target.write_text(content, encoding="utf-8")
    print(f"Meeting progress note updated: {target}")
    return target


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a concise meeting progress note into Obsidian."
    )
    parser.add_argument("--summary", required=True, help="Entry title/summary")
    parser.add_argument("--progress", action="append", default=[], help="主要进展")
    parser.add_argument("--result", action="append", default=[], help="可展示成果")
    parser.add_argument("--risk", action="append", default=[], help="当前问题/风险")
    parser.add_argument("--next-step", action="append", default=[], help="下一步")
    parser.add_argument("--commit", default="", help="Related commit hash")
    parser.add_argument("--tag", default="", help="Related git tag")
    parser.add_argument("--archive-note", default="", help="Related Obsidian archive path")
    parser.add_argument("--vault-path", default=str(DEFAULT_VAULT_PATH))
    parser.add_argument("--note-path", default=str(DEFAULT_NOTE_PATH))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    append_meeting_progress_note(
        summary=args.summary,
        progress=args.progress,
        results=args.result,
        risks=args.risk,
        next_steps=args.next_step,
        commit=args.commit,
        tag=args.tag,
        archive_note=args.archive_note,
        vault_path=Path(args.vault_path),
        note_path=Path(args.note_path),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
