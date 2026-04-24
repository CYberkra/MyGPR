#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archive a stable project checkpoint into the Obsidian vault."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAULT_PATH = Path(r"D:\ClawX-Data\Obsidian\uav_gpr")
DEFAULT_NOTE_DIR = Path("01-项目/版本归档")
DEFAULT_INDEX_NOTE = Path("01-项目/版本归档索引.md")


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", str(text).strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "checkpoint"


def _format_bullets(items: list[str], fallback: str) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return f"- {fallback}"
    return "\n".join(f"- {item}" for item in cleaned)


def _resolve_archive_topic(topic: str, summary: str, commit_subject: str) -> str:
    """归档标题优先使用用户显式输入，其次才回退到 git 提交标题。"""
    explicit_topic = str(topic or "").strip()
    if explicit_topic:
        return explicit_topic

    explicit_summary = str(summary or "").strip()
    if explicit_summary:
        return explicit_summary

    fallback_subject = str(commit_subject or "").strip()
    if fallback_subject:
        return fallback_subject

    return "checkpoint"


def _build_note_content(
    *,
    title: str,
    project_name: str,
    repo_name: str,
    branch: str,
    commit_full: str,
    commit_short: str,
    commit_subject: str,
    head_tags: list[str],
    summary: str,
    changes: list[str],
    risks: list[str],
    next_steps: list[str],
    recent_commits: list[str],
    worktree_status: str,
    archived_at: str,
) -> str:
    tag_block = "\n".join(f"  - {tag}" for tag in (["归档", "版本"] + head_tags))
    recent_block = _format_bullets(recent_commits, "暂无最近提交信息")
    status_block = worktree_status or "工作区干净"
    return f"""---
type: archive
domain: uav-gpr
section: 项目
status: active
llm_ready: true
project: {project_name}
repo: {repo_name}
branch: {branch}
commit: {commit_short}
archived_at: {archived_at}
tags:
{tag_block}
---

# {title}

## 摘要

{summary}

## 本次变更

{_format_bullets(changes, "待补充")}

## Git 状态

- 分支：`{branch}`
- 提交：`{commit_short}`
- 提交说明：{commit_subject or "无"}
- 标签：{", ".join(head_tags) if head_tags else "无"}

## 工作区状态

```text
{status_block}
```

## 最近提交

{recent_block}

## 风险点

{_format_bullets(risks, "待补充")}

## 下一步

{_format_bullets(next_steps, "待补充")}

## 关联

- [[01-项目/项目总览]]
- [[01-项目/LLM长期记忆与归档规则]]
"""


def _ensure_index_note(index_path: Path) -> None:
    if index_path.exists():
        return
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        """---
type: index
domain: uav-gpr
section: 项目
status: active
llm_ready: true
tags:
  - 项目
  - 归档
  - 索引
---

# 版本归档索引

这里汇总重要版本、阶段性可回滚节点和长期需要回查的 GUI / 算法变更记录。

## 条目

""",
        encoding="utf-8",
    )


def _update_index_note(index_path: Path, note_relpath: str, title: str) -> None:
    _ensure_index_note(index_path)
    line = f"- [[{note_relpath}|{title}]]"
    content = index_path.read_text(encoding="utf-8")
    if line in content:
        return
    if not content.endswith("\n"):
        content += "\n"
    content += line + "\n"
    index_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive a project checkpoint into Obsidian."
    )
    parser.add_argument("--summary", required=True, help="本次归档的简短摘要")
    parser.add_argument("--topic", default="", help="归档主题，参与文件命名")
    parser.add_argument(
        "--changes",
        action="append",
        default=[],
        help="本次相较之前的重要改动，可重复传入",
    )
    parser.add_argument(
        "--risks",
        action="append",
        default=[],
        help="当前已知风险点，可重复传入",
    )
    parser.add_argument(
        "--next-steps",
        action="append",
        default=[],
        help="建议的后续动作，可重复传入",
    )
    parser.add_argument(
        "--vault-path",
        default=str(DEFAULT_VAULT_PATH),
        help="Obsidian vault 根目录",
    )
    parser.add_argument(
        "--note-dir",
        default=str(DEFAULT_NOTE_DIR),
        help="归档笔记所在目录，相对于 vault 根目录",
    )
    parser.add_argument(
        "--index-note",
        default=str(DEFAULT_INDEX_NOTE),
        help="归档索引笔记路径，相对于 vault 根目录",
    )
    parser.add_argument(
        "--project-name",
        default="UAV-GPR GUI",
        help="项目显示名称",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出即将写入的路径和内容，不真正写文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vault_path = Path(args.vault_path)
    note_dir = Path(args.note_dir)
    index_note = Path(args.index_note)

    timestamp = datetime.now()
    stamp = timestamp.strftime("%Y-%m-%d %H:%M")
    file_stamp = timestamp.strftime("%Y%m%d-%H%M")

    repo_name = REPO_ROOT.name
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    commit_full = _run_git(["rev-parse", "HEAD"]) or "unknown"
    commit_short = commit_full[:7] if commit_full != "unknown" else "unknown"
    commit_subject = _run_git(["log", "-1", "--pretty=%s"]) or ""
    head_tags = [
        line for line in _run_git(["tag", "--points-at", "HEAD"]).splitlines() if line
    ]
    worktree_status = _run_git(["status", "--short"])
    recent_commits = [
        line for line in _run_git(["log", "--oneline", "-5"]).splitlines() if line
    ]

    topic = _resolve_archive_topic(args.topic, args.summary, commit_subject)
    slug = _slugify(topic)
    title = f"版本归档 - {stamp} - {topic}"
    note_name = f"{file_stamp}-{slug}.md"
    note_relpath = note_dir / note_name
    note_abspath = vault_path / note_relpath
    index_abspath = vault_path / index_note

    content = _build_note_content(
        title=title,
        project_name=args.project_name,
        repo_name=repo_name,
        branch=branch,
        commit_full=commit_full,
        commit_short=commit_short,
        commit_subject=commit_subject,
        head_tags=head_tags,
        summary=args.summary.strip(),
        changes=args.changes,
        risks=args.risks,
        next_steps=args.next_steps,
        recent_commits=recent_commits,
        worktree_status=worktree_status,
        archived_at=stamp,
    )

    if args.dry_run:
        print(f"[dry-run] note: {note_abspath}")
        print(f"[dry-run] index: {index_abspath}")
        print(content)
        return 0

    note_abspath.parent.mkdir(parents=True, exist_ok=True)
    note_abspath.write_text(content, encoding="utf-8")
    _update_index_note(index_abspath, str(note_relpath).replace("\\", "/"), title)

    print(f"Archived note: {note_abspath}")
    print(f"Updated index: {index_abspath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
