#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for verified Git checkpoint automation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import git_checkpoint


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "codex@example.invalid")
    _git(repo, "config", "user.name", "Codex Test")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "init")
    return repo


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def test_dry_run_does_not_stage_commit_or_tag(tmp_path: Path):
    repo = _init_repo(tmp_path)
    before = _head(repo)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")

    code = git_checkpoint.main(
        [
            "--summary",
            "test: dry run",
            "--files",
            "tracked.txt",
            "--mode",
            "important",
            "--topic",
            "dry-run-topic",
            "--dry-run",
        ],
        repo_root=repo,
    )

    assert code == 0
    assert _head(repo) == before
    assert _git(repo, "diff", "--cached", "--name-only") == ""
    assert _git(repo, "tag", "--list") == ""


def test_verify_failure_does_not_stage_or_commit(tmp_path: Path):
    repo = _init_repo(tmp_path)
    before = _head(repo)
    fail_script = repo / "fail_verify.py"
    fail_script.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")

    code = git_checkpoint.main(
        [
            "--summary",
            "test: should not commit",
            "--files",
            "tracked.txt",
            "--verify",
            f'"{sys.executable}" "{fail_script}"',
        ],
        repo_root=repo,
    )

    assert code == 1
    assert _head(repo) == before
    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_empty_staged_diff_skips_commit(tmp_path: Path):
    repo = _init_repo(tmp_path)
    before = _head(repo)

    code = git_checkpoint.main(
        ["--summary", "test: empty", "--files", "tracked.txt"],
        repo_root=repo,
    )

    assert code == 0
    assert _head(repo) == before
    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_files_pathspec_limits_committed_changes(tmp_path: Path):
    repo = _init_repo(tmp_path)
    (repo / "tracked.txt").write_text("committed\n", encoding="utf-8")
    (repo / "left_dirty.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "left_dirty.txt")
    _git(repo, "commit", "-m", "add dirty target")
    before = _head(repo)
    (repo / "tracked.txt").write_text("committed\n", encoding="utf-8")
    (repo / "left_dirty.txt").write_text("not committed\n", encoding="utf-8")

    code = git_checkpoint.main(
        [
            "--summary",
            "test: commit only selected file",
            "--files",
            "tracked.txt",
            "--verify",
            f'"{sys.executable}" -c "print(123)"',
        ],
        repo_root=repo,
    )

    assert code == 0
    assert _head(repo) != before
    assert _git(repo, "log", "-1", "--pretty=%s") == "test: commit only selected file"
    status = _git(repo, "status", "--short")
    assert "M left_dirty.txt" in status
    assert "tracked.txt" not in status


def test_important_mode_tags_and_archives_after_commit(monkeypatch, tmp_path: Path):
    repo = _init_repo(tmp_path)
    (repo / "tracked.txt").write_text("important\n", encoding="utf-8")
    archive_calls: list[tuple[str, str, Path]] = []

    def fake_archive(*, repo_root: Path, summary: str, topic: str) -> None:
        archive_calls.append((summary, topic, repo_root))

    monkeypatch.setattr(git_checkpoint, "_run_archive_checkpoint", fake_archive)

    code = git_checkpoint.main(
        [
            "--summary",
            "feat: important checkpoint",
            "--files",
            "tracked.txt",
            "--mode",
            "important",
            "--topic",
            "important checkpoint",
        ],
        repo_root=repo,
    )

    assert code == 0
    tags = _git(repo, "tag", "--list").splitlines()
    assert len(tags) == 1
    assert tags[0].startswith("v")
    assert tags[0].endswith("-important-checkpoint")
    assert archive_calls == [("feat: important checkpoint", "important checkpoint", repo)]


def test_preexisting_staged_changes_are_rejected(tmp_path: Path):
    repo = _init_repo(tmp_path)
    (repo / "tracked.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    before = _head(repo)

    code = git_checkpoint.main(
        ["--summary", "test: reject", "--files", "tracked.txt"],
        repo_root=repo,
    )

    assert code == 1
    assert _head(repo) == before
    assert _git(repo, "diff", "--cached", "--name-only") == "tracked.txt"
