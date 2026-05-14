#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply a ChatGPT-generated patch with MyGPR smoke verification."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / ".chatgpt_inbox" / "_reports"


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"[run] {' '.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed with exit {result.returncode}: {' '.join(command)}")
    return result


def _git_output(args: list[str]) -> str:
    result = _run(["git", *args], check=True)
    return result.stdout or ""


def _ensure_clean() -> None:
    status = _git_output(["status", "--porcelain"])
    if status.strip():
        raise RuntimeError("working tree is not clean; refusing to apply patch\n" + status)


def _changed_files() -> list[str]:
    output = _git_output(["diff", "--name-only"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def _current_head() -> str:
    return _git_output(["rev-parse", "HEAD"]).strip()


def _current_branch() -> str:
    return _git_output(["branch", "--show-current"]).strip()


def _origin_blob_url(head: str, path: str) -> str:
    remote = _git_output(["config", "--get", "remote.origin.url"]).strip()
    if remote.endswith(".git"):
        remote = remote[:-4]
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote[len("git@github.com:") :]
    return f"{remote}/blob/{head}/{path.replace('\\', '/')}"


def _parse_touched_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", patch_text, re.MULTILINE):
        path = match.group(2).strip()
        if path and path != "/dev/null":
            files.append(path)
    return sorted(dict.fromkeys(files))


def _extract_base_commit(patch_text: str) -> str | None:
    patterns = [
        r"(?im)^\s*(?:#\s*)?base[-_ ]commit\s*[:=]\s*([0-9a-f]{7,40})\s*$",
        r"(?im)^\s*(?:#\s*)?base[-_ ]head\s*[:=]\s*([0-9a-f]{7,40})\s*$",
        r"(?im)^\s*(?:#\s*)?latest[-_ ]commit\s*[:=]\s*([0-9a-f]{7,40})\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, patch_text)
        if match:
            return match.group(1).lower()
    return None


def _write_failure_report(
    patch_file: Path,
    *,
    reason: str,
    command_output: str,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    patch_text = patch_file.read_text(encoding="utf-8", errors="replace")
    touched = _parse_touched_files(patch_text)
    head = _current_head()
    branch = _current_branch()
    base = _extract_base_commit(patch_text)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = REPORT_DIR / f"{timestamp}_{patch_file.stem}_failure.md"

    lines = [
        "# ChatGPT Patch Failure Report",
        "",
        f"- patch: `{patch_file.name}`",
        f"- branch: `{branch}`",
        f"- current_head: `{head}`",
        f"- declared_base_commit: `{base or 'MISSING'}`",
        f"- reason: {reason}",
        "",
        "## Touched Files",
    ]
    if touched:
        for path in touched:
            lines.append(f"- `{path}`")
            lines.append(f"  - current: {_origin_blob_url(head, path)}")
    else:
        lines.append("- No `diff --git` file entries were parsed.")

    lines.extend(
        [
            "",
            "## What To Send Back To Web GPT",
            "",
            "Regenerate the patch from the exact current branch state below. Do not reuse older snippets.",
            "",
            "```text",
            f"branch: {branch}",
            f"base commit: {head}",
            "Patch requirements:",
            "1. Fetch the current file contents from the links above before editing.",
            "2. Put `Base-Commit: <hash>` as the first line of the patch or patch message.",
            "3. Generate a unified diff against that exact base commit.",
            "4. Do not invent fake git index hashes such as aaaaaaa/c0ffee0; omit index lines if unknown.",
            "```",
            "",
            "## Command Output",
            "",
            "```text",
            command_output.strip() or "(no output)",
            "```",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[apply] wrote failure report: {report}", flush=True)
    return report


def _validate_declared_base(patch_file: Path, *, allow_missing_base: bool) -> None:
    patch_text = patch_file.read_text(encoding="utf-8", errors="replace")
    declared = _extract_base_commit(patch_text)
    head = _current_head().lower()
    if not declared:
        if not allow_missing_base:
            report = _write_failure_report(
                patch_file,
                reason="missing Base-Commit header",
                command_output="Patch did not declare Base-Commit.",
            )
            raise RuntimeError(f"patch missing Base-Commit header; see {report}")
        print("[apply] warning: patch does not declare Base-Commit", flush=True)
        return

    if not head.startswith(declared.lower()) and not declared.lower().startswith(head):
        report = _write_failure_report(
            patch_file,
            reason="declared base commit does not match current HEAD",
            command_output=f"declared={declared}\ncurrent={head}",
        )
        raise RuntimeError(f"patch base commit mismatch; see {report}")


def _apply_patch_file(patch_file: Path) -> None:
    """Apply patch, falling back to git's 3-way merge for shifted context."""
    check_result = _run(["git", "apply", "--check", str(patch_file)], check=False)
    if check_result.returncode == 0:
        _run(["git", "apply", str(patch_file)])
        return

    print("[apply] direct git apply failed; trying git apply --3way", flush=True)
    three_way_result = _run(["git", "apply", "--3way", str(patch_file)], check=False)
    if three_way_result.returncode != 0:
        report = _write_failure_report(
            patch_file,
            reason="git apply and git apply --3way both failed",
            command_output=(check_result.stdout or "") + "\n" + (three_way_result.stdout or ""),
        )
        raise RuntimeError(
            "patch could not be applied directly or with --3way; "
            f"it is likely based on stale file content; see {report}"
        )

    unresolved = _git_output(["diff", "--name-only", "--diff-filter=U"])
    if unresolved.strip():
        raise RuntimeError("patch produced unresolved conflicts\n" + unresolved)


def _summary_from_patch(patch_file: Path) -> str:
    stem = patch_file.stem
    stem = re.sub(r"^\d+[_-]*", "", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return f"apply: {stem}" if stem else f"apply: {patch_file.name}"


def _verify_mygpr_smoke(changed: list[str]) -> None:
    py_files = [path for path in changed if path.endswith(".py") and Path(REPO_ROOT, path).exists()]
    if py_files:
        _run([sys.executable, "-m", "py_compile", *py_files])

    workflow_touched = any(
        path.startswith("ui/workflow")
        or path.startswith("ui\\workflow")
        or path in {"ui/gui_workflow_page.py", "ui\\gui_workflow_page.py", "app_qt.py"}
        for path in changed
    )
    if workflow_touched:
        _run(
            [
                "pytest",
                "tests\\test_workflow_page_editor.py",
                "tests\\test_workflow_ui_contracts.py",
                "-q",
            ]
        )

    _run([sys.executable, "scripts\\preflight_check.py"])


def apply_patch(args: argparse.Namespace) -> int:
    patch_file = Path(args.patch_file).expanduser().resolve()
    if not patch_file.exists():
        raise FileNotFoundError(patch_file)

    _ensure_clean()
    _run(["git", "checkout", args.branch])
    _run(["git", "pull", "--ff-only"])
    _ensure_clean()

    _validate_declared_base(patch_file, allow_missing_base=args.allow_missing_base)
    _apply_patch_file(patch_file)

    changed = _changed_files()
    if not changed:
        print("[apply] patch produced no changes; nothing to commit")
        return 0

    if args.mygpr_smoke:
        _verify_mygpr_smoke(changed)

    _run(["git", "diff", "--check"])
    _run(["git", "add", "--", *changed])
    _run(["git", "commit", "-m", args.summary or _summary_from_patch(patch_file)])

    if args.push:
        _run(["git", "push", "origin", args.branch])

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--patch-file", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--mygpr-smoke", action="store_true")
    parser.add_argument(
        "--allow-missing-base",
        action="store_true",
        help="Allow legacy patches without a Base-Commit header.",
    )
    push_group = parser.add_mutually_exclusive_group()
    push_group.add_argument("--push", action="store_true")
    push_group.add_argument("--no-push", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.no_push:
        args.push = False
    try:
        return apply_patch(args)
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failure.
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
