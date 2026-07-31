#!/usr/bin/env python3
"""Compile Python sources in memory without creating repository cache files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence


def _iter_python_files(root: Path, values: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for value in values:
        path = (root / value).resolve()
        if path.is_file() and path.suffix == ".py":
            files.add(path)
        elif path.is_dir():
            files.update(
                candidate
                for candidate in path.rglob("*.py")
                if not any(part in {".venv", "venv", ".git"} for part in candidate.parts)
            )
    return sorted(files)


def check_compile(root: Path, values: Iterable[str]) -> dict[str, object]:
    files = _iter_python_files(root, values)
    errors: list[str] = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8-sig")
            compile(source, str(path), "exec", dont_inherit=True)
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(root).as_posix()}: {exc}")
    return {
        "schema": "mygpr.compile_check.v1",
        "ok": not errors,
        "file_count": len(files),
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["core", "ui", "scripts", "tests"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    payload = check_compile(args.root.resolve(), args.paths)
    print(f"compile_check: {'OK' if payload['ok'] else 'FAILED'} files={payload['file_count']}")
    for error in payload["errors"]:
        print("  -", error)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
