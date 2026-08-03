#!/usr/bin/env python3
"""Check that VERSION, pyproject.toml, and package metadata are consistent."""
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_version_consistency() -> str:
    version_file = (ROOT / "VERSION").read_text(encoding="utf-8-sig").strip()
    with (ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    pyproject_version = str(pyproject.get("project", {}).get("version", ""))
    if version_file != pyproject_version:
        raise AssertionError(
            f"VERSION={version_file!r} != pyproject.toml version={pyproject_version!r}"
        )
    return version_file


if __name__ == "__main__":
    print(check_version_consistency())
