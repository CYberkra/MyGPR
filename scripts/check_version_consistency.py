#!/usr/bin/env python3
"""Check the current release identity across metadata and active entry documents."""
from __future__ import annotations

import tomllib
import json
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
    manifest_path = ROOT / "config/backend_source_manifest.json"
    manifest_version = json.loads(manifest_path.read_text(encoding="utf-8"))["version"]
    if manifest_version != version_file:
        raise AssertionError(f"config/backend_source_manifest.json version={manifest_version!r} != VERSION={version_file!r}")
    markers = {
        "README.md": f"# MyGPR {version_file}\n",
        "AGENTS.md": f"processing, v{version_file}.",
        "CLAUDE.md": f"# MyGPR v{version_file} —",
        "GUI_README.md": f"# MyGPR v{version_file} —",
        "CURRENT_STATE.md": f"Version: {version_file}.",
        "TODO.md": f"当前后端基线：**v{version_file}**",
    }
    for name, marker in markers.items():
        if marker not in (ROOT / name).read_text(encoding="utf-8"):
            raise AssertionError(f"{name} does not declare current VERSION={version_file!r}")
    release_headers = [line for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
                       if line.startswith("## ") and line != "## Unreleased"]
    if not release_headers or not release_headers[0].startswith(f"## {version_file} "):
        raise AssertionError(f"CHANGELOG.md latest release must be VERSION={version_file!r}")
    return version_file


if __name__ == "__main__":
    print(check_version_consistency())
