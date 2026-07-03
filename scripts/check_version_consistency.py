from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_version(root: Path) -> str:
    version_path = root / "VERSION"
    if not version_path.exists():
        raise AssertionError("VERSION file is missing")
    version = version_path.read_text(encoding="utf-8-sig").strip()
    if not SEMVER_RE.match(version):
        raise AssertionError(f"VERSION must be MAJOR.MINOR.PATCH, got: {version!r}")
    return version


def check_version_consistency(expected: str | None = None) -> str:
    root = _repo_root()
    version = read_version(root)
    if expected and version != expected:
        raise AssertionError(f"VERSION mismatch: expected {expected}, got {version}")

    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        raise AssertionError("CHANGELOG.md is missing")
    changelog_text = changelog.read_text(encoding="utf-8")
    if f"## {version}" not in changelog_text:
        raise AssertionError(f"CHANGELOG.md does not contain entry for {version}")

    policy_candidates = [
        root / "docs" / "developer" / "versioning_policy.md",
        root / "docs" / "versioning_policy.md",
    ]
    if not any(path.exists() for path in policy_candidates):
        raise AssertionError("versioning policy is missing from docs/developer/versioning_policy.md")

    spec = root / "gpr_gui.spec"
    if spec.exists():
        spec_text = spec.read_text(encoding="utf-8")
        if "('VERSION', '.')" not in spec_text and '("VERSION", ".")' not in spec_text:
            raise AssertionError("gpr_gui.spec must include VERSION in datas")

    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check MyGPR release version consistency.")
    parser.add_argument("--expected", help="Expected semantic version, e.g. 0.8.38")
    args = parser.parse_args(argv)
    try:
        version = check_version_consistency(args.expected)
    except AssertionError as exc:
        print(f"version_check_failed: {exc}", file=sys.stderr)
        return 1
    print(f"version_check_ok: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
