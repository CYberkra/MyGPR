"""Timestamp discipline: every record timestamp must be UTC + millisecond precision.

Why this is a gate rather than a style preference: several call sites *sort*
these strings (``core/job_manager.py`` orders job records by ``created_at``,
``core/processing_artifact_index.py`` orders artifacts by it). Mixing
microsecond and millisecond precision — or worse, mixing UTC with local time —
breaks the invariant that lexicographic order equals chronological order, and
does so intermittently: the bug only appears when two records land in the same
millisecond bucket with different sub-millisecond digits.
"""
from __future__ import annotations

import ast
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("core", "mygpr", "ui")

# datetime.now(...) without timezone.utc -> a local-time timestamp.
NAIVE_NOW = re.compile(r"datetime\.now\(\s*\)|datetime\.now\(\s*(?!\s*timezone\.utc)")
# isoformat() without an explicit timespec -> microsecond precision, which is
# inconsistent with core.storage_primitives.utc_now.
BARE_ISO = re.compile(r"\.isoformat\(\s*\)")

# Files allowed to keep a compact, non-ISO stamp: the value becomes a path
# segment or a user-facing label, so local time is the correct choice.
PATH_STAMP_ALLOWLIST = {
    "core/report_export_rows.py",        # report package directory name
    "core/field_project_operations.py",  # default project directory name
    "ui/widgets/output_panel.py",         # log line prefix + export filename
}


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        files.extend(p for p in (ROOT / root).rglob("*.py") if "__pycache__" not in p.parts)
    return files


def _timestamp_lines(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, source) for lines that produce a timestamp string."""
    hits: list[tuple[int, str]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        if "isoformat(" in line or "strftime(" in line:
            hits.append((index, line))
    return hits


def test_no_naive_datetime_now_in_production_code():
    """A record timestamp without timezone.utc silently means local time."""
    offenders: list[str] = []
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in PATH_STAMP_ALLOWLIST:
            continue
        for index, line in _timestamp_lines(path):
            if NAIVE_NOW.search(line):
                offenders.append(f"{rel}:{index}: {line.strip()}")
    assert not offenders, "时间戳必须带 timezone.utc:\n" + "\n".join(offenders)


def test_no_bare_isoformat_in_production_code():
    """Bare .isoformat() yields microseconds; the project standard is milliseconds."""
    offenders: list[str] = []
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        for index, line in _timestamp_lines(path):
            if BARE_ISO.search(line):
                offenders.append(f"{rel}:{index}: {line.strip()}")
    assert not offenders, "请显式指定 timespec=\"milliseconds\":\n" + "\n".join(offenders)


def test_utc_now_is_the_single_source_of_truth():
    """core.storage_primitives.utc_now defines the canonical format."""
    tree = ast.parse((ROOT / "core/storage_primitives.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "utc_now" in functions, "core.storage_primitives 必须提供 utc_now"
    body = ast.unparse(functions["utc_now"])
    assert "timezone.utc" in body, f"utc_now 必须用 UTC: {body}"
    assert "milliseconds" in body, f"utc_now 必须用毫秒精度: {body}"


def test_utc_now_output_is_lexicographically_sortable():
    produced = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    assert produced.endswith("+00:00")
    # Same instant rendered twice must keep the same millisecond field width.
    assert len(produced.split("T")[1].split("+")[0]) == len("00:00:00.000")


def test_report_package_timestamp_is_utc():
    """The report-package directory stamp is a path segment, but still UTC."""
    source = (ROOT / "core/report_export_rows.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_timestamp":
            body = ast.unparse(node)
            assert "timezone.utc" in body, f"报告包目录名应使用 UTC: {body}"
            assert "%Y%m%d_%H%M%S" in body, "目录名保持紧凑格式"
            return
    pytest.fail("core/report_export_rows.py 缺少 _timestamp")


def test_no_local_timezone_stamp_in_report_export():
    """Guard the specific regression: datetime.now() without tz in report rows."""
    source = (ROOT / "core/report_export_rows.py").read_text(encoding="utf-8")
    for index, line in enumerate(source.splitlines(), start=1):
        if "strftime(" in line and "timezone.utc" not in line:
            pytest.fail(f"core/report_export_rows.py:{index} 的 strftime 未指定 UTC: {line.strip()}")
