from __future__ import annotations

import json
from pathlib import Path

from ui.field_panels.capture_service import build_capture_summary, validate_capture_summary, write_capture_summary

ROOT = Path(__file__).resolve().parents[1]


def test_capture_summary_uses_current_source_root(tmp_path: Path) -> None:
    summary = build_capture_summary(
        source_root=ROOT,
        output_dir=tmp_path,
        version="0.8.80",
        screenshot_files=["00_home_project_overview_v0.8.80.png"],
    )
    validate_capture_summary(summary, expected_source_root=ROOT, expected_version="0.8.80")
    assert summary["source_root"] == str(ROOT.resolve())
    assert summary["source_root_name"] == ROOT.name
    assert "v0.8.72" not in summary["source_root"]


def test_capture_summary_roundtrip_json(tmp_path: Path) -> None:
    summary = build_capture_summary(
        source_root=ROOT,
        output_dir=tmp_path,
        version="0.8.80",
        screenshot_files=["02_line_processing_v0.8.80.png"],
    )
    path = write_capture_summary(tmp_path, summary)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_capture_summary(payload, expected_source_root=ROOT, expected_version="0.8.80")
    assert payload["entrypoint"] == "app_qt.py"
    assert payload["capture_size"] == [1920, 1080]
