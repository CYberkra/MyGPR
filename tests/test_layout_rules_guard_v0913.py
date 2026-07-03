from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ui.field_panels.layout_diagnostics_rules import check_layout_diagnostics_payload


def _payload(*, map_card_h: int = 70, map_canvas_h: int = 54) -> dict:
    return {
        "schema": "mygpr.layout_diagnostics.v1",
        "version": "0.9.24",
        "capture_size": [1536, 816],
        "pages": [
            {
                "workspace_key": "processing_lab",
                "window": {"width": 1536, "height": 816},
                "widgets": {
                    "processingRawBscanCard": {"width": 553, "height": 431},
                    "processingRawBscanCanvas": {"width": 553, "height": 397},
                    "processingProcessedBscanCard": {"width": 553, "height": 431},
                    "processingProcessedBscanCanvas": {"width": 553, "height": 397},
                    "processingParamsCard": {"width": 255, "height": 390},
                    "processingMessagesCard": {"width": 1066, "height": 138},
                    "processingLineOverviewCard": {"width": 267, "height": 138},
                    "processingLineOverviewMapCard": {"width": 251, "height": map_card_h},
                    "processingLineOverviewMapCanvas": {"width": 249, "height": map_canvas_h},
                },
            }
        ],
    }


def test_layout_rules_pass_for_valid_processing_diagnostics() -> None:
    report = check_layout_diagnostics_payload(_payload())
    assert report["ok"] is True
    assert report["issue_count"] == 0


def test_layout_rules_fail_when_small_plot_card_clips_canvas() -> None:
    report = check_layout_diagnostics_payload(_payload(map_card_h=18, map_canvas_h=54))
    assert report["ok"] is False
    assert any(issue["rule"] == "canvas_within_plot_card" for issue in report["issues"])
    assert any(issue["rule"] == "small_plot_card_min_height" for issue in report["issues"])


def test_check_layout_diagnostics_cli_returns_nonzero_for_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "layout_diagnostics.json"
    path.write_text(json.dumps(_payload(map_card_h=18, map_canvas_h=54), ensure_ascii=False), encoding="utf-8")
    cp = subprocess.run(
        [sys.executable, "scripts/check_layout_diagnostics.py", str(path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert cp.returncode == 1
    assert "layout_check_failed" in cp.stderr


def test_check_layout_diagnostics_cli_accepts_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "layout_diagnostics.json"
    path.write_text(json.dumps(_payload(), ensure_ascii=False), encoding="utf-8")
    cp = subprocess.run(
        [sys.executable, "scripts/check_layout_diagnostics.py", str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert cp.returncode == 0
    assert "layout_check_ok" in cp.stdout
