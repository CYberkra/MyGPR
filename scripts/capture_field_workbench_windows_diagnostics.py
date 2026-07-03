#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture MyGPR pages and real Windows screen diagnostics.

Run this on the target notebook after launching from the same Python used by
MyGPR.  The JSON records full screen geometry, available geometry excluding the
Windows taskbar, DPI and the compact-mode decision used by the main window.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if "--offscreen" in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from app_qt import create_main_window  # noqa: E402
from ui.field_panels.capture_service import capture_workbench_screenshots, validate_capture_summary  # noqa: E402


def read_version(root: Path) -> str:
    return (root / "VERSION").read_text(encoding="utf-8-sig").strip()


def screen_profile(app: QApplication) -> dict:
    screen = app.primaryScreen()
    if screen is None:
        return {"source": "no_screen"}
    geo = screen.geometry()
    available = screen.availableGeometry()
    return {
        "source": screen.name() or "primary_screen",
        "geometry": {"x": geo.x(), "y": geo.y(), "width": geo.width(), "height": geo.height()},
        "available_geometry": {
            "x": available.x(),
            "y": available.y(),
            "width": available.width(),
            "height": available.height(),
        },
        "device_pixel_ratio": screen.devicePixelRatio(),
        "logical_dpi": {"x": screen.logicalDotsPerInchX(), "y": screen.logicalDotsPerInchY()},
        "physical_dpi": {"x": screen.physicalDotsPerInchX(), "y": screen.physicalDotsPerInchY()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture MyGPR Windows fit diagnostics and screenshots.")
    parser.add_argument("--output", default=str(ROOT.parent / "mygpr_windows_fit_diagnostics"))
    parser.add_argument("--width", type=int, default=0, help="Override screenshot width. 0 means use available geometry.")
    parser.add_argument("--height", type=int, default=0, help="Override screenshot height. 0 means use available geometry.")
    parser.add_argument("--offscreen", action="store_true", help="Force QT_QPA_PLATFORM=offscreen for CI/sandbox capture.")
    args = parser.parse_args(argv)

    version = read_version(ROOT)
    app = QApplication.instance() or QApplication([])
    profile = screen_profile(app)
    available = profile.get("available_geometry", {}) if isinstance(profile, dict) else {}
    width = args.width or int(available.get("width") or 1450)
    height = args.height or int(available.get("height") or 790)
    # Avoid Qt offscreen's artificial 800×800 screen from shrinking the release screenshots.
    if width < 1120 or height < 650:
        width, height = 1450, 790

    window = create_main_window()
    summary = capture_workbench_screenshots(
        window,
        source_root=ROOT,
        output_dir=Path(args.output),
        version=version,
        size=(width, height),
    )
    validate_capture_summary(summary, expected_source_root=ROOT, expected_version=version)
    diagnostics = {
        "schema": "mygpr.windows_fit_diagnostics.v1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": version,
        "source_root": str(ROOT.resolve()),
        "screen_profile": profile,
        "window_screen_profile": getattr(window, "screen_profile", {}),
        "compact_mode": getattr(window, "compact_mode", None),
        "capture_size": [width, height],
        "summary": summary,
    }
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "screen_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    window.close()
    app.quit()
    print(f"windows_fit_capture_ok: {version} -> {out}")
    print(f"screen_available: {available.get('width', '--')}x{available.get('height', '--')} | capture: {width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
