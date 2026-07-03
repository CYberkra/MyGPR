#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture 1080P screenshots for the current MyGPR field workbench."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep the script usable in headless CI/sandbox environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_qt import create_main_window  # noqa: E402
from ui.field_panels.capture_service import capture_workbench_screenshots, validate_capture_summary  # noqa: E402


def read_version(root: Path) -> str:
    return (root / "VERSION").read_text(encoding="utf-8-sig").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture MyGPR field workbench screenshots.")
    parser.add_argument("--output", default=str(ROOT.parent / "mygpr_v0.9.24_screenshots"), help="Screenshot output directory")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args(argv)

    version = read_version(ROOT)
    app = QApplication.instance() or QApplication([])
    window = create_main_window()
    summary = capture_workbench_screenshots(
        window,
        source_root=ROOT,
        output_dir=Path(args.output),
        version=version,
        size=(args.width, args.height),
    )
    validate_capture_summary(summary, expected_source_root=ROOT, expected_version=version)
    window.close()
    app.quit()
    print(f"capture_ok: {summary['version']} -> {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
