#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screenshot capture helpers for the MyGPR field workbench.

The service writes a strict ``capture_summary.json`` that records the exact
source root used for screenshots.  This prevents release artifacts for one
version from silently reusing paths from a previous package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PyQt6.QtWidgets import QApplication, QWidget

from ui.field_panels.layout_diagnostics_rules import check_layout_diagnostics_payload
from ui.field_panels.visual_comfort_rules import check_visual_comfort_payload


@dataclass(frozen=True)
class CaptureItem:
    workspace_key: str
    filename: str
    description: str


DEFAULT_CAPTURE_ITEMS = (
    CaptureItem("home", "00_home_project_overview", "项目总览"),
    CaptureItem("data_management", "01_project_management", "项目管理"),
    CaptureItem("processing_lab", "02_line_processing", "测线处理"),
    CaptureItem("interpretation", "03_target_positioning", "目标定位"),
    CaptureItem("spatial", "04_spatial_results", "空间成果"),
    CaptureItem("delivery", "05_delivery_report", "成果报告"),
)


def _read_version(source_root: Path) -> str:
    version_path = source_root / "VERSION"
    if version_path.exists():
        return version_path.read_text(encoding="utf-8-sig").strip()
    return "unknown"


def build_capture_summary(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    version: str | None = None,
    entrypoint: str = "app_qt.py",
    screenshot_files: Iterable[str] = (),
    capture_size: tuple[int, int] = (1920, 1080),
    screen_profile: dict | None = None,
    compact_mode: bool | None = None,
) -> dict:
    """Build the canonical screenshot summary payload."""
    root = Path(source_root).resolve()
    out = Path(output_dir).resolve()
    version_value = version or _read_version(root)
    return {
        "schema": "mygpr.capture_summary.v1",
        "software": "MyGPR",
        "version": version_value,
        "source_root": str(root),
        "source_root_name": root.name,
        "output_dir": str(out),
        "entrypoint": entrypoint,
        "capture_size": list(capture_size),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "screen_profile": screen_profile or {},
        "compact_mode": compact_mode,
        "screenshots": list(screenshot_files),
    }


def write_capture_summary(output_dir: str | Path, summary: dict) -> Path:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "capture_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_capture_summary(summary: dict, *, expected_source_root: str | Path, expected_version: str | None = None) -> None:
    expected_root = str(Path(expected_source_root).resolve())
    if summary.get("source_root") != expected_root:
        raise AssertionError(f"capture_summary source_root mismatch: {summary.get('source_root')!r} != {expected_root!r}")
    if expected_version and summary.get("version") != expected_version:
        raise AssertionError(f"capture_summary version mismatch: {summary.get('version')!r} != {expected_version!r}")


def _widget_layout_key(widget: QWidget) -> str:
    value = widget.property("layoutKey")
    if value:
        return str(value)
    name = widget.objectName()
    if name and any(name.startswith(prefix) for prefix in ("project", "processing", "spatial", "delivery")):
        return name
    return ""


def collect_layout_diagnostics(window, workspace_key: str) -> dict:
    """Collect stable geometry metrics for critical workbench widgets.

    The diagnostics use an explicit ``layoutKey`` dynamic property so widgets can
    keep their visual object names (for QSS styling) while still being visible
    to automated layout checks.  Geometry is recorded in Qt logical pixels.
    """
    widgets: dict[str, dict] = {}
    for widget in window.findChildren(QWidget):
        if not widget.isVisible():
            continue
        key = _widget_layout_key(widget)
        if not key:
            continue
        rect = widget.geometry()
        widgets[key] = {
            "x": int(rect.x()),
            "y": int(rect.y()),
            "width": int(rect.width()),
            "height": int(rect.height()),
            "visible": bool(widget.isVisible()),
            "class": type(widget).__name__,
        }
    return {
        "workspace_key": workspace_key,
        "window": {"width": int(window.width()), "height": int(window.height())},
        "widgets": widgets,
    }


def capture_workbench_screenshots(
    window,
    *,
    source_root: str | Path,
    output_dir: str | Path,
    items: Iterable[CaptureItem] = DEFAULT_CAPTURE_ITEMS,
    version: str | None = None,
    size: tuple[int, int] = (1920, 1080),
) -> dict:
    """Capture the supplied workbench window and write a summary.

    ``window`` is intentionally duck-typed so this service stays independent of
    the large field workbench class.  It only requires ``switch_workspace`` and
    ``grab`` to be available.
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    window.resize(*size)
    window.show()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
    screenshot_files: list[str] = []
    layout_pages: list[dict] = []
    version_value = version or _read_version(Path(source_root).resolve())
    for item in items:
        window.switch_workspace(item.workspace_key)
        if app is not None:
            app.processEvents()
        filename = f"{item.filename}_v{version_value}.png"
        path = out / filename
        window.grab().save(str(path))
        screenshot_files.append(filename)
        layout_pages.append(collect_layout_diagnostics(window, item.workspace_key))
    summary = build_capture_summary(
        source_root=source_root,
        output_dir=out,
        version=version_value,
        screenshot_files=screenshot_files,
        capture_size=size,
        screen_profile=getattr(window, "screen_profile", {}),
        compact_mode=getattr(window, "compact_mode", None),
    )
    write_capture_summary(out, summary)
    layout_payload = {
        "schema": "mygpr.layout_diagnostics.v1",
        "version": version_value,
        "source_root": str(Path(source_root).resolve()),
        "capture_size": list(size),
        "created_at": summary.get("created_at"),
        "pages": layout_pages,
    }
    layout_check = check_layout_diagnostics_payload(layout_payload)
    visual_check = check_visual_comfort_payload(layout_payload)
    layout_payload["check"] = layout_check
    layout_payload["visual_check"] = visual_check
    (out / "layout_diagnostics.json").write_text(json.dumps(layout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "layout_check_report.json").write_text(json.dumps(layout_check, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "visual_check_report.json").write_text(json.dumps(visual_check, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "CaptureItem",
    "collect_layout_diagnostics",
    "DEFAULT_CAPTURE_ITEMS",
    "build_capture_summary",
    "capture_workbench_screenshots",
    "validate_capture_summary",
    "write_capture_summary",
]
