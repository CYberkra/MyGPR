#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heuristic visual-comfort checks for MyGPR screenshot diagnostics.

These rules complement layout geometry checks.  They do not try to replace
human UI review; they encode the recurring discomfort points that appeared in
Windows captures: target B-scan too flat, report cover too small, and auxiliary
panels dominating the primary workspace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from ui.field_panels.layout_diagnostics_rules import SeverityLevel


@dataclass(frozen=True)
class VisualComfortIssue:
    severity: SeverityLevel
    page: str
    rule: str
    widget: str
    message: str
    expected: str
    observed: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _iter_pages(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    pages = payload.get("pages", [])
    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, dict):
                yield page


def _page_name(page: dict[str, Any]) -> str:
    return str(page.get("workspace_key") or page.get("page") or "unknown")


def _widgets(page: dict[str, Any]) -> dict[str, dict[str, Any]]:
    widgets = page.get("widgets", {})
    return widgets if isinstance(widgets, dict) else {}


def _geom(widgets: dict[str, dict[str, Any]], key: str) -> dict[str, Any] | None:
    value = widgets.get(key)
    return value if isinstance(value, dict) else None


def _ratio(a: int, b: int) -> float:
    return float(a) / float(max(b, 1))


def check_visual_comfort_payload(payload: dict[str, Any]) -> dict[str, Any]:
    issues: list[VisualComfortIssue] = []
    pages_checked = 0

    for page in _iter_pages(payload):
        pages_checked += 1
        name = _page_name(page)
        widgets = _widgets(page)

        target = _geom(widgets, "targetBscanCanvas")
        target_card = _geom(widgets, "targetBscanCard")
        if target and target_card:
            h = _int(target.get("height"))
            card_h = _int(target_card.get("height"))
            if h < 250 or _ratio(h, card_h) < 0.48:
                issues.append(VisualComfortIssue(
                    SeverityLevel.HIGH,
                    name,
                    "target_bscan_comfort_height",
                    "targetBscanCanvas",
                    "目标定位主剖面太扁，标注页视觉重心不足。",
                    "height >= 250 and height/card >= 0.48",
                    f"height={h}, card_height={card_h}",
                ))

        delivery_preview = _geom(widgets, "deliveryReportPreviewCard")
        delivery_cover = _geom(widgets, "deliveryReportCover")
        delivery_toc = _geom(widgets, "deliveryReportToc")
        if delivery_preview and delivery_cover:
            cover_w = _int(delivery_cover.get("width"))
            preview_w = _int(delivery_preview.get("width"))
            if cover_w < 290 or _ratio(cover_w, preview_w) < 0.25:
                issues.append(VisualComfortIssue(
                    SeverityLevel.MEDIUM,
                    name,
                    "report_cover_visual_weight",
                    "deliveryReportCover",
                    "报告封面预览视觉权重不足，页面不像正式报告预览器。",
                    "cover_width >= 290 and cover/preview >= 0.25",
                    f"cover_width={cover_w}, preview_width={preview_w}",
                ))
        if delivery_preview and delivery_toc and delivery_cover:
            toc_w = _int(delivery_toc.get("width"))
            cover_w = _int(delivery_cover.get("width"))
            if toc_w > cover_w * 1.85:
                issues.append(VisualComfortIssue(
                    SeverityLevel.MEDIUM,
                    name,
                    "report_toc_not_dominant",
                    "deliveryReportToc",
                    "报告目录区过宽，压过封面预览的视觉主次。",
                    "toc_width <= cover_width * 1.85",
                    f"toc_width={toc_w}, cover_width={cover_w}",
                ))

        spatial_map = _geom(widgets, "spatialMapCanvas")
        spatial_side = _geom(widgets, "spatialAuxSidePanel")
        if spatial_map and spatial_side:
            map_w = _int(spatial_map.get("width"))
            side_w = _int(spatial_side.get("width"))
            if side_w > map_w * 0.42:
                issues.append(VisualComfortIssue(
                    SeverityLevel.MEDIUM,
                    name,
                    "spatial_aux_not_heavy",
                    "spatialAuxSidePanel",
                    "空间成果右侧辅助栏视觉权重偏高。",
                    "side_width <= map_width * 0.42",
                    f"side_width={side_w}, map_width={map_w}",
                ))

    issue_payload = [issue.to_dict() for issue in issues]
    return {
        "schema": "mygpr.visual_comfort_check.v1",
        "ok": not issues,
        "issue_count": len(issues),
        "pages_checked": pages_checked,
        "rules": [
            "target_bscan_comfort_height",
            "report_cover_visual_weight",
            "report_toc_not_dominant",
            "spatial_aux_not_heavy",
        ],
        "issues": issue_payload,
    }


__all__ = ["VisualComfortIssue", "check_visual_comfort_payload"]
