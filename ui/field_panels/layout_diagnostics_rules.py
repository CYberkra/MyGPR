#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule-based checks for MyGPR layout diagnostics.

The capture service records widget geometries in ``layout_diagnostics.json``.
This module turns those measurements into a deterministic pass/fail report so
Windows 1080P layout regressions are caught without manual screenshot review.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


class SeverityLevel(StrEnum):
    HIGH = "P2"
    MEDIUM = "P3"


@dataclass(frozen=True)
class LayoutIssue:
    severity: SeverityLevel
    page: str
    rule: str
    widget: str
    message: str
    expected: str
    observed: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


CANVAS_CARD_PAIRS: dict[str, str] = {
    "projectSummaryMapCanvas": "projectSummaryMapCard",
    "projectQuickPreviewBscanCanvas": "projectQuickPreviewBscanCard",
    "projectQuickPreviewMapCanvas": "projectQuickPreviewMapCard",
    "processingRawBscanCanvas": "processingRawBscanCard",
    "processingProcessedBscanCanvas": "processingProcessedBscanCard",
    "processingLineOverviewMapCanvas": "processingLineOverviewMapCard",
    "targetBscanCanvas": "targetBscanPlotCard",
    "targetPreviewMapCanvas": "targetPreviewMapCard",
    "spatialMapCanvas": "spatialMapCard",
    "spatialProfileCanvas": "spatialProfileCard",
    "spatialDemCanvas": "spatialDemCard",
    "spatialCorrelationCanvas": "spatialCorrelationCard",
    "deliveryReportThumbCanvas": "deliveryReportThumbCard",
}

# Cards whose visual contract includes a title bar.  They need additional room
# above the canvas; no-title plot cards still need card padding/borders.
TITLED_PLOT_CARDS = {
    "projectQuickPreviewMapCard",
    "processingRawBscanCard",
    "processingProcessedBscanCard",
    "spatialProfileCard",
    "spatialDemCard",
    "spatialCorrelationCard",
    "targetPreviewMapCard",
}

SMALL_PLOT_PAIRS: dict[str, str] = {
    "projectQuickPreviewMapCanvas": "projectQuickPreviewMapCard",
    "processingLineOverviewMapCanvas": "processingLineOverviewMapCard",
    "targetPreviewMapCanvas": "targetPreviewMapCard",
}

BSCAN_CANVASES = ("processingRawBscanCanvas", "processingProcessedBscanCanvas")
TARGET_BSCAN_CANVASES = ("targetBscanCanvas",)
BOTTOM_CARDS = (
    "projectTaskTabsCard",
    "projectQuickPreviewCard",
    "processingMessagesCard",
    "processingLineOverviewCard",
    "deliveryFilesCard",
)


def _geom(widgets: dict[str, dict[str, Any]], key: str) -> dict[str, Any] | None:
    value = widgets.get(key)
    return value if isinstance(value, dict) else None


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


def _page_window_height(page: dict[str, Any], payload: dict[str, Any]) -> int:
    window = page.get("window", {}) if isinstance(page.get("window"), dict) else {}
    height = _int(window.get("height"), 0)
    if height > 0:
        return height
    capture_size = payload.get("capture_size", [])
    if isinstance(capture_size, list) and len(capture_size) >= 2:
        return _int(capture_size[1], 0)
    return 0


def check_layout_diagnostics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-readable layout check report.

    Rules are intentionally conservative and focused on the 1080P/125% layout
    failures that have appeared in user captures.
    """

    issues: list[LayoutIssue] = []
    pages_checked = 0

    for page in _iter_pages(payload):
        pages_checked += 1
        name = _page_name(page)
        widgets = page.get("widgets", {}) if isinstance(page.get("widgets"), dict) else {}
        window_h = _page_window_height(page, payload)

        for canvas_key, card_key in CANVAS_CARD_PAIRS.items():
            canvas = _geom(widgets, canvas_key)
            card = _geom(widgets, card_key)
            if not canvas or not card:
                continue
            canvas_h = _int(canvas.get("height"))
            card_h = _int(card.get("height"))
            if canvas_h > card_h:
                issues.append(LayoutIssue(
                    severity=SeverityLevel.HIGH,
                    page=name,
                    rule="canvas_within_plot_card",
                    widget=canvas_key,
                    message="画布高度超过所属 PlotCard 高度，存在裁切风险。",
                    expected=f"{canvas_key}.height <= {card_key}.height",
                    observed=f"{canvas_h} > {card_h}",
                ))

        for canvas_key, card_key in SMALL_PLOT_PAIRS.items():
            canvas = _geom(widgets, canvas_key)
            card = _geom(widgets, card_key)
            if not canvas or not card:
                continue
            canvas_h = _int(canvas.get("height"))
            card_h = _int(card.get("height"))
            margin = 34 if card_key in TITLED_PLOT_CARDS else 16
            if card_h < canvas_h + margin:
                issues.append(LayoutIssue(
                    severity=SeverityLevel.HIGH,
                    page=name,
                    rule="small_plot_card_min_height",
                    widget=card_key,
                    message="小图卡高度不足，无法容纳画布和标题/边距。",
                    expected=f"height >= canvas + {margin}",
                    observed=f"{card_h} < {canvas_h} + {margin}",
                ))

        for canvas_key in BSCAN_CANVASES:
            canvas = _geom(widgets, canvas_key)
            if not canvas:
                continue
            h = _int(canvas.get("height"))
            proportional_ok = bool(window_h and h / max(window_h, 1) >= 0.44)
            absolute_ok = h >= 360
            if not (absolute_ok or proportional_ok):
                issues.append(LayoutIssue(
                    severity=SeverityLevel.HIGH,
                    page=name,
                    rule="processing_bscan_min_height",
                    widget=canvas_key,
                    message="测线处理主 B-scan 高度不足。",
                    expected="height >= 360 or height/window >= 0.44",
                    observed=f"height={h}, window_height={window_h}",
                ))


        for canvas_key in TARGET_BSCAN_CANVASES:
            canvas = _geom(widgets, canvas_key)
            if not canvas:
                continue
            h = _int(canvas.get("height"))
            proportional_ok = bool(window_h and h / max(window_h, 1) >= 0.30)
            absolute_ok = h >= 250
            if not (absolute_ok or proportional_ok):
                issues.append(LayoutIssue(
                    severity=SeverityLevel.HIGH,
                    page=name,
                    rule="target_bscan_visual_height",
                    widget=canvas_key,
                    message="目标定位 B-scan 显示过扁，会影响标注舒适性。",
                    expected="height >= 250 or height/window >= 0.30",
                    observed=f"height={h}, window_height={window_h}",
                ))

        params = _geom(widgets, "processingParamsCard")
        if params:
            w = _int(params.get("width"))
            if w < 240:
                issues.append(LayoutIssue(
                    severity=SeverityLevel.HIGH,
                    page=name,
                    rule="processing_params_width_min",
                    widget="processingParamsCard",
                    message="测线处理参数栏过窄，连续处理按钮和数值输入框会被挤压。",
                    expected="width >= 240",
                    observed=f"width={w}",
                ))
            if w > 300:
                issues.append(LayoutIssue(
                    severity=SeverityLevel.HIGH,
                    page=name,
                    rule="processing_params_width_max",
                    widget="processingParamsCard",
                    message="测线处理参数栏过宽，会挤压 B-scan 主工作区。",
                    expected="width <= 300",
                    observed=f"width={w}",
                ))

        for key in (
            "processingExecuteStepButton",
            "processingUndoStepButton",
            "processingResetChainButton",
            "processingCompareButton",
            "processingSaveResultButton",
        ):
            btn = _geom(widgets, key)
            if not btn:
                continue
            h = _int(btn.get("height"))
            min_h = 28 if key in {"processingExecuteStepButton", "processingSaveResultButton"} else 24
            if h < min_h:
                issues.append(LayoutIssue(
                    severity=SeverityLevel.HIGH,
                    page=name,
                    rule="processing_chain_button_min_height",
                    widget=key,
                    message="连续处理按钮高度不足，可能出现文字压缩或点击困难。",
                    expected=f"height >= {min_h}",
                    observed=f"height={h}",
                ))

        continuous = _geom(widgets, "processingContinuousCard")
        if continuous:
            h = _int(continuous.get("height"))
            if h < 150:
                issues.append(LayoutIssue(
                    severity=SeverityLevel.HIGH,
                    page=name,
                    rule="processing_continuous_card_min_height",
                    widget="processingContinuousCard",
                    message="连续处理操作卡高度不足，按钮区存在重叠/裁切风险。",
                    expected="height >= 150",
                    observed=f"height={h}",
                ))

        if window_h:
            max_bottom = int(window_h * 0.22) + 1
            for key in BOTTOM_CARDS:
                card = _geom(widgets, key)
                if not card:
                    continue
                h = _int(card.get("height"))
                if h > max_bottom:
                    issues.append(LayoutIssue(
                        severity=SeverityLevel.HIGH,
                        page=name,
                        rule="bottom_region_height_max",
                        widget=key,
                        message="底部辅助区高度超过页面高度 22%，可能挤压主工作区。",
                        expected=f"height <= {max_bottom}",
                        observed=f"height={h}, window_height={window_h}",
                    ))

    issue_payload = [issue.to_dict() for issue in issues]
    return {
        "schema": "mygpr.layout_check.v1",
        "ok": not issues,
        "issue_count": len(issues),
        "pages_checked": pages_checked,
        "rules": [
            "canvas_within_plot_card",
            "small_plot_card_min_height",
            "processing_bscan_min_height",
            "processing_params_width_min",
            "processing_params_width_max",
            "processing_chain_button_min_height",
            "processing_continuous_card_min_height",
            "target_bscan_visual_height",
            "bottom_region_height_max",
        ],
        "issues": issue_payload,
    }


__all__ = ["LayoutIssue", "SeverityLevel", "check_layout_diagnostics_payload"]
