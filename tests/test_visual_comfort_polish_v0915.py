from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ui.field_panels.layout_metrics import layout_metrics_for
from ui.field_panels.table_utils import FieldTableMixin
from ui.field_panels.visual_comfort_rules import check_visual_comfort_payload

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if isinstance(app, QApplication):
        _APP = app
        return app
    _APP = QApplication([])
    return _APP


class _Host(FieldTableMixin):
    compact_mode = True


def test_table_fill_uses_light_backgrounds_for_highlight_and_alternating_rows() -> None:
    _app()
    host = _Host()
    table = host._table(["测线", "状态"], 3)
    host._fill_table(table, [("L01", "已导入"), ("L02", "待处理"), ("L03", "● 通过")], highlight_row=1)

    assert table.focusPolicy().name == "NoFocus"
    assert table.item(0, 0).background().color().name().lower() == "#ffffff"
    assert table.item(1, 0).background().color().name().lower() == "#ddf4f7"
    assert table.item(2, 0).background().color().name().lower() == "#ffffff"
    assert table.item(1, 1).foreground().color().name().lower() == "#b7791f"
    table.close()


def test_visual_comfort_rules_accept_polished_capture_geometry() -> None:
    payload = {
        "pages": [
            {
                "workspace_key": "interpretation",
                "widgets": {
                    "targetBscanCanvas": {"width": 1120, "height": 280},
                    "targetBscanCard": {"width": 1140, "height": 520},
                },
            },
            {
                "workspace_key": "delivery",
                "widgets": {
                    "deliveryReportPreviewCard": {"width": 1080, "height": 530},
                    "deliveryReportCover": {"width": 320, "height": 460},
                    "deliveryReportToc": {"width": 500, "height": 460},
                },
            },
            {
                "workspace_key": "spatial",
                "widgets": {
                    "spatialMapCanvas": {"width": 980, "height": 360},
                    "spatialAuxSidePanel": {"width": 350, "height": 510},
                },
            },
        ]
    }
    report = check_visual_comfort_payload(payload)
    assert report["ok"] is True
    assert report["issue_count"] == 0


def test_visual_comfort_rules_reject_flat_target_bscan_and_toc_dominance() -> None:
    payload = {
        "pages": [
            {
                "workspace_key": "interpretation",
                "widgets": {
                    "targetBscanCanvas": {"width": 1120, "height": 138},
                    "targetBscanCard": {"width": 1140, "height": 500},
                },
            },
            {
                "workspace_key": "delivery",
                "widgets": {
                    "deliveryReportPreviewCard": {"width": 1080, "height": 530},
                    "deliveryReportCover": {"width": 210, "height": 460},
                    "deliveryReportToc": {"width": 700, "height": 460},
                },
            },
        ]
    }
    report = check_visual_comfort_payload(payload)
    assert report["ok"] is False
    rules = {issue["rule"] for issue in report["issues"]}
    assert "target_bscan_comfort_height" in rules
    assert "report_cover_visual_weight" in rules
    assert "report_toc_not_dominant" in rules


def test_layout_metrics_include_target_and_larger_report_cover() -> None:
    app = _app()
    # A temporary QWidget is enough; metrics fall back to screen geometry in offscreen mode.
    from PyQt6.QtWidgets import QWidget

    host = QWidget()
    host.compact_mode = True  # type: ignore[attr-defined]
    metrics = layout_metrics_for(host)
    assert metrics.target_bscan_h >= 260
    assert metrics.target_table_max_h >= 112
    assert metrics.target_info_w <= 235
    assert metrics.delivery_cover_w >= 300
    host.close()
