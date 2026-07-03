from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ui.field_panels.table_utils import FieldTableMixin


class _TableHost(FieldTableMixin):
    pass


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _hex_color(item) -> str:
    return item.foreground().color().name().lower()


def test_field_table_default_text_keeps_readable_dark_foreground() -> None:
    app = _app()
    host = _TableHost()
    assert app is not None
    table = host._table(["测线", "状态"], 0)
    host._fill_table(table, [("L03", "未定位"), ("L04", "--")], highlight_row=0)

    assert _hex_color(table.item(0, 0)) == "#243447"
    assert _hex_color(table.item(1, 1)) == "#243447"
    assert table.item(0, 0).background().color().name().lower() == "#ddf4f7"


def test_field_table_status_colors_remain_explicit_after_contrast_fix() -> None:
    app = _app()
    host = _TableHost()
    assert app is not None
    table = host._table(["状态"], 0)
    host._fill_table(table, [("● 已完成",), ("⚠ 待补充",), ("✕ 失败",)])

    assert _hex_color(table.item(0, 0)) == "#16a05d"
    assert _hex_color(table.item(1, 0)) == "#b7791f"
    assert _hex_color(table.item(2, 0)) == "#e5484d"
