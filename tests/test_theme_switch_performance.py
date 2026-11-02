"""Theme switching repaints palettes without repolishing the entire widget tree."""
from __future__ import annotations

import pytest

pytest.importorskip("qfluentwidgets")

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
from qfluentwidgets import qconfig

from ui import constants
from ui import theme_helpers


def test_switch_keeps_global_qss_stable_and_existing_native_views_readable(qapp):
    theme_helpers.apply_theme(constants.THEME_LIGHT)
    table = QTableWidget(2, 2)
    table.setItem(0, 0, QTableWidgetItem("Preserved data"))
    table.setStyleSheet("QTableWidget {border-radius: 3px;}")
    table.show()
    qapp.processEvents()
    stylesheet = qapp.styleSheet()
    local_stylesheet = table.styleSheet()
    for theme, dark in ((constants.THEME_DARK, True), (constants.THEME_LIGHT, False)):
        theme_helpers.apply_theme(theme)
        qapp.processEvents()
        assert qapp.styleSheet() == stylesheet
        colors = theme_helpers.control_palette(dark)
        assert table.viewport().palette().color(QPalette.ColorRole.Base).name() == colors["table_base"]
        assert table.palette().color(QPalette.ColorRole.Text).name() == colors["table_text"]
        assert table.styleSheet() == local_stylesheet
        assert table.item(0, 0).text() == "Preserved data"
    table.close()


def test_theme_switch_uses_lazy_fluent_styles_and_keeps_existing_accent(qapp, monkeypatch):
    import qfluentwidgets

    theme_helpers.apply_theme(constants.THEME_LIGHT)
    qconfig.set(qconfig.themeColor, QColor(constants.ACCENT_SOLID), save=False)
    theme_calls, accent_calls = [], []
    original = theme_helpers.setTheme

    def theme_spy(*args, **kwargs):
        theme_calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(theme_helpers, "setTheme", theme_spy)
    monkeypatch.setattr(qfluentwidgets, "setThemeColor", lambda *args, **kwargs: accent_calls.append(args))
    theme_helpers.apply_theme(constants.THEME_DARK)
    assert theme_calls == [{"lazy": True}]
    assert accent_calls == []


def test_native_view_created_after_switch_uses_the_current_palette(qapp):
    theme_helpers.apply_theme(constants.THEME_DARK)
    table = QTableWidget(1, 1)
    table.show()
    qapp.processEvents()
    assert table.viewport().palette().color(QPalette.ColorRole.Base).name() == theme_helpers.control_palette(True)["table_base"]
    table.close()
    theme_helpers.apply_theme(constants.THEME_LIGHT)


def test_table_corner_header_changes_theme_with_the_other_headers(qapp):
    from PyQt6.QtWidgets import QAbstractButton

    theme_helpers.apply_theme(constants.THEME_LIGHT)
    table = QTableWidget(2, 2)
    table.show()
    qapp.processEvents()
    corner = next(widget for widget in table.findChildren(QAbstractButton) if widget.inherits("QTableCornerButton"))
    for theme, dark in ((constants.THEME_DARK, True), (constants.THEME_LIGHT, False)):
        theme_helpers.apply_theme(theme)
        qapp.processEvents()
        expected = theme_helpers.control_palette(dark)["table_header_bg"]
        assert corner.palette().color(QPalette.ColorRole.AlternateBase).name() == expected
        image = corner.grab().toImage()
        assert image.pixelColor(image.width() // 2, image.height() // 2).name() == expected
    table.close()
