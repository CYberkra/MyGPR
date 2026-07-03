from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

from ui.field_panels.widgets import CollapsibleSidePanel, PlotCard

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if isinstance(app, QApplication):
        _APP = app
        return app
    _APP = QApplication([])
    return _APP


def test_collapsible_side_panel_toggles_width_and_content_visibility() -> None:
    app = _app()
    content = QLabel("参数区")
    panel = CollapsibleSidePanel(title="处理设置", content=content, expanded_width=210, collapsed_width=34)
    panel.show()
    app.processEvents()
    assert panel.is_expanded() is True
    assert content.isVisible() is True
    assert panel.minimumWidth() == 210
    assert panel.maximumWidth() == 210

    panel.toggle()
    app.processEvents()
    assert panel.is_expanded() is False
    assert content.isVisible() is False
    assert panel.minimumWidth() == 34
    assert panel.maximumWidth() == 34

    panel.toggle()
    app.processEvents()
    assert panel.is_expanded() is True
    assert content.isVisible() is True
    assert panel.maximumWidth() == 210
    panel.close()


def test_plot_card_expand_button_is_added_without_changing_canvas_contract() -> None:
    _app()

    def draw(_canvas) -> None:
        return None

    card = PlotCard("主图", height=180, expand_title="主图放大", expand_callback=draw)
    buttons = card.findChildren(QPushButton)
    assert any(button.toolTip().startswith("放大查看") for button in buttons)
    assert card.canvas.minimumHeight() == 180
    assert card.canvas.maximumHeight() == 180
    assert card.minimumHeight() >= card.canvas.maximumHeight()
