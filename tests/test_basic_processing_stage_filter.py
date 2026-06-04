import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ui.gui_basic_flow import BasicFlowPage
from core.methods_registry import get_method_category

_APP = None


def _app():
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP = app
    return app


def test_processing_page_stage_filter_replaces_static_flow_card():
    _app()
    page = BasicFlowPage()
    assert not hasattr(page, "_flow_step_chips")
    assert hasattr(page, "_stage_filter_buttons")
    assert "all" in page._stage_filter_buttons
    assert page._stage_filter_buttons["all"].isChecked()


def test_stage_filter_limits_visible_method_categories():
    _app()
    page = BasicFlowPage()
    page.set_method_stage_filter("suppress")
    assert page.method_keys
    categories = {get_method_category(key) for key in page.method_keys}
    assert categories <= page.METHOD_STAGE_CATEGORIES["suppress"]
    assert any(key in page.method_keys for key in ["median_background_2D", "svd_bg", "subtracting_average_2D"])


def test_set_method_by_key_switches_stage_when_filtered_out():
    _app()
    page = BasicFlowPage()
    page.set_method_stage_filter("suppress")
    page.set_method_by_key("agcGain")
    assert page.get_current_method_key() == "agcGain"
    assert page._active_method_stage == "gain"
