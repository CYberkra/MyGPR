"""Shared component metrics and public interfaces for the workbench."""
import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402
from ui import constants  # noqa: E402


def test_shared_hit_areas_and_collapse_metrics_are_consistent():
    from ui.page_scaffold import SIDE_COLLAPSE_WIDTH
    from ui.widgets.file_tree_panel import FileTreePanel

    assert constants.TOOL_BTN_SIZE >= (36, 36)
    assert constants.TOOL_BTN_COMPACT >= (32, 32)
    assert SIDE_COLLAPSE_WIDTH == constants.DOCK_COLLAPSED_WIDTH
    assert constants.DOCK_COLLAPSED_WIDTH >= 32
    assert constants.FILE_TREE_PANEL_WIDTH == 320
    panel = FileTreePanel()
    assert panel._expanded_width == constants.FILE_TREE_PANEL_WIDTH
    panel.deleteLater()


def test_card_surface_has_a_single_token_source():
    from ui.design_tokens import tokens

    assert tokens(False)['surface_card'] == '#ffffff'
    assert tokens(True)['surface_card'] == '#2d2e32'


def test_job_table_exposes_safe_row_actions(qapp):
    from ui.widgets.job_widgets import JobTable

    table = JobTable()
    table.upsert_job('J1', '处理')
    assert table.context_table() is not None
    assert table.job_id_at(0) == 'J1'
    assert table.job_snapshot('J1')['title'] == '处理'
    assert table.job_snapshot('J1')['status'] == 'queued'
    assert table.can_cancel('J1')
    table.set_status('J1', 'completed')
    assert not table.can_cancel('J1')
    table.set_status('J1', 'running')
    cancelled = []
    table.cancel_requested.connect(cancelled.append)
    table.cancel_job('J1')
    assert cancelled == ['J1']
    table.deleteLater()


def test_chain_strip_empty_hint_has_a_public_setter(qapp):
    from ui.widgets.chain_strip import ChainStrip

    strip = ChainStrip()
    strip.set_empty_hint('＋ 添加处理步骤')
    assert strip._empty_hint.text() == '＋ 添加处理步骤'
    strip.deleteLater()


def test_panel_state_supports_single_side_and_priority_order(qapp):
    from ui.page_scaffold import PanelStateMixin
    from ui.widgets.collapsible_panel import CollapsiblePanel

    class SingleSidePage(PanelStateMixin, QWidget):
        _PANEL_STATE_PREFIX = 'test_single'
        _AUTO_COLLAPSE_ORDER = ('right',)

        def __init__(self):
            super().__init__()
            self.setLayout(QVBoxLayout())
            self._sm = None
            self._right_panel = CollapsiblePanel(expand_width=320)

    page = SingleSidePage()
    page.resize(500, 300)
    page._auto_collapse_side_panels()
    assert page._right_panel.is_collapsed()
    assert page.panel_states() == {'right': True}
    page.deleteLater()


def test_spatial_priority_collapses_detail_before_layers(qapp):
    from ui.page_scaffold import PanelStateMixin
    from ui.widgets.collapsible_panel import CollapsiblePanel

    class TwoSidePage(PanelStateMixin, QWidget):
        _PANEL_STATE_PREFIX = 'test_spatial'
        _AUTO_COLLAPSE_ORDER = ('right', 'left')

        def __init__(self):
            super().__init__()
            self.setLayout(QVBoxLayout())
            self._sm = None
            self._left_panel = CollapsiblePanel('left', expand_width=320)
            self._right_panel = CollapsiblePanel('right', expand_width=360)

    page = TwoSidePage()
    page.resize(800, 300)
    page._auto_collapse_side_panels()
    assert page._right_panel.is_collapsed()
    assert not page._left_panel.is_collapsed()
    page.deleteLater()
