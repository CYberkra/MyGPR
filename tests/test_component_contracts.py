"""Shared component metrics and public interfaces for the workbench."""
import pytest

pytest.importorskip('PyQt6')

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
