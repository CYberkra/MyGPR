"""Compact display controls preserve the active input and pending draft."""
import pytest

pytest.importorskip('PyQt6')
from tests.test_processing_chain_editor import page  # noqa: F401


def test_source_path_toggle_only_changes_presentation(page, qapp):  # noqa: F811 - shared fixture
    page.select_artifact('B')
    page._add_method_to_pipeline('dewow')
    before = page.current_pipeline()
    assert not page._saved_path.isVisibleTo(page)
    page._source_path_btn.click()
    qapp.processEvents()
    assert page._saved_path.isVisibleTo(page)
    assert page.current_pipeline() == before
    page._source_path_btn.click()
    assert not page._saved_path.isVisibleTo(page)
    assert page.current_pipeline() == before


def test_compact_header_keeps_long_input_identity_in_tooltip(page, qapp):  # noqa: F811 - shared fixture
    page._sources['original']['title'] = '原始输入-' + '很长的测线名称' * 20
    page._refresh_ui()
    page.resize(800, 620)
    qapp.processEvents()
    assert page._sources['original']['title'] in page._input_label.toolTip()
    assert page._resources_btn.isVisibleTo(page)
    assert page._tools_btn.isVisibleTo(page)
    assert page._layout_combo.isVisibleTo(page)
    assert page.current_pipeline()['input_artifact_id'] == ''
