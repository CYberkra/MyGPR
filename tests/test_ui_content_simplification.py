"""Visible UI copy stays subordinate to the workbench layout."""
import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QLabel  # noqa: E402
from tests.test_processing_chain_editor import page as processing_page  # noqa: F401,E402


def _label_texts(widget):
    return [label.text() for label in widget.findChildren(QLabel)]


def test_startup_and_output_panel_drop_repeated_instructions(qapp):
    from ui.pages.home_page import HomePage
    from ui.widgets.output_panel import OutputPanel

    home = HomePage()
    assert '打开已有项目，或新建项目后导入测线。' not in _label_texts(home)
    panel = OutputPanel()
    assert not any('完整历史见' in text for text in _label_texts(panel))
    home.deleteLater()
    panel.deleteLater()


def test_processing_uses_state_text_instead_of_static_tutorials(processing_page):  # noqa: F811
    page = processing_page
    assert page._input_label.text().startswith('输入 · ')
    assert page._param_empty_hint.text() == '未选择处理步骤'
    assert page._estimate_label.text() == ''
    assert page._autotune_method_label.text() == ''
    assert page._autotune_result_label.text() == ''
    texts = _label_texts(page)
    assert '仅改变当前视图；不修改数据或处理步骤' not in texts
    assert '色阶、显示增益、坐标与导出可在画布右键调整' not in texts
    assert '数据来源与分支' not in texts
    assert '参数与调参' not in texts
    assert '待执行' not in texts
    tabs = [page._tools_panel.tabText(i) for i in range(page._tools_panel.count())]
    assert tabs == ['处理', '历史']
    assert page._execution_card.is_collapsed()
    assert page._autotune_card.is_collapsed()


def test_processing_display_and_fit_buttons_live_in_workspace_tab_bar(processing_page, monkeypatch):  # noqa: F811
    from PyQt6.QtCore import QPoint

    page = processing_page
    assert not hasattr(page, '_display_settings_btn')
    assert not hasattr(page, '_fit_view_btn')
    workspace = page._workspace
    group = workspace.group_widget()
    for button, name in ((group.display_settings, '当前视图显示设置'),
                         (group.fit_view, '适应全图')):
        assert button.isVisible()
        assert button.accessibleName() == name
        assert workspace.isAncestorOf(button)
        # Same tab-bar row as the group "more" button, ordered before it.
        assert button.geometry().center().y() == group.more.geometry().center().y()
        assert button.mapTo(group, QPoint(0, 0)).x() < group.more.mapTo(group, QPoint(0, 0)).x()
        # Above the group's B-Scan view, not on a separate page-level row.
        assert button.mapTo(group, QPoint(0, 0)).y() < group.view.mapTo(group, QPoint(0, 0)).y()
    shown = []
    monkeypatch.setattr(group.view, 'show_display_menu', shown.append)
    group.display_settings.click()
    assert len(shown) == 1
    calls = []
    monkeypatch.setattr(group.view, 'fit_all', lambda: calls.append(1))
    group.fit_view.click()
    assert calls == [1]


def test_interpretation_has_one_session_status_and_one_point_count(qapp):
    from ui.pages.interpretation_page import InterpretationPage

    page = InterpretationPage()
    assert not hasattr(page, '_info_label')
    texts = _label_texts(page)
    assert '标注工具' not in texts
    assert '剖面标注' not in texts
    assert not any('左键点击拾取标注点' in text for text in texts)
    assert page._velocity_result_label.text() == ''
    page.deleteLater()


def test_spatial_and_jobs_remove_redundant_region_titles(qapp):
    from ui.pages.jobs_page import JobsPage
    from ui.pages.spatial_page import SpatialPage

    spatial = SpatialPage()
    texts = _label_texts(spatial)
    assert '空间视图' not in texts
    assert '来源:' not in texts
    assert not any('统计仅基于已载入' in text for text in texts)
    jobs = JobsPage()
    assert '任务记录' not in _label_texts(jobs)
    spatial.deleteLater()
    jobs.deleteLater()


def test_delivery_uses_controls_as_the_primary_affordance(qapp):
    from ui.pages.delivery_page import DeliveryPage

    page = DeliveryPage()
    assert not hasattr(page, '_open_dir_hint')
    assert page._report_name_edit.placeholderText() == '留空自动命名'
    assert page._restore_name_edit.placeholderText() == '留空自动命名'
    assert page._incremental_check.text() == '增量备份'
    assert '选择测线（可多选）' not in _label_texts(page)
    page.deleteLater()


def test_project_tree_does_not_repeat_global_project_identity(qapp):
    from ui.widgets.file_tree_panel import FileTreePanel

    panel = FileTreePanel()
    assert panel._project_label.isHidden()
    panel.deleteLater()
