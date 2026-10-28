"""Canonical terminology, discoverability, and display-only legacy draft upgrades."""
from copy import deepcopy

import pytest


def test_catalog_uses_one_name_per_stable_id():
    from core.method_registry_metadata import METHOD_METADATA
    from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS
    from mygpr.infrastructure.processing.native_adapter import NativeProcessingCatalog
    from ui.desktop_backend_facade import method_catalog

    entries = {entry.method_id: entry for entry in method_catalog()}
    assert set(entries) == set(METHOD_METADATA) == set(NATIVE_ALGORITHMS)
    names = [item['display_name'] for item in METHOD_METADATA.values()]
    assert len(names) == len(set(names))
    catalog = NativeProcessingCatalog()
    for mid, meta in METHOD_METADATA.items():
        assert entries[mid].display_name == catalog.get(mid).name == meta['display_name']
        assert entries[mid].description == meta['description']
        assert meta['description'].strip()


def test_trace_cut_is_available_with_clear_application_parameters():
    from ui.desktop_backend_facade import method_catalog

    entry = next(item for item in method_catalog() if item.method_id == 'trace_cut')
    assert entry.display_name == '测道裁剪'
    assert entry.category == 'time_correction'
    assert [(item['name'], item['label'], item['default']) for item in entry.parameter_schema] == [
        ('trace_start', '起始道号（从1开始）', 1),
        ('trace_end', '结束道号（0=末道）', 0),
    ]


def test_gain_labels_describe_the_existing_curves_instead_of_claiming_sec():
    from core.method_registry_metadata import METHOD_METADATA

    assert METHOD_METADATA['compensatingGain']['display_name'] == '线性分贝增益'
    assert METHOD_METADATA['sec_gain']['display_name'] == '时间幂函数增益'
    assert 'SEC 增益' in METHOD_METADATA['sec_gain']['description']
    assert '实验' in METHOD_METADATA['rtm_migration']['description']
    assert '零偏移' in METHOD_METADATA['rtm_migration']['description']


@pytest.fixture
def browser(qapp):
    from ui.controllers.processing_controller import build_method_dicts
    from ui.desktop_backend_facade import method_catalog
    from ui.widgets.method_browser import MethodBrowser

    widget = MethodBrowser()
    widget.set_methods(build_method_dicts(method_catalog()))
    yield widget
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize(('query', 'expected'), [
    ('SEC 增益', 'sec_gain'),
    ('sec增益', 'sec_gain'),
    ('时间幂函数增益', 'sec_gain'),
    ('Wavelet 2D 去噪', 'wavelet_2d'),
    ('wavelet2d', 'wavelet_2d'),
    ('二维小波', 'wavelet_2d'),
    ('低频漂移矫正(Dewow)', 'dewow'),
    ('道向 Savitzky-Golay 平滑', 'trace_savgol_filter'),
    ('SG', 'trace_savgol_filter'),
    ('道间多项式平滑（S-G）', 'trace_savgol_filter'),
    ('Moving Average', 'running_average_2D'),
    ('Median Filter', 'trace_median_filter'),
    ('Hankel-SVD 去噪', 'hankel_svd'),
    ('Inverse-Q', 'inverse_q'),
    ('UAV 运动补偿 V2', 'motion_compensation_v2'),
    ('速度误差补偿', 'motion_compensation_speed'),
    ('RTM 反向时间迁移', 'rtm_migration'),
    ('希尔伯特', 'hilbert_envelope'),
    ('F-K', 'fk_filter'),
    ('偏移成像', 'kirchhoff_migration'),
])
def test_former_names_abbreviations_and_categories_remain_searchable(browser, query, expected):
    from PyQt6.QtCore import Qt

    browser._search.setText(query)
    found = []
    for i in range(browser._tree.topLevelItemCount()):
        group = browser._tree.topLevelItem(i)
        for j in range(group.childCount()):
            child = group.child(j)
            if not group.isHidden() and not child.isHidden():
                found.append(child.data(0, Qt.ItemDataRole.UserRole))
    assert expected in found
    assert browser._nomatch.isHidden()


def test_alias_search_still_adds_the_original_method_id(browser):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel

    browser._search.setText('SEC 增益')
    added = []
    browser.sig_add_requested.connect(added.append)
    for i in range(browser._tree.topLevelItemCount()):
        group = browser._tree.topLevelItem(i)
        for j in range(group.childCount()):
            child = group.child(j)
            if child.data(0, Qt.ItemDataRole.UserRole) == 'sec_gain':
                assert child.data(0, Qt.ItemDataRole.UserRole + 1) == '时间幂函数增益'
                assert 'SEC 增益' in child.toolTip(0)
                label = browser._tree.itemWidget(child, 0).findChild(QLabel)
                assert label.text() == '时间幂函数增益'
                assert label.toolTip() == child.toolTip(0)
                browser._on_double_clicked(child, 0)
    assert added == ['sec_gain']


def test_catalog_refresh_relabels_the_view_without_rewriting_a_draft(qapp):
    from ui.controllers.processing_controller import build_method_dicts
    from ui.desktop_backend_facade import method_catalog
    from ui.pages.processing_page import ProcessingPage

    page = ProcessingPage()
    steps = [
        {'method_id': 'sec_gain', 'label': 'SEC 增益',
         'params': {'power': 2.5, 'gain_max': 4.0}, 'enabled': False},
        {'method_id': 'plugin_private', 'label': '自定义步骤', 'params': {'p': 3}, 'enabled': True},
    ]
    page._pipeline_list.set_steps(deepcopy(steps))
    before = page._pipeline_list.steps()
    epoch = page._draft_edit_epoch
    changes = []
    page._pipeline_list.sig_changed.connect(lambda: changes.append(True))
    page.set_methods(build_method_dicts(method_catalog()))
    assert page._chain_strip._steps[0]['label'] == '时间幂函数增益'
    assert page._chain_strip._steps[1]['label'] == '自定义步骤'
    assert page._pipeline_list.steps() == before
    assert page._draft_edit_epoch == epoch
    assert changes == []
    page.close()
    page.deleteLater()
    qapp.processEvents()
