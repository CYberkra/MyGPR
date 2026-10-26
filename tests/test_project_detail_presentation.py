"""Regression coverage for the selected-line details and local scroll surface."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui.design_tokens import color
from ui.pages.project_page import ProjectPage
from ui.pages.project_visuals import make_detail_scroll


def test_detail_scroll_does_not_paint_system_dark_background(qapp):
    parent = QWidget()
    palette = parent.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor('#f5f7fa'))
    parent.setPalette(palette)
    parent.setAutoFillBackground(True)
    layout = QVBoxLayout(parent)
    scroll = make_detail_scroll(QLabel('测线详情'), parent)
    layout.addWidget(scroll)
    # Reproduce a system-dark child palette inside the light workbench.
    content_palette = scroll.widget().palette()
    content_palette.setColor(QPalette.ColorRole.Window, QColor('#303030'))
    scroll.widget().setPalette(content_palette)
    parent.resize(640, 200)
    parent.show()
    qapp.processEvents()
    point = scroll.viewport().rect().bottomLeft()
    point.setX(point.x() + 30)
    point.setY(point.y() - 15)
    actual = parent.grab().toImage().pixelColor(scroll.viewport().mapTo(parent, point))
    parent.close()
    assert actual.name() == '#f5f7fa'


def _selected_page():
    page = ProjectPage()
    page.set_project_info(SimpleNamespace(project_id='details', read_only=False))
    page.set_lines([SimpleNamespace(
        line_id='L01', name='L01', sample_count=501, trace_count=709,
        data_quality='★★★★★')])
    page.select_line('L01')
    return page


def test_details_preserve_quality_and_complete_copyable_source(qapp):
    page = _selected_page()
    source = 'C:\\采集数据\\' + '长目录名称\\' * 15 + 'lineData_streaming.csv'
    page.set_line_source_path('L01', source)
    panel = page._line_detail
    assert [label.text() for label in panel._values] == ['L01', '501', '709', '★★★★★']
    assert panel._source_name.text() == 'lineData_streaming.csv'
    assert panel._source_path.text() == source
    assert panel._source_path.toolTip() == source
    assert panel._source_path.textFormat() == Qt.TextFormat.PlainText
    assert panel._source_path.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    page.close()


def test_selection_and_project_changes_clear_detail_source(qapp):
    page = _selected_page()
    assert page._line_detail._source_name.text() == '正在查询数据来源…'
    page.set_line_source_path('L01', None)
    assert page._line_detail._source_name.text() == '未记录数据来源'
    page.set_line_source_path('L01', 'C:\\data\\old.csv')
    page.set_project_info(SimpleNamespace(project_id='other', read_only=False))
    page.set_lines([])
    assert page._line_detail._source_path.text() == ''
    assert page._line_detail._source_name.text() == ''
    page.close()


@pytest.mark.parametrize('dark', [False, True, False])
def test_detail_theme_and_short_pane_keep_source_reachable(qapp, dark):
    page = _selected_page()
    page.set_line_source_path('L01', 'D:\\' + '野外勘探数据\\' * 35 + 'lineData.csv')
    page.apply_theme(dark)
    panel = page._line_detail
    scroll = panel.parentWidget().parentWidget().parentWidget()
    page.resize(1200, 700)
    page.show()
    qapp.processEvents()
    # Restrict the actual details viewport, as when the ledger splitter is dragged.
    scroll.setFixedSize(440, 100)
    qapp.processEvents()
    assert color('text_primary', dark) in panel._values[0].styleSheet()
    assert color('text_secondary', dark) in panel._source_path.styleSheet()
    assert not scroll.widget().autoFillBackground()
    assert not scroll.viewport().autoFillBackground()
    assert scroll.verticalScrollBar().maximum() > 0
    assert scroll.horizontalScrollBar().maximum() == 0
    scroll.ensureWidgetVisible(panel._source_path)
    qapp.processEvents()
    assert scroll.verticalScrollBar().value() > 0
    page.close()
