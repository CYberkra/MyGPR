"""Spatial export/delete application contracts."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('PyQt6')


def test_project_service_delegates_spatial_export_and_permanent_delete():
    from mygpr.application.project.service import ProjectService

    calls = []
    session = NS(
        summary=NS(read_only=False),
        export_spatial_result=lambda *args, **kwargs: calls.append(('export', args, kwargs)) or 'exported',
        delete_spatial_result=lambda result_id: calls.append(('delete', result_id)) or 'deleted',
    )
    service = ProjectService(NS())
    service._sessions['P'] = session
    assert service.export_spatial_result('P', 'R1', 'D:/out.zip', format_name='zip') == 'exported'
    assert service.delete_spatial_result('P', 'R1') == 'deleted'
    assert calls[0][0] == 'export'
    assert calls[1] == ('delete', 'R1')


def test_delivery_spatial_menu_enables_real_actions(qapp):
    from ui.pages.delivery_page import DeliveryPage

    page = DeliveryPage()
    result = NS(result_id='R1', name='成果', line_ids=('L01',), created_at='2026-10-26')
    page.set_spatial_results([result])
    rect = page._spatial_table.visualItemRect(page._spatial_table.item(0, 0))
    menu = page._spatial_result_menu(rect.center())
    actions = {action.data(): action for action in menu.actions() if action.data()}
    assert actions['export-result'].isEnabled()
    assert actions['delete-result'].isEnabled()
    assert actions['set-current-result'].isEnabled()
    menu.close()
    page.close()
    page.deleteLater()


def test_delivery_shows_spatial_warnings_and_missing_surface(qapp):
    from ui.pages.delivery_page import DeliveryPage

    page = DeliveryPage()
    page.set_spatial_results([NS(result_id='R1', name='单线', line_ids=('L01',),
        created_at='2026-10-26', status='generated_with_warnings',
        files={'profiles_csv': 'profiles.csv', 'interfaces_geojson': 'interfaces.geojson'},
        summary={'preflight_warnings': ['测线数量不足，连续界面曲面不会生成。']})])
    status = page._spatial_table.item(0, 3)
    assert status is not None
    assert '警告' in status.text()
    assert '曲面' in status.text()
    assert '测线数量不足' in status.toolTip()


def test_delivery_distinguishes_control_points_from_surface(qapp):
    from ui.pages.delivery_page import DeliveryPage
    page = DeliveryPage()
    page.set_spatial_results([NS(result_id='R1', name='点集', line_ids=('L01',),
        files={'surface_control_points_geojson': 'surface_control_points.geojson'},
        options={'surface_generated': False, 'surface_control_points_generated': True})])
    status = page._spatial_table.item(0, 3).text()
    assert '控制点' in status and '未生成曲面' in status
    page.close()


def test_spatial_result_opens_right_sidebar_and_selects_exact_version(qapp):
    from ui.pages.spatial_page import SpatialPage

    page = SpatialPage()
    try:
        tools = page.delivery_tools
        tools.set_spatial_results([
            {'result_id': 'A', 'name': '范围', 'revision': 1, 'line_ids': ['L01']},
            {'result_id': 'B', 'name': '范围', 'revision': 2, 'line_ids': ['L02']},
        ])
        located = []
        tools.spatial_locate_requested.connect(located.append)
        assert page.open_spatial_result('B')
        assert page._spatial_tools_stack.currentWidget() is tools
        assert tools._selected_spatial_id() == 'B'
        assert located == [['L02']]
        assert not page.open_spatial_result('missing')
        assert tools._selected_spatial_id() == 'B'
    finally:
        page.close()
        page.deleteLater()
