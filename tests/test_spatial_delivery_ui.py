"""Spatial export/delete application contracts."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QPoint


def test_project_service_delegates_spatial_export_and_recycle():
    from mygpr.application.project.service import ProjectService

    calls = []
    session = NS(
        export_spatial_result=lambda *args, **kwargs: calls.append(('export', args, kwargs)) or 'exported',
        delete_spatial_result=lambda result_id: calls.append(('delete', result_id)) or 'recycled',
    )
    service = ProjectService(NS())
    service._sessions['P'] = session
    assert service.export_spatial_result('P', 'R1', 'D:/out.zip', format_name='zip') == 'exported'
    assert service.delete_spatial_result('P', 'R1') == 'recycled'
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
