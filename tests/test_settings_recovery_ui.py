"""Malformed individual preferences must not prevent native pages loading."""
import json

import pytest

pytest.importorskip('PyQt6')
pytest.importorskip('qfluentwidgets')

from ui.pages.home_page import HomePage
from ui.pages.settings_page import SettingsPage
from ui.settings_manager import SettingsManager


def test_bad_preferences_can_be_replayed_into_settings_and_startup_pages(qapp, tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({
        'max_workers': None, 'default_dielectric': {},
        'pinned_recent_projects': [{'path': 'invalid'}],
        'recent_projects': [str(tmp_path)], 'bscan_tvg_db': float('inf'),
        'auto_prefetch_basemap': 'false', 'extension': {'preserve': True},
    }), encoding='utf-8')
    settings = SettingsManager(str(path))
    page, startup = SettingsPage(), HomePage()
    try:
        page.load_settings(settings.get_all())
        startup.set_settings_manager(settings)
        assert page.settings()['max_workers'] == 2
        assert page.settings()['default_dielectric'] == 9.0
        assert page.settings()['auto_prefetch_basemap'] is False
        assert startup._recent_list.count() == 1
        assert settings.get('extension') == {'preserve': True}
    finally:
        page.deleteLater()
        startup.deleteLater()
        qapp.processEvents()
