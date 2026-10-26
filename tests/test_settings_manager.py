"""Settings isolation used by native desktop and installer QA."""
import json
import threading
from unittest.mock import patch

import pytest

from ui import settings_manager as module
from ui.settings_manager import SettingsManager


def test_environment_selects_isolated_settings(tmp_path, monkeypatch):
    target = tmp_path / 'qa' / 'settings.json'
    monkeypatch.setenv('MYGPR_SETTINGS_FILE', str(target))
    settings = SettingsManager()
    assert settings.settings_file == str(target)
    settings.set('recent_projects', ['C:/qa/project'])
    assert settings.save()
    assert SettingsManager().get('recent_projects') == ['C:/qa/project']


def test_explicit_settings_path_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('MYGPR_SETTINGS_FILE', str(tmp_path / 'environment.json'))
    target = tmp_path / 'explicit.json'
    assert SettingsManager(str(target)).settings_file == str(target)


@pytest.mark.parametrize('key,bad,expected', [
    ('max_workers', None, 2), ('max_workers', 'not-a-number', 2),
    ('max_workers', float('inf'), 2), ('max_workers', 1000000, 2),
    ('default_dielectric', None, 9.0), ('default_dielectric', float('nan'), 9.0),
    ('default_dielectric', -2, 9.0),
    ('project_root', {}, module.constants.DEFAULT_PROJECT_ROOT),
    ('pinned_recent_projects', [{'path': 'broken'}], []),
    ('auto_prefetch_basemap', 'false', False),
    ('auto_prefetch_basemap', 0, False),
    ('bscan_colorbar_visible', 'no', False),
    ('processing_left_collapsed', '1', True),
])
def test_invalid_known_settings_recover_individually(tmp_path, key, bad, expected):
    target = tmp_path / 'settings.json'
    target.write_text(json.dumps({key: bad, 'extension': {'preserve': 1}}), encoding='utf-8')
    settings = SettingsManager(str(target))
    assert settings.get(key) == expected
    assert settings.get('extension') == {'preserve': 1}
    assert target.exists()  # one invalid field must not discard the whole file


def test_interleaved_settings_writers_never_publish_partial_json(tmp_path):
    target = tmp_path / 'settings.json'
    first, second = SettingsManager(str(target)), SettingsManager(str(target))
    first.set('extension', 'short')
    second.set('extension', 'long' * 2000)
    entered, release = threading.Event(), threading.Event()
    outcomes = []
    original_dump = json.dump
    original_dumps = json.dumps

    def before_first_write():
        if threading.current_thread().name == 'settings-first-writer':
            entered.set()
            assert release.wait(5)

    def controlled_dump(*args, **kwargs):
        before_first_write()
        return original_dump(*args, **kwargs)

    def controlled_dumps(*args, **kwargs):
        before_first_write()
        return original_dumps(*args, **kwargs)

    thread = threading.Thread(target=lambda: outcomes.append(first.save()),
                              name='settings-first-writer')
    with patch.object(json, 'dump', controlled_dump), patch.object(json, 'dumps', controlled_dumps):
        thread.start()
        try:
            assert entered.wait(5)
            second_saved = second.save()
        finally:
            release.set()
            thread.join(5)
    assert not thread.is_alive()
    assert second_saved and outcomes == [True]
    payload = json.loads(target.read_text(encoding='utf-8'))
    assert payload['extension'] in {'short', 'long' * 2000}
    assert not list(tmp_path.glob('*.tmp'))


def test_serialization_error_preserves_previous_settings(tmp_path):
    target = tmp_path / 'settings.json'
    settings = SettingsManager(str(target))
    assert settings.save()
    previous = target.read_bytes()
    settings.set('extension', object())
    assert settings.save() is False
    assert target.read_bytes() == previous
    assert not list(tmp_path.glob('*.tmp'))


def test_replace_failure_preserves_file_and_cleans_own_temporary(tmp_path, monkeypatch):
    from core import storage_primitives

    target = tmp_path / 'settings.json'
    settings = SettingsManager(str(target))
    assert settings.save()
    previous = target.read_bytes()
    settings.set('extension', 'new')

    def deny_replace(*args):
        raise PermissionError('settings destination is locked')

    monkeypatch.setattr(storage_primitives.os, 'replace', deny_replace)
    assert settings.save() is False
    assert target.read_bytes() == previous
    assert not list(tmp_path.glob('*.tmp'))
