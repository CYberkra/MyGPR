"""Settings isolation used by native desktop and installer QA."""
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
