"""Runtime path overrides must isolate diagnostics from user application data."""
from pathlib import Path

from core.app_paths import get_logs_dir


def test_explicit_log_directory_is_used_and_created(tmp_path, monkeypatch):
    target = tmp_path / 'qa' / 'logs'
    monkeypatch.setenv('MYGPR_LOG_DIR', str(target))
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'user-data'))
    assert Path(get_logs_dir()) == target
    assert target.is_dir()
    assert not (tmp_path / 'user-data' / 'MyGPR').exists()


def test_log_directory_defaults_to_user_app_data(tmp_path, monkeypatch):
    monkeypatch.delenv('MYGPR_LOG_DIR', raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    assert Path(get_logs_dir()) == tmp_path / 'MyGPR' / 'output' / 'logs'
