"""Distribution trust, version ordering and the native ABI without network/UI."""
import base64
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from mygpr.infrastructure.updates.configuration import UpdateConfiguration, https_url, windows_build_version
from mygpr.infrastructure.updates.winsparkle import WinSparkle
from scripts.publish_windows_update import NAMESPACE, make_appcast, public_key

KEY = base64.b64encode(bytes(range(32))).decode()


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / 'updates.json'
    path.write_text(json.dumps({'schema': 'mygpr.windows_updates.v1', 'public_key': KEY,
                               'stable_feed': 'https://example.com/appcast.xml', 'preview_feed': ''}))
    return path


def test_build_configuration_and_unavailable_preview(config_file):
    config = UpdateConfiguration.load(config_file)
    assert config.public_key == KEY
    assert config.feed('stable') == 'https://example.com/appcast.xml'
    with pytest.raises(ValueError):
        config.feed('preview')
    assert UpdateConfiguration(KEY, config.stable_feed, 'https://example.com/preview.xml').feed('preview').endswith('preview.xml')


@pytest.mark.parametrize('change', [dict(schema='other'), dict(public_key=''), dict(public_key='invalid'),
                                  dict(public_key=base64.b64encode(b'short').decode()),
                                  dict(stable_feed='http://example.com'), dict(preview_feed='file:///feed')])
def test_rejects_untrusted_configuration(config_file, change):
    payload = json.loads(config_file.read_text())
    payload.update(change)
    config_file.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        UpdateConfiguration.load(config_file)


@pytest.mark.parametrize('url', ['http://example.com', 'file:///tmp/a', 'https://u:p@example.com',
                               'https://example.com/#fragment', 'https://'])
def test_requires_https_without_credentials_or_fragment(url):
    with pytest.raises(ValueError):
        https_url(url)


def test_prerelease_build_numbers_sort_before_final():
    versions = ['0.11.4', '0.12.0a1', '0.12.0b1', '0.12.0rc1', '0.12.0', '0.12.1a1']
    builds = [tuple(map(int, windows_build_version(v).split('.'))) for v in versions]
    assert builds == sorted(set(builds))
    assert windows_build_version('0.12.0') == '0.12.0.60000'


@pytest.mark.parametrize('version', ['v0.1.0', '0.1', '0.1.0.dev1', '65536.1.0', '0.1.0rc10000'])
def test_unsupported_windows_version_is_rejected(version):
    with pytest.raises(ValueError):
        windows_build_version(version)


class NativeFunction:
    def __init__(self, result=None):
        self.result, self.calls = result, []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


def native_library(accepted=1):
    names = ('set_app_details', 'set_app_build_version', 'set_lang', 'set_appcast_url',
             'set_eddsa_public_key', 'set_automatic_check_for_updates', 'init', 'cleanup',
             'check_update_without_ui', 'check_update_with_ui',
             *(f'set_{name}_callback' for name in ('error', 'did_find_update', 'did_not_find_update',
                'update_cancelled', 'update_dismissed', 'can_shutdown', 'shutdown_request')))
    return SimpleNamespace(**{'win_sparkle_' + name: NativeFunction(accepted if name == 'set_eddsa_public_key' else None)
                              for name in names})


def test_native_abi_callbacks_remain_alive_through_cleanup():
    library, events, shutdowns = native_library(), [], []
    client = WinSparkle(Path('not-loaded.dll'), library=library)
    with pytest.raises(RuntimeError):
        client.check()
    client.start(version='0.12.0', public_key=KEY, feed='https://example.com/feed',
                 event=events.append, can_shutdown=lambda: 1, shutdown=lambda: shutdowns.append(True))
    assert library.win_sparkle_set_eddsa_public_key.restype is ctypes.c_int
    assert library.win_sparkle_set_app_build_version.calls == [('0.12.0.60000',)]
    assert library.win_sparkle_set_automatic_check_for_updates.calls == [(0,)]
    for name in ('error', 'did_find_update', 'did_not_find_update', 'update_cancelled', 'update_dismissed'):
        getattr(library, f'win_sparkle_set_{name}_callback').calls[0][0]()
    assert events == ['error', 'available', 'current', 'cancelled', 'dismissed']
    assert library.win_sparkle_set_can_shutdown_callback.calls[0][0]() == 1
    library.win_sparkle_set_shutdown_request_callback.calls[0][0]()
    assert shutdowns == [True]
    client.check()
    client.check(silent=True)
    assert library.win_sparkle_check_update_with_ui.calls == [()]
    assert library.win_sparkle_check_update_without_ui.calls == [()]
    client.cleanup()
    client.cleanup()
    assert library.win_sparkle_cleanup.calls == [()]
    assert len(client._callbacks) == 7


def test_native_rejected_key_never_initializes():
    library = native_library(accepted=0)
    client = WinSparkle(Path('unused.dll'), library=library)
    with pytest.raises(ValueError):
        client.start(version='0.12.0', public_key=KEY, feed='https://example.com',
                     event=lambda value: None, can_shutdown=lambda: 0, shutdown=lambda: None)
    client.cleanup()
    assert not library.win_sparkle_init.calls
    assert not library.win_sparkle_cleanup.calls


def test_appcast_has_verifiable_signature_platform_and_visible_install_progress():
    signature = base64.b64encode(bytes(range(64))).decode()
    xml = make_appcast(version='0.12.0rc1', url='https://example.com/setup.exe',
                       notes_url='https://example.com/notes', signature=signature, length=1234)
    item = ET.fromstring(xml).find('channel/item')
    enclosure = item.find('enclosure')
    assert enclosure.attrib[f'{{{NAMESPACE}}}edSignature'] == signature
    assert enclosure.attrib[f'{{{NAMESPACE}}}version'] == '0.12.0.30001'
    assert enclosure.attrib[f'{{{NAMESPACE}}}os'] == 'windows-x64'
    assert '/SILENT ' in enclosure.attrib[f'{{{NAMESPACE}}}installerArguments']
    assert '/MYGPRUPDATE=1' in enclosure.attrib[f'{{{NAMESPACE}}}installerArguments']
    with pytest.raises(ValueError):
        make_appcast(version='0.12.0', url='https://example.com/setup.zip', notes_url='https://example.com',
                     signature=signature, length=1)


def test_public_key_reader_never_accepts_ambiguous_output(monkeypatch):
    monkeypatch.setattr('scripts.publish_windows_update.subprocess.check_output', lambda *a, **k: 'public key: ' + KEY)
    assert public_key(Path('tool'), Path('key')) == KEY
    monkeypatch.setattr('scripts.publish_windows_update.subprocess.check_output', lambda *a, **k: 'invalid')
    with pytest.raises(ValueError):
        public_key(Path('tool'), Path('key'))


def test_distribution_interface_loads_only_the_explicit_native_path(config_file, monkeypatch):
    from mygpr.interfaces.updates import update_client, update_configuration
    loaded = []
    monkeypatch.setattr('mygpr.infrastructure.updates.winsparkle.ctypes.CDLL', lambda path: loaded.append(path) or native_library())
    assert update_configuration(config_file).public_key == KEY
    assert update_client(Path('absolute-selected.dll')) is not None
    assert loaded == [str(Path('absolute-selected.dll').resolve())]


def test_frozen_process_mutex_is_retained_and_only_created_once(monkeypatch):
    from mygpr.infrastructure.updates import process_mutex
    create = NativeFunction(123)
    monkeypatch.setattr(process_mutex, '_HANDLE', None)
    monkeypatch.setattr(process_mutex, 'sys', SimpleNamespace(platform='linux', frozen=True))
    process_mutex.hold_installer_mutex()
    assert not create.calls
    monkeypatch.setattr(process_mutex, 'sys', SimpleNamespace(platform='win32', frozen=True))
    monkeypatch.setattr(process_mutex.ctypes, 'WinDLL', lambda *a, **kw: SimpleNamespace(CreateMutexW=create), raising=False)
    process_mutex.hold_installer_mutex()
    process_mutex.hold_installer_mutex()
    assert create.calls == [(None, False, r'Local\MyGPR.Application')]
    assert process_mutex._HANDLE == 123
